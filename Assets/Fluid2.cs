// StableFluids - A GPU implementation of Jos Stam's Stable Fluids on Unity
// https://github.com/keijiro/StableFluids

using UnityEngine;

namespace StableFluids
{
    public class Fluid2 : MonoBehaviour
    {
        #region Editable attributes

        [SerializeField] int _resolution = 512;
        // [SerializeField] float _viscosity = 1e-6f;
        [SerializeField] private float _diffusion = 1000;
        [SerializeField] float _source = 1.0f;
        [SerializeField] float _sourceDistance = 100;
        [SerializeField] Texture _initial;
        [SerializeField] bool isRunning = true;
        [SerializeField] private float _velocityDiffusion = 10;

        #endregion
        
        #region Properties

        public void SetInitial(RenderTexture texture)
        {
            _initial = texture;
        }
        
        #endregion

        #region Internal resources

        [SerializeField] ComputeShader _compute;
        [SerializeField] Shader _shader;

        #endregion

        #region Private members

        Material _shaderSheet;
        Vector2 _previousInput;

        static class Kernels
        {
            public const int AddSource = 0;
            public const int Diffusion = 1;
            public const int Advection = 2;
            public const int CreateVelocityField = 3;
            public const int ProjectionSetup = 4;
            public const int Projection = 5;
            public const int ProjectionFinish = 6;
        }

        int ThreadCountX { get { return (_resolution                                + 7) / 8; } }
        int ThreadCountY { get { return (_resolution * Screen.height / Screen.width + 7) / 8; } }

        int ResolutionX { get { return ThreadCountX * 8; } }
        int ResolutionY { get { return ThreadCountY * 8; } }

        // Vector field buffers
        static class VFB
        {
            public static RenderTexture D0; // density - ending density from previous frame
            public static RenderTexture D1; // density - usually for beginning of frame
            public static RenderTexture D2; // density - end of frame / final density
            public static RenderTexture D3; // density - buffer for GaussSeidel
            public static RenderTexture V0; // velocity field - RG corresponds to XY velocity components
            public static RenderTexture V1; // velocity field - RG corresponds to XY velocity components
            public static RenderTexture V2; // used for diffusion and for height field
            public static RenderTexture V3; // addtl buffer for Gauss Seidel diffusion
        }

        // Color buffers (for double buffering)
        RenderTexture _colorRT1;
        RenderTexture _colorRT2;

        RenderTexture AllocateBuffer(int componentCount, int width = 0, int height = 0)
        {
            var format = RenderTextureFormat.ARGBHalf;
            if (componentCount == 1) format = RenderTextureFormat.RHalf;
            if (componentCount == 2) format = RenderTextureFormat.RGHalf;

            if (width  == 0) width  = ResolutionX;
            if (height == 0) height = ResolutionY;

            var rt = new RenderTexture(width, height, 0, format);
            rt.enableRandomWrite = true;
            rt.Create();
            return rt;
        }

        #endregion

        #region MonoBehaviour implementation

        void OnValidate()
        {
            _resolution = Mathf.Max(_resolution, 8);
        }

        void Start()
        {
            _shaderSheet = new Material(_shader);

            AllocateTextureBuffers();

            #if UNITY_IOS
            Application.targetFrameRate = 60;
            #endif

            BeginSimulation();
        }

        private void AllocateTextureBuffers()
        {
            VFB.D0 = AllocateBuffer(1);
            VFB.D1 = AllocateBuffer(1);
            VFB.D2 = AllocateBuffer(1);
            VFB.D3 = AllocateBuffer(1);
            VFB.V0 = AllocateBuffer(2);
            VFB.V1 = AllocateBuffer(2);
            VFB.V2 = AllocateBuffer(2);
            VFB.V3 = AllocateBuffer(2);
            // VFB.V1.enableRandomWrite = true;
            // VFB.V3 = AllocateBuffer(2);
            // VFB.P1 = AllocateBuffer(1);
            // VFB.P2 = AllocateBuffer(1);

            _colorRT1 = AllocateBuffer(4, Screen.width, Screen.height);
            _colorRT2 = AllocateBuffer(4, Screen.width, Screen.height);
            
            
            // Set up our velocity field 
            _compute.SetTexture(Kernels.CreateVelocityField, "V_in", VFB.V1);
            _compute.Dispatch(Kernels.CreateVelocityField, ThreadCountX, ThreadCountY, 1);
        }

        public void BeginSimulation()
        {
            Graphics.Blit(_initial, _colorRT1);
            var comparer = FindObjectOfType<TextureComparer>();
            comparer.currentTexture = _colorRT1;
            isRunning = true;
        }

        void OnDestroy()
        {
            Destroy(_shaderSheet);

            DestroyTextures();
        }

        private void DestroyTextures()
        {
            Destroy(VFB.D0);
            Destroy(VFB.D1);
            Destroy(VFB.D2);
            Destroy(VFB.D3);
            Destroy(VFB.V0);
            Destroy(VFB.V1);
            Destroy(VFB.V2);
            Destroy(VFB.V3);

            Destroy(_colorRT1);
            Destroy(_colorRT2);
        }

        void Update()
        {
            if (!isRunning) return;
            
            var dt = Time.deltaTime;
            var dx = 1.0f / ResolutionY;

            // Input point
            var input = new Vector2(
                (Input.mousePosition.x - Screen.width  * 0.5f) / Screen.width,
                (Input.mousePosition.y - Screen.height * 0.5f) / Screen.height
            );

            // Common variables
            _compute.SetFloat("Time", Time.time);
            _compute.SetFloat("DeltaTime", dt);


            // Add source
            _compute.SetVector("SourceOrigin", input);
            // Debug.Log(("Setting SourceOrigin to ", input));
            _compute.SetFloat("SourceDistance", _sourceDistance);
            _compute.SetTexture(Kernels.AddSource, "D_out", VFB.D2); // D2 will hold source-added density
            _compute.SetTexture(Kernels.AddSource, "D_in", VFB.D1); // D1 is our previous ending density

            if (Input.GetMouseButton(0))
                // Add Source
                _compute.SetFloat("SourceStrength", _source);
            else
                _compute.SetFloat("SourceStrength", 0f);

            _compute.Dispatch(Kernels.AddSource, _resolution, _resolution, 1);
            
            
            // Diffusion
            Graphics.CopyTexture(VFB.D2, VFB.D1); // Copy so that we have source established in both textures
            Graphics.CopyTexture(VFB.D2, VFB.D3);
            _compute.SetTexture(Kernels.Diffusion, "D_in", VFB.D1); // pre-diffusion
            _compute.SetFloat("Alpha", dt * _diffusion * _resolution * _resolution); // pre-diffusion
            _compute.SetFloat("Beta", 1f / (1f + 4f * (dt * _diffusion * _resolution * _resolution))); // pre-diffusion
                // / (1. + 4.*Alpha)
            for (var i = 0; i < 10; i++)
            {
                _compute.SetTexture(Kernels.Diffusion, "D_out", VFB.D3); 
                _compute.SetTexture(Kernels.Diffusion, "D_gsbuff", VFB.D2);
                _compute.Dispatch(Kernels.Diffusion, ThreadCountX, ThreadCountY, 1);

                _compute.SetTexture(Kernels.Diffusion, "D_out", VFB.D2); // D2 will be post-diffusion
                _compute.SetTexture(Kernels.Diffusion, "D_gsbuff", VFB.D3); // GaussSeidel buffer
                _compute.Dispatch(Kernels.Diffusion, ThreadCountX, ThreadCountY, 1);
            }
            
            // Advection step - move density along vector field
            Graphics.CopyTexture(VFB.D2, VFB.D0);
            _compute.SetTexture(Kernels.Advection, "V_in", VFB.V1);
            _compute.SetTexture(Kernels.Advection, "D_in", VFB.D0);
            _compute.SetTexture(Kernels.Advection, "D_out", VFB.D2);
            _compute.Dispatch(Kernels.Advection, ThreadCountX, ThreadCountY, 1);
            
            
            // Simulate source set 
            _compute.SetVector("SourceOrigin", input);
            // Debug.Log(("Setting SourceOrigin to ", input));
            _compute.SetFloat("SourceDistance", _sourceDistance);
            _compute.SetTexture(Kernels.AddSource, "D_out", VFB.V2);
            _compute.SetTexture(Kernels.AddSource, "D_in", VFB.V1);
            
            if (Input.GetMouseButton(1))
                // Add Source
                _compute.SetFloat("SourceStrength", _source);
            else
                _compute.SetFloat("SourceStrength", 0f);
            
            _compute.Dispatch(Kernels.AddSource, _resolution, _resolution, 1);
            
            // Graphics.CopyTexture(VFB.V1, VFB.V2); // V2 ends up being our source-added texture
            
            // Diffusion
            Graphics.CopyTexture(VFB.V2, VFB.V3);
            _compute.SetTexture(Kernels.Diffusion, "D_out", VFB.V1); // pre-diffusion
            _compute.SetFloat("Alpha", dt * _velocityDiffusion * _resolution * _resolution); // pre-diffusion
            _compute.SetFloat("Beta", 1f / (1f + 4 * (dt * _velocityDiffusion * _resolution * _resolution))); // pre-diffusion
            
            for (var i = 0; i < 10; i++)
            {
                _compute.SetTexture(Kernels.Diffusion, "D_out", VFB.V3); 
                _compute.SetTexture(Kernels.Diffusion, "D_gsbuff", VFB.V2);
                _compute.Dispatch(Kernels.Diffusion, ThreadCountX, ThreadCountY, 1);
            
                _compute.SetTexture(Kernels.Diffusion, "D_out", VFB.V2); // D2 will be post-diffusion
                _compute.SetTexture(Kernels.Diffusion, "D_gsbuff", VFB.V3); // GaussSeidel buffer
                _compute.Dispatch(Kernels.Diffusion, ThreadCountX, ThreadCountY, 1);
            }
            
            
            // Projection Setup
            _compute.SetFloat("h", 1f / VFB.V2.height);
            _compute.SetTexture(Kernels.ProjectionSetup, "V_in", VFB.V2);
            _compute.SetTexture(Kernels.ProjectionSetup, "H_out", VFB.V3);
            _compute.Dispatch(Kernels.ProjectionSetup, ThreadCountX, ThreadCountY, 1);
            
            Graphics.CopyTexture(VFB.V3, VFB.V1);
            
            // Projection - Guass Seidel Relaxation
            for (var i = 0; i < 10; i++)
            {
                _compute.SetTexture(Kernels.Projection, "H_in", VFB.V1); // pre-diffusion
                _compute.SetTexture(Kernels.Projection, "H_out", VFB.V3); // pre-diffusion
                _compute.Dispatch(Kernels.Projection, ThreadCountX, ThreadCountY, 1);
                Graphics.CopyTexture(VFB.V3, VFB.V1);

                _compute.SetTexture(Kernels.Projection, "H_in", VFB.V3); // pre-diffusion
                _compute.SetTexture(Kernels.Projection, "H_out", VFB.V1); // pre-diffusion
                _compute.Dispatch(Kernels.Projection, ThreadCountX, ThreadCountY, 1);
                Graphics.CopyTexture(VFB.V1, VFB.V3);
            }
            // V1 is our real H_out at this point
            
            // Projection finish
            _compute.SetTexture(Kernels.ProjectionFinish, "H_in", VFB.V1);
            _compute.SetTexture(Kernels.ProjectionFinish, "V_out", VFB.V2);
            _compute.Dispatch(Kernels.ProjectionFinish, ThreadCountX, ThreadCountY, 1);
            
            
            // Advection step - move density along vector field
            Graphics.CopyTexture(VFB.V2, VFB.V0);
            Graphics.CopyTexture(VFB.V2, VFB.V1);
            _compute.SetTexture(Kernels.Advection, "V_in", VFB.V1);
            _compute.SetTexture(Kernels.Advection, "D_in", VFB.V0);
            _compute.SetTexture(Kernels.Advection, "D_out", VFB.V2);
            _compute.Dispatch(Kernels.Advection, ThreadCountX, ThreadCountY, 1);

            
            // Projection Setup
            _compute.SetTexture(Kernels.ProjectionSetup, "V_in", VFB.V2);
            _compute.SetTexture(Kernels.ProjectionSetup, "H_out", VFB.V3);
            _compute.Dispatch(Kernels.ProjectionSetup, ThreadCountX, ThreadCountY, 1);
            
            Graphics.CopyTexture(VFB.V3, VFB.V1);
            
            // Projection - Guass Seidel Relaxation
            for (var i = 0; i < 10; i++)
            {
                _compute.SetTexture(Kernels.Projection, "H_in", VFB.V1); // pre-diffusion
                _compute.SetTexture(Kernels.Projection, "H_out", VFB.V3); // pre-diffusion
                _compute.Dispatch(Kernels.Projection, ThreadCountX, ThreadCountY, 1);
                Graphics.CopyTexture(VFB.V3, VFB.V1);

                _compute.SetTexture(Kernels.Projection, "H_in", VFB.V3); // pre-diffusion
                _compute.SetTexture(Kernels.Projection, "H_out", VFB.V1); // pre-diffusion
                _compute.Dispatch(Kernels.Projection, ThreadCountX, ThreadCountY, 1);
                Graphics.CopyTexture(VFB.V1, VFB.V3);
            }
            // V1 is our real H_out at this point
            
            // Projection finish
            _compute.SetFloat("h", VFB.V2.height);
            _compute.SetTexture(Kernels.ProjectionFinish, "H_in", VFB.V1);
            _compute.SetTexture(Kernels.ProjectionFinish, "V_out", VFB.V2);
            _compute.Dispatch(Kernels.ProjectionFinish, ThreadCountX, ThreadCountY, 1);
            
            
            // End
            var temp = VFB.D1;
            VFB.D1 = VFB.D2;
            VFB.D2 = temp;
            
            Graphics.CopyTexture(VFB.V2, VFB.V1);
            // Graphics.CopyTexture(VFB.V2, VFB.V0);


            // temp = VFB.V1;
            // VFB.V1 = VFB.V2;
            // VFB.V2 = temp;
        }

        void OnRenderImage(RenderTexture source, RenderTexture destination)
        {
            // Graphics.Blit(_colorRT1, destination, _shaderSheet, 1); // render pass
            Graphics.Blit(VFB.D2, destination, _shaderSheet, 1); // render pass
            // Graphics.Blit(VFB.V2, destination, _shaderSheet, 1); // render pass
        }

        #endregion
    }
}
