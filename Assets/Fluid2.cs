// StableFluids - A GPU implementation of Jos Stam's Stable Fluids on Unity
// https://github.com/keijiro/StableFluids

using UnityEngine;

namespace StableFluids
{
    public class Fluid2 : MonoBehaviour
    {
        #region Editable attributes

        [SerializeField] int _resolution = 512;
        [SerializeField] float _viscosity = 1e-6f;
        [SerializeField] float _source = 1.0f;
        [SerializeField] float _sourceDistance = 100;
        [SerializeField] Texture _initial;
        [SerializeField] bool isRunning = true;

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
        }

        int ThreadCountX { get { return (_resolution                                + 7) / 8; } }
        int ThreadCountY { get { return (_resolution * Screen.height / Screen.width + 7) / 8; } }

        int ResolutionX { get { return ThreadCountX * 8; } }
        int ResolutionY { get { return ThreadCountY * 8; } }

        // Vector field buffers
        static class VFB
        {
            public static RenderTexture D1; // density
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
            VFB.D1 = AllocateBuffer(1);
            // VFB.V2 = AllocateBuffer(2);
            // VFB.V3 = AllocateBuffer(2);
            // VFB.P1 = AllocateBuffer(1);
            // VFB.P2 = AllocateBuffer(1);

            _colorRT1 = AllocateBuffer(4, Screen.width, Screen.height);
            _colorRT2 = AllocateBuffer(4, Screen.width, Screen.height);
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
            Destroy(VFB.D1);

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
            Debug.Log(("Setting SourceOrigin to ", input));
            _compute.SetFloat("SourceDistance", _sourceDistance);
            _compute.SetTexture(Kernels.AddSource, "D_out", VFB.D1);
            _compute.SetTexture(Kernels.AddSource, "D_in", VFB.D1);

            if (Input.GetMouseButton(0))
                // Add Source
                _compute.SetFloat("SourceStrength", _source);
            else
                _compute.SetFloat("SourceStrength", 0f);

            _compute.Dispatch(Kernels.AddSource, _resolution, _resolution, 1);

            // // Projection setup
            // _compute.SetTexture(Kernels.PSetup, "W_in", VFB.V3);
            // _compute.SetTexture(Kernels.PSetup, "DivW_out", VFB.V2);
            // _compute.SetTexture(Kernels.PSetup, "P_out", VFB.P1);
            // _compute.Dispatch(Kernels.PSetup, ThreadCountX, ThreadCountY, 1);
            //
            // // Jacobi iteration
            // _compute.SetFloat("Alpha", -dx * dx);
            // _compute.SetFloat("Beta", 4);
            // _compute.SetTexture(Kernels.Jacobi1, "B1_in", VFB.V2);
            //
            // for (var i = 0; i < 20; i++)
            // {
            //     _compute.SetTexture(Kernels.Jacobi1, "X1_in", VFB.P1);
            //     _compute.SetTexture(Kernels.Jacobi1, "X1_out", VFB.P2);
            //     _compute.Dispatch(Kernels.Jacobi1, ThreadCountX, ThreadCountY, 1);
            //
            //     _compute.SetTexture(Kernels.Jacobi1, "X1_in", VFB.P2);
            //     _compute.SetTexture(Kernels.Jacobi1, "X1_out", VFB.P1);
            //     _compute.Dispatch(Kernels.Jacobi1, ThreadCountX, ThreadCountY, 1);
            // }
            //
            // // Projection finish
            // _compute.SetTexture(Kernels.PFinish, "W_in", VFB.V3);
            // _compute.SetTexture(Kernels.PFinish, "P_in", VFB.P1);
            // _compute.SetTexture(Kernels.PFinish, "U_out", VFB.V1);
            // _compute.Dispatch(Kernels.PFinish, ThreadCountX, ThreadCountY, 1);
            //
            // // Apply the velocity field to the color buffer.
            // var offs = Vector2.one * (Input.GetMouseButton(1) ? 0 : 1e+7f);
            // _shaderSheet.SetVector("_ForceOrigin", input + offs);
            // _shaderSheet.SetFloat("_ForceExponent", _exponent);
            // _shaderSheet.SetTexture("_VelocityField", VFB.V1);
            // Graphics.Blit(_colorRT1, _colorRT2, _shaderSheet, 0); // Advect pass
            //
            // // Swap the color buffers.
            // var temp = _colorRT1;
            // _colorRT1 = _colorRT2;
            // _colorRT2 = temp;
            //
            // _previousInput = input;
        }

        void OnRenderImage(RenderTexture source, RenderTexture destination)
        {
            // Graphics.Blit(_colorRT1, destination, _shaderSheet, 1); // render pass
            Graphics.Blit(VFB.D1, destination, _shaderSheet, 1); // render pass
        }

        #endregion
    }
}
