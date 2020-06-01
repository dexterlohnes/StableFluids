// StableFluids - A GPU implementation of Jos Stam's Stable Fluids on Unity
// https://github.com/keijiro/StableFluids

using System;
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
        [SerializeField] float _tempAdd = 1.0f;
        [SerializeField] float _force = 10.0f;
        [SerializeField] float _sourceDistance = 5;
        [SerializeField] float _forceDistance = 2;
        [SerializeField] Texture _initial;
        [SerializeField] bool isRunning = true;
        [SerializeField] private float _velocityDiffusion = 10;
        [SerializeField] private float _vorticityConfinement = 10f;
        [SerializeField] private Mode mode;

        #endregion
        
        #region Enums

        enum Mode
        {
            Density,
            Temp,
            Fuel,
            Velocity
        }
        #endregion
        
        #region Properties

        public void SetInitial(RenderTexture texture)
        {
            _initial = texture;
        }
        
        #endregion

        #region Internal resources

        [SerializeField] ComputeShader _compute;
        [SerializeField] ComputeShader _curl; // vorticity confinement
        [SerializeField] Shader _shader;

        #endregion

        #region Private members

        Material _shaderSheet;
        Vector2 _previousInput;

        static class Kernels
        {
            // Fluid compute
            public const int AddSource = 0;
            public const int Diffusion = 1;
            public const int Advection = 2;
            public const int CreateVelocityField = 3;
            public const int ProjectionSetup = 4;
            public const int Projection = 5;
            public const int ProjectionFinish = 6;
            
            // Vorticity confinement compute
            public const int VorticityConfinement = 0;
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
            
            public static RenderTexture F1; // Fuel
            public static RenderTexture F2; 
            public static RenderTexture F3; 
            
            public static RenderTexture T1; // Temperature - creates buoyancy in velocity field
            public static RenderTexture T2;
            public static RenderTexture T3;
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
            
            VFB.T1 = AllocateBuffer(1);
            VFB.T2 = AllocateBuffer(1);
            VFB.T3 = AllocateBuffer(1);
            
            // Fuel has two channels 
            // R channel represents solid state of fuel, which is what is rendered
            // G channel represents gaseous state of fuel, which is what actually ignites
            // R -> G via simulated / fudged process of pyrolysis
            // Fuel density determines basically how much G is stored per unit of R
            VFB.F1 = AllocateBuffer(2);
            VFB.F2 = AllocateBuffer(2);
            VFB.F3 = AllocateBuffer(2);

            _colorRT1 = AllocateBuffer(4, Screen.width, Screen.height);
            _colorRT2 = AllocateBuffer(4, Screen.width, Screen.height);
            
            // Set up our velocity field 
            _compute.SetTexture(Kernels.CreateVelocityField, "V_in", VFB.V1);
            _compute.Dispatch(Kernels.CreateVelocityField, ThreadCountX, ThreadCountY, 1);
            _compute.SetTexture(Kernels.CreateVelocityField, "V_in", VFB.V2);
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
            Destroy(VFB.T1);
            Destroy(VFB.T2);
            Destroy(VFB.T3);
            Destroy(VFB.F1);
            Destroy(VFB.F2);
            Destroy(VFB.F3);
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

            if (Input.GetKeyDown(KeyCode.F))
            {
                mode = Mode.Fuel;
            }
            else if (Input.GetKeyDown(KeyCode.T))
            {
                mode = Mode.Temp;
            }
            else if (Input.GetKeyDown(KeyCode.D))
            {
                mode = Mode.Density;
            } else if (Input.GetKeyDown(KeyCode.V))
            {
                mode = Mode.Velocity;
            }

            // Common variables
            _compute.SetFloat("Time", Time.time);
            _compute.SetFloat("DeltaTime", dt);

            UpdateTemp(input, dt);

            UpdateFuel(input, dt);
            
            UpdateDensity(input, dt);

            UpdateVelocity(input, dt);

            EndUpdate();
        }

        private static void EndUpdate()
        {
            RenderTexture temp;
            // End
            temp = VFB.D1;
            VFB.D1 = VFB.D2;
            VFB.D2 = temp;
            
            temp = VFB.T1;
            VFB.T1 = VFB.T2;
            VFB.T2 = temp;

            // Graphics.CopyTexture(VFB.V1, VFB.V2);
            Graphics.CopyTexture(VFB.V2, VFB.V1);
        }
        
        private void UpdateFuel(Vector2 input, float dt)
        {
            // Add source - temperature
            _compute.SetVector("SourceOrigin", input);
            _compute.SetFloat("SourceDistance", _sourceDistance);
            _compute.SetTexture(Kernels.AddSource, "D_out", VFB.F2); // F2 will hold source-added fuel
            _compute.SetTexture(Kernels.AddSource, "D_in", VFB.F1); // F1 is our previous ending fuel

            if (mode == Mode.Fuel && Input.GetMouseButton(0))
                // Add Source
                _compute.SetFloat("SourceStrength", _source);
            else
                _compute.SetFloat("SourceStrength", 0f);

            _compute.Dispatch(Kernels.AddSource, ThreadCountX, ThreadCountY, 1);
            
            // For each cell of fuel, if the temperature is high enough to cause combustion there,
            // we cause combustion (do we need to check if combustion has already started somehow?)
            // Combustion should spread naturally as a result of flame
        }
        
        private void UpdateTemp(Vector2 input, float dt)
        {
            // Add source - temperature
            _compute.SetVector("SourceOrigin", input);
            _compute.SetFloat("SourceDistance", _sourceDistance);
            _compute.SetTexture(Kernels.AddSource, "D_out", VFB.T2); // T2 will hold source-added temp
            _compute.SetTexture(Kernels.AddSource, "D_in", VFB.T1); // T1 is our previous ending temp

            if (mode == Mode.Temp && Input.GetMouseButton(0))
                // Add Source
                _compute.SetFloat("SourceStrength", _tempAdd);
            else
                _compute.SetFloat("SourceStrength", 0f);

            _compute.Dispatch(Kernels.AddSource, ThreadCountX, ThreadCountY, 1);

            // Diffusion
            Graphics.CopyTexture(VFB.T2, VFB.T1); // Copy so that we have source established in both textures
            Graphics.CopyTexture(VFB.T2, VFB.T3);
            _compute.SetTexture(Kernels.Diffusion, "D_in", VFB.T1); // pre-diffusion
            _compute.SetFloat("Alpha", dt * _diffusion * _resolution * _resolution); // pre-diffusion
            _compute.SetFloat("Beta", 1f / (1f + 4f * (dt * _diffusion * _resolution * _resolution))); // pre-diffusion

            for (var i = 0; i < 10; i++)
            {
                _compute.SetTexture(Kernels.Diffusion, "D_out", VFB.T3);
                _compute.SetTexture(Kernels.Diffusion, "D_gsbuff", VFB.T2);
                _compute.Dispatch(Kernels.Diffusion, ThreadCountX, ThreadCountY, 1);

                _compute.SetTexture(Kernels.Diffusion, "D_out", VFB.T2); // D2 will be post-diffusion
                _compute.SetTexture(Kernels.Diffusion, "D_gsbuff", VFB.T3); // GaussSeidel buffer
                _compute.Dispatch(Kernels.Diffusion, ThreadCountX, ThreadCountY, 1);
            }

            // Advection step - move density along vector field
            Graphics.CopyTexture(VFB.T2, VFB.T3);
            _compute.SetTexture(Kernels.Advection, "V_in", VFB.V1);
            _compute.SetTexture(Kernels.Advection, "D_in", VFB.T3);
            _compute.SetTexture(Kernels.Advection, "D_out", VFB.T2);
            _compute.Dispatch(Kernels.Advection, ThreadCountX, ThreadCountY, 1);
        }

        private void UpdateVelocity(Vector2 input, float dt)
        {
            // Vorticity confinement
            // Graphics.CopyTexture(VFB.V2, VFB.V1);
            //
            // _curl.SetFloat( "DeltaTime", dt);
            // _curl.SetFloat( "Vorticity", _vorticityConfinement);
            // _curl.SetTexture(Kernels.VorticityConfinement, "V_in", VFB.V1);
            // _curl.SetTexture(Kernels.VorticityConfinement, "V_out", VFB.V2);
            // _curl.Dispatch(Kernels.VorticityConfinement, ThreadCountX, ThreadCountY, 1);

            // Simulate source set 
            _compute.SetVector("SourceOrigin", input);
            // Debug.Log(("Setting SourceOrigin to ", input));
            _compute.SetFloat("SourceDistance", _forceDistance);
            _compute.SetTexture(Kernels.AddSource, "D_out", VFB.V2);
            _compute.SetTexture(Kernels.AddSource, "D_in", VFB.V1);

            if (Input.GetMouseButton(1))
                // Add Source
                _compute.SetFloat("SourceStrength", _force);
            else
                _compute.SetFloat("SourceStrength", 0f);

            _compute.Dispatch(Kernels.AddSource, ThreadCountX, ThreadCountY, 1);

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
        }

        private void UpdateDensity(Vector2 input, float dt)
        {
            // Add source
            _compute.SetVector("SourceOrigin", input);
            // Debug.Log(("Setting SourceOrigin to ", input));
            _compute.SetFloat("SourceDistance", _sourceDistance);
            _compute.SetTexture(Kernels.AddSource, "D_out", VFB.D2); // D2 will hold source-added density
            _compute.SetTexture(Kernels.AddSource, "D_in", VFB.D1); // D1 is our previous ending density

            if (mode == Mode.Density && Input.GetMouseButton(0))
                // Add Source
                _compute.SetFloat("SourceStrength", _source);
            else
                _compute.SetFloat("SourceStrength", 0f);

            // _compute.Dispatch(Kernels.AddSource, _resolution, _resolution, 1);
            _compute.Dispatch(Kernels.AddSource, ThreadCountX, ThreadCountY, 1);


            // Diffusion
            Graphics.CopyTexture(VFB.D2, VFB.D1); // Copy so that we have source established in both textures
            Graphics.CopyTexture(VFB.D2, VFB.D3);
            _compute.SetTexture(Kernels.Diffusion, "D_in", VFB.D1); // pre-diffusion
            _compute.SetFloat("Alpha", dt * _diffusion * _resolution * _resolution); // pre-diffusion
            _compute.SetFloat("Beta", 1f / (1f + 4f * (dt * _diffusion * _resolution * _resolution))); // pre-diffusion

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
        }

        void OnRenderImage(RenderTexture source, RenderTexture destination)
        {
            switch (mode)
            {
                case Mode.Density:
                    Graphics.Blit(VFB.D2, destination, _shaderSheet, 1); // render pass
                    break;
                case Mode.Temp:
                    Graphics.Blit(VFB.T2, destination, _shaderSheet, 2); // render pass
                    break;
                case Mode.Fuel:
                    Graphics.Blit(VFB.F2, destination, _shaderSheet, 1); // render pass
                    break;
                case Mode.Velocity:
                    Graphics.Blit(VFB.V2, destination, _shaderSheet, 1); // render pass
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        #endregion
    }
}
