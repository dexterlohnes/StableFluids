using StableFluids;
using UnityEngine;

public class TextureGrabber : MonoBehaviour
{
    [SerializeField] private bool shouldGrab = false;

    public RenderTexture texture;
    
    // Start is called before the first frame update
    void Start()
    {
        texture = new RenderTexture(Screen.width, Screen.height, 1, RenderTextureFormat.ARGBHalf);
        GetComponent<Camera>().targetTexture = texture;
    }

    private void OnPostRender()
    {
        if (shouldGrab)
        {
            var fluid = FindObjectOfType<Fluid>();
            var comparer = FindObjectOfType<TextureComparer>();
            comparer.SetOriginalTexture(texture);
            fluid.SetInitial(texture);
            fluid.BeginSimulation();
            shouldGrab = false;
        }
    }
}
