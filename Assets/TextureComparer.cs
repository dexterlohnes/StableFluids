﻿using System;
using UnityEngine;

public class TextureComparer : MonoBehaviour
{
    
    [SerializeField] ComputeShader _comparer;
    [SerializeField] private Texture originalTexture;
    [SerializeField] public Texture currentTexture;
    [SerializeField] public RenderTexture bufferTex;

    private Int32[] data;
    private ComputeBuffer buffer;
    
    
    private float[] colors;
    private ComputeBuffer colorsBuffer;
    
    static class Kernels
    {
        public const int Compare = 0;
    }


    public void SetOriginalTexture(Texture tex)
    {
        originalTexture = tex;
        data = new Int32[originalTexture.height * originalTexture.width / 10];
        buffer?.Dispose();
        buffer = new ComputeBuffer(data.Length, 4);
        
    }

    void Update()
    {
        if(Input.GetKeyDown(KeyCode.C))
        {
            Debug.Log("Here we are");
            DoCompare();
        }
    }

    void DoCompare()
    {
        Debug.Log("Doing comparison");
        
        //INITIALIZE DATA HERE

        buffer.SetData(data);
        _comparer.SetBuffer(Kernels.Compare, "compareResult", buffer);
        _comparer.SetTexture(Kernels.Compare, "OrigTex", originalTexture);
        _comparer.SetTexture(Kernels.Compare, "CurTex", currentTexture);
        _comparer.Dispatch(Kernels.Compare, originalTexture.width, originalTexture.height,1);
        buffer.GetData(data);

        Int32 numChanged = 0;
        Int32 numUnchanged = 0;


        for (var i = 0; i < data.Length; i++)
        {
            if (data[i] > 0)
            {
                numChanged++;
            }
            else
            {
                numUnchanged++;
            }
        }
        
        Debug.Log($"Changed {numChanged} out of {numChanged + numUnchanged} pixels");
    }

    private void OnDestroy()
    {
        buffer.Dispose();
    }
}
