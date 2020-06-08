// StableFluids - A GPU implementation of Jos Stam's Stable Fluids on Unity
// https://github.com/keijiro/StableFluids

Shader "Hidden/StableFluids2"
{
    Properties
    {
        _MainTex("", 2D) = ""
        _VelocityField("", 2D) = ""
        _SecondaryTex("", 2D) = ""
    }

    CGINCLUDE

    #include "UnityCG.cginc"

    sampler2D _SecondaryTex;
    sampler2D _MainTex;
    float4 _MainTex_TexelSize;

    sampler2D _VelocityField;

    float2 _ForceOrigin;
    float _ForceExponent;

    half4 frag_advect(v2f_img i) : SV_Target
    {
        // Time parameters
        float time = _Time.y;
        float deltaTime = unity_DeltaTime.x;

        // Aspect ratio coefficients
        float2 aspect = float2(_MainTex_TexelSize.y * _MainTex_TexelSize.z, 1);
        float2 aspect_inv = float2(_MainTex_TexelSize.x * _MainTex_TexelSize.w, 1);

        // Color advection with the velocity field
        float2 delta = tex2D(_VelocityField, i.uv).xy * aspect_inv * deltaTime;
        float3 color = tex2D(_MainTex, i.uv - delta).xyz;

        // Dye (injection color)
        float3 dye = saturate(sin(time * float3(2.72, 5.12, 4.98)) + 0.5);

        // Blend dye with the color from the buffer.
        float2 pos = (i.uv - 0.5) * aspect;
        float amp = exp(-_ForceExponent * distance(_ForceOrigin, pos));
        color = lerp(color, dye, saturate(amp * 100));

        return half4(color, 1);
    }

    half4 frag_render(v2f_img i) : SV_Target
    {
        half3 rgb = tex2D(_MainTex, i.uv).rgb;

        // Mixing channels up to get slowly changing false colors
        //rgb = sin(float3(3.43, 4.43, 3.84) * rgb +
        //          float3(0.12, 0.23, 0.44) * _Time.y) * 0.5 + 0.5;

        return half4(rgb, 1);
    }
    
    half4 frag_render_temp(v2f_img i) : SV_Target
    {
        half temp = tex2D(_MainTex, i.uv).r * 10000.;
        
        //half3 color = half3(0., 0., 0.);
        half3 color = tex2D(_SecondaryTex, i.uv).rgb;
        
        half3 black550 = half3(0.2, 0.11, 0.);
        half3 red1490 = half3(0.76, 0.09, 0.08);
        half3 yellow2372 = half3(1., 0.91, 0.68);
        
        half blacktemp = 1.0;

        if(temp > 2372.) {
            color = yellow2372;
        } else if(temp > 1490.) {
            color = lerp(red1490, yellow2372, (temp - 1490.) / (2372. - 1490.));
        } else if(temp > 550.) {
            color = lerp(black550, red1490, (temp - 550.) / (1490. - 550.));
        } else {
            color = lerp(color, black550, temp / 550.);
        }

        // Mixing channels up to get slowly changing false colors
        //rgb = sin(half3(3.43, 4.43, 3.84) * rgb +
        //          half3(0.12, 0.23, 0.44) * _Time.y) * 0.5 + 0.5;

        return half4(color, 1);
    }
    
    ENDCG

    SubShader
    {
        Cull Off ZWrite Off ZTest Always
        Pass
        {
            CGPROGRAM
            #pragma vertex vert_img
            #pragma fragment frag_advect
            ENDCG
        }
        Pass
        {
            CGPROGRAM
            #pragma vertex vert_img
            #pragma fragment frag_render
            ENDCG
        }
        Pass
        {
            CGPROGRAM
            #pragma vertex vert_img
            #pragma fragment frag_render_temp
            ENDCG
        }
    }
}
