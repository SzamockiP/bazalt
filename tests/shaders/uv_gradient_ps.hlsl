// Twin of uv_gradient.frag. The gradient (not a solid color) makes the image
// comparison sensitive to Y orientation, not just to "did anything draw".

float4 main(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target0 {
    return float4(uv, 0.25, 1.0);
}
