// One file, both stages — the case entry_point= exists for. HLSL has no rule
// that the entry point is called main, and a shader pack written for D3D
// normally puts VSMain and PSMain side by side.
//
// The geometry and the gradient are the same as fullscreen_vs.hlsl plus
// uv_gradient_ps.hlsl, so a test can compare against those and prove that
// choosing the entry point changed which function got compiled, not just that
// something compiled.

struct VSOut {
    float4 pos : SV_Position;
    float2 uv  : TEXCOORD0;
};

VSOut VSMain(uint id : SV_VertexID) {
    VSOut o;
    o.uv  = float2(id & 2, (id << 1) & 2);
    o.pos = float4(o.uv * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

float4 PSMain(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target0 {
    return float4(uv, 0.25, 1.0);
}
