// HLSL twin of fullscreen.vert — the vertex order (and therefore the winding)
// is ported VERBATIM: glslang's HLSL frontend does not invert Y, so coordinates
// carry over unchanged and the same order stays front-facing under the pipeline
// default (cull BACK, front face COUNTER_CLOCKWISE).

struct VSOut {
    float4 pos : SV_Position;
    float2 uv  : TEXCOORD0;
};

VSOut main(uint id : SV_VertexID) {
    VSOut o;
    o.uv  = float2(id & 2, (id << 1) & 2);
    o.pos = float4(o.uv * 2.0 - 1.0, 0.0, 1.0);
    return o;
}
