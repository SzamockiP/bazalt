// A texture and a sampler, each annotated the way the compile_shader docstring
// tells an HLSL author to annotate a resource. glslang emits that pair as a
// SAMPLED_IMAGE at binding 0 plus a SAMPLER at binding 1, and bazalt has no
// declarator for a lone sampler — so before 0.29 the pipeline named a binding
// nothing could declare. Folded, both halves are the one COMBINED_IMAGE_SAMPLER
// at binding 0 that .texture() declares.

[[vk::binding(0, 0)]] Texture2D tex;
[[vk::binding(1, 0)]] SamplerState samp;

float4 main(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target0 {
    return tex.Sample(samp, uv);
}
