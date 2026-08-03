#version 450

// Three ways to look at the same multisampled image.
//
// The driver's resolve (mode 0) reads the single-sample image the pass resolved
// into: `sampler2D`, ordinary filtering, one colour per pixel. The other two
// read the multisampled image itself: `sampler2DMS`, texelFetch, an integer
// pixel and a sample index. That is the whole difference the feature buys.

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D resolved;
layout(set = 0, binding = 1) uniform sampler2DMS samples;

layout(push_constant) uniform Push {
    int mode;
    int sampleCount;
} pc;

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);

    if (pc.mode == 0) {
        outColor = texture(resolved, uv);
        return;
    }

    float first = texelFetch(samples, coord, 0).r;
    if (pc.mode == 1) {
        // One sample, so the edges come back hard: this is the picture MSAA is
        // hiding, drawn from inside the multisampled buffer.
        outColor = vec4(first, first, first, 1.0);
        return;
    }

    // How much the samples disagree. Non-zero only on a silhouette, which makes
    // it a map of every pixel the hardware resolve is actually working on — the
    // input a custom resolve (per-sample shading, a TAA history reject) keys off.
    float disagreement = 0.0;
    for (int s = 1; s < pc.sampleCount; ++s) {
        disagreement += abs(texelFetch(samples, coord, s).r - first);
    }
    disagreement /= float(max(pc.sampleCount - 1, 1));
    outColor = vec4(disagreement, disagreement * 0.35, 1.0 - disagreement, 1.0);
}
