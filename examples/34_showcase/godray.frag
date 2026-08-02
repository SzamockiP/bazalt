#version 450
// Screen-space light shafts: march from each pixel toward the sun's screen
// position, accumulating the sky mask with exponential decay. Intensity
// arrives premodulated by the sun's visibility, so the pass fades out on its
// own when the sun leaves the frame or sets.

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D maskTex;

layout(push_constant) uniform PC {
    vec2  sunPos;    // sun in uv space
    float intensity;
    float decay;
} pc;

void main() {
    const int TAPS = 64;
    vec2 delta = (pc.sunPos - uv) / float(TAPS);
    vec2 p = uv;
    vec3 acc = vec3(0.0);
    float weight = 1.0;
    for (int i = 0; i < TAPS; ++i) {
        p += delta;
        acc += texture(maskTex, p).rgb * weight;
        weight *= pc.decay;
    }
    outColor = vec4(acc * (pc.intensity / float(TAPS)), 1.0);
}
