#version 450
// One half-resolution MRT pass feeds the whole post chain: attachment 0 is
// the bloom bright-pass, attachment 1 is the god-ray occlusion mask.

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outBright;
layout(location = 1) out vec4 outMask;

layout(set = 0, binding = 0) uniform sampler2D sceneTex;
layout(set = 0, binding = 1) uniform sampler2D depthTex;

layout(push_constant) uniform PC {
    float threshold;
    float pad0, pad1, pad2;
} pc;

void main() {
    vec3 c = texture(sceneTex, uv).rgb;
    float lum = max(max(c.r, c.g), c.b);
    outBright = vec4(c * smoothstep(pc.threshold, pc.threshold + 1.0, lum), 1.0);

    // Sky pixels keep the cleared depth (1.0): only they feed the god rays,
    // so geometry occludes the shafts by simply not being sky.
    float depth = texture(depthTex, uv).r;
    outMask = vec4(c * step(0.999999, depth), 1.0);
}
