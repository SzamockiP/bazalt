#version 450
// Separable gaussian — the direction arrives premultiplied by the texel size,
// so ONE shader serves both the horizontal and the vertical pass.

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D src;

layout(push_constant) uniform PC {
    vec2 dir; // texel-scaled step
    vec2 pad;
} pc;

void main() {
    const float w[5] = float[](0.227027, 0.194594, 0.121622, 0.054054, 0.016216);
    vec3 acc = texture(src, uv).rgb * w[0];
    for (int i = 1; i < 5; ++i) {
        acc += texture(src, uv + pc.dir * float(i)).rgb * w[i];
        acc += texture(src, uv - pc.dir * float(i)).rgb * w[i];
    }
    outColor = vec4(acc, 1.0);
}
