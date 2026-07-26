#version 450

layout(location = 0) in vec3 fragNormal;

// Each window pushes its own tint, so one pipeline and one mesh still produce
// two visibly different views.
layout(push_constant) uniform Tint {
    vec3 colour;
} tint;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 n = normalize(fragNormal);
    float light = 0.35 + 0.65 * max(dot(n, normalize(vec3(0.4, 0.8, 0.5))), 0.0);
    outColor = vec4(tint.colour * light, 1.0);
}
