#version 450

layout(location = 0) in vec3 fragNormal;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 n = normalize(fragNormal);
    float light = 0.30 + 0.70 * max(dot(n, normalize(vec3(0.4, 0.8, 0.5))), 0.0);
    outColor = vec4(vec3(0.95, 0.55, 0.25) * light, 1.0);
}
