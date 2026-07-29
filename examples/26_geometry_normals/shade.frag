#version 450

layout(location = 0) in vec3 vWorldPos;
layout(location = 1) in vec3 vWorldNormal;

layout(location = 0) out vec4 outColor;

void main() {
    const vec3 light = normalize(vec3(0.35, 0.8, 0.5));
    float diffuse = max(dot(normalize(vWorldNormal), light), 0.0);
    vec3 albedo = vec3(0.28, 0.45, 0.62);
    outColor = vec4(albedo * (0.2 + 0.8 * diffuse), 1.0);
}
