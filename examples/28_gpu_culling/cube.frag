#version 450

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vTint;

layout(location = 0) out vec4 outColor;

void main() {
    const vec3 light = normalize(vec3(0.4, 0.85, 0.35));
    float diffuse = max(dot(normalize(vNormal), light), 0.0);
    outColor = vec4(vTint * (0.25 + 0.75 * diffuse), 1.0);
}
