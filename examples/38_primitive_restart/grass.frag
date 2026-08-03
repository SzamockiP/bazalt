#version 450

// Dark at the root, bright at the tip, so the shape of each triangle in the strip
// is readable without the wireframe.

layout(location = 0) in float height;
layout(location = 0) out vec4 outColor;

void main() {
    vec3 root = vec3(0.05, 0.22, 0.08);
    vec3 tip = vec3(0.45, 0.85, 0.30);
    outColor = vec4(mix(root, tip, height), 1.0);
}
