#version 450

// Dark at the root, bright at the tip. The three blades are the same colour on
// purpose: any difference you see is the topology, not the shading.

layout(location = 0) in float height;
layout(location = 0) out vec4 outColor;

void main() {
    vec3 root = vec3(0.06, 0.24, 0.09);
    vec3 tip = vec3(0.48, 0.88, 0.32);
    outColor = vec4(mix(root, tip, height), 1.0);
}
