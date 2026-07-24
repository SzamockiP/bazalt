#version 450

// Draws vertex-coloured geometry with a pushed view-projection. Used for both
// the six cube-capture passes and the window pass — only the matrix differs.

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inColor;

layout(push_constant) uniform Push { mat4 viewProj; } pc;

layout(location = 0) out vec3 vColor;

void main() {
    vColor = inColor;
    gl_Position = pc.viewProj * vec4(inPos, 1.0);
}
