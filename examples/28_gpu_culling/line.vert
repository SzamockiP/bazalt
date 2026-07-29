#version 450

// Plain world-space lines: the frustum wireframe the observer draws so the culled
// wedge has a visible shape.
layout(location = 0) in vec3 inPos;

layout(push_constant) uniform Push {
    mat4 view_proj;
} push;

void main() {
    gl_Position = push.view_proj * vec4(inPos, 1.0);
}
