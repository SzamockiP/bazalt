// The points, straight out of the storage buffer the PREVIOUS frame's dispatch
// wrote. No vertex buffer: gl_VertexIndex is the index into it, which is the
// same trick the fullscreen triangle uses.
#version 450

layout(set = 0, binding = 0) readonly buffer State { vec4 points[]; };
layout(push_constant) uniform Push { mat4 view_proj; } pc;

layout(location = 0) out float depth01;

void main() {
    vec4 state = points[gl_VertexIndex];
    gl_Position = pc.view_proj * vec4(state.xyz, 1.0);
    depth01 = clamp(state.z * 0.5 + 0.5, 0.0, 1.0);
    gl_PointSize = 1.0;
}
