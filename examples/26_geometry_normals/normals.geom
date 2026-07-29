#version 450

// Triangles in, LINES out. This is the geometry stage's own trick and the reason
// it survives next to tessellation: tessellation subdivides a patch and always
// produces the same kind of primitive it was given, so it cannot turn a surface
// into a set of lines at all.
//
// Three input vertices become three two-vertex line strips, each running from the
// surface along its normal. Nothing else in bazalt can draw this from the same
// vertex buffer as the shaded mesh -- the alternative is a second buffer built on
// the CPU and kept in sync by hand.
layout(triangles) in;
layout(line_strip, max_vertices = 6) out;

layout(set = 0, binding = 0) uniform Camera {
    mat4 view_proj;
    mat4 model;
} cam;

layout(push_constant) uniform Push {
    float normal_length;
} push;

layout(location = 0) in vec3 vWorldPos[];
layout(location = 1) in vec3 vWorldNormal[];

void main() {
    for (int i = 0; i < 3; ++i) {
        gl_Position = cam.view_proj * vec4(vWorldPos[i], 1.0);
        EmitVertex();
        gl_Position = cam.view_proj * vec4(vWorldPos[i] + vWorldNormal[i] * push.normal_length, 1.0);
        EmitVertex();
        // One strip per normal. Without this the six vertices would join up into
        // one zigzag across the whole triangle.
        EndPrimitive();
    }
}
