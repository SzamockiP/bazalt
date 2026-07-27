#version 450

// One shader for both passes. `inflate` is 0 for the object itself and a small
// positive number for the outline pass, which pushes every vertex along its
// normal — the silhouette grows, and the stencil test keeps only the ring that
// was not covered the first time.

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(set = 0, binding = 0) uniform Camera {
    mat4 view_proj;
    mat4 model;
} cam;

layout(push_constant) uniform Push {
    vec4 color;
    float inflate;
} pc;

layout(location = 0) out vec3 v_normal;

void main() {
    vec3 inflated = position + normal * pc.inflate;
    v_normal = mat3(cam.model) * normal;
    gl_Position = cam.view_proj * cam.model * vec4(inflated, 1.0);
}
