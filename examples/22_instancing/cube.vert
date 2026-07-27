#version 450

// Binding 0 is per vertex, binding 1 is per instance. The locations continue
// across the two: the mesh takes 0 and 1, so the instance data starts at 2.

layout(location = 0) in vec3 position;   // per vertex
layout(location = 1) in vec3 normal;     // per vertex
layout(location = 2) in vec3 offset;     // per instance
layout(location = 3) in float scale;     // per instance
layout(location = 4) in vec4 tint;       // per instance (4 bytes, not 4 floats)

layout(set = 0, binding = 0) uniform Camera {
    mat4 view_proj;
    float time;
} cam;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec4 v_tint;

void main() {
    // Every cube bobs on its own phase, so the field is alive without a single
    // per-instance write from the CPU each frame.
    float bob = sin(cam.time + offset.x * 0.7 + offset.z * 0.4) * 0.35;
    vec3 world = position * scale + offset + vec3(0.0, bob, 0.0);

    v_normal = normal;
    v_tint = tint;
    gl_Position = cam.view_proj * vec4(world, 1.0);
}
