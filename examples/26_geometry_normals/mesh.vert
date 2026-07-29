#version 450

// One vertex shader for both pipelines. It hands on the WORLD position and
// normal as well as writing gl_Position, because the two consumers want
// different things: the shading pass reads the varyings per fragment, and the
// geometry pass needs the world-space values so it can build a line of a length
// measured in world units and project it itself.
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;

layout(set = 0, binding = 0) uniform Camera {
    mat4 view_proj;
    mat4 model;
} cam;

layout(location = 0) out vec3 vWorldPos;
layout(location = 1) out vec3 vWorldNormal;

void main() {
    vec4 world = cam.model * vec4(inPos, 1.0);
    vWorldPos = world.xyz;
    vWorldNormal = normalize(mat3(cam.model) * inNormal);
    gl_Position = cam.view_proj * world;
}
