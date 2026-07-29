#version 450

// The instance position comes out of the buffer the compute pass compacted, NOT
// out of a per-instance vertex binding. That is the difference an indirect draw
// makes: the CPU cannot fill a per-instance binding without knowing how many
// instances survived, and it does not know.
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;

layout(std430, set = 0, binding = 0) readonly buffer Visible {
    vec4 visible[];
};

layout(push_constant) uniform Push {
    mat4 view_proj;
} push;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec3 vTint;

void main() {
    vec4 sphere = visible[gl_InstanceIndex];
    vec3 world = sphere.xyz + inPos * sphere.w;

    vNormal = inNormal;

    // Colour derived from the cube's own POSITION, never from gl_InstanceIndex.
    //
    // The slot a survivor lands in comes from atomicAdd, so it depends on the order
    // the workgroups happened to finish in and is different every frame even when
    // nothing moved. Tinting by the slot makes all of them strobe — which looks
    // like a broken draw and is really just the compaction being unordered. The
    // position is stable, so each cube keeps its colour and the only thing that
    // changes is which cubes are inside the frustum.
    float t = dot(sphere.xyz, vec3(0.07, 0.11, 0.13));
    vTint = 0.45 + 0.55 * vec3(sin(t), sin(t + 2.1), sin(t + 4.2));

    gl_Position = push.view_proj * vec4(world, 1.0);
}
