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
    // Colour by slot so the compaction is visible: the survivors are packed from 0
    // upwards, so the palette shifts as the camera turns rather than each cube
    // keeping one colour.
    float t = float(gl_InstanceIndex) * 0.13;
    vTint = 0.5 + 0.5 * vec3(sin(t), sin(t + 2.1), sin(t + 4.2));

    gl_Position = push.view_proj * vec4(world, 1.0);
}
