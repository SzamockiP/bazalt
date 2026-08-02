#version 450
#include "frame.glsl"

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec2 uv;
layout(location = 1) flat out uint materialIndex;

void main() {
    gl_Position = u.lightVP * vec4(inPos, 1.0);
    uv = inUV;
    // firstInstance in the indirect command carries the material index and
    // instanceCount is always 0 or 1, so gl_InstanceIndex IS the material.
    materialIndex = uint(gl_InstanceIndex);
}
