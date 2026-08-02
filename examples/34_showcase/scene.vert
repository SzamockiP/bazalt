#version 450
#include "frame.glsl"

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 worldPos;
layout(location = 1) out vec3 normal;
layout(location = 2) out vec2 uv;
layout(location = 3) out vec4 lightSpacePos;
layout(location = 4) flat out uint materialIndex;

void main() {
    gl_Position = u.viewProj * vec4(inPos, 1.0);
    worldPos = inPos;
    normal = inNormal;
    uv = inUV;
    // The same matrix the shadow pass rasterized with (example 09's
    // contract), sampled a couple of centimetres along the normal — the
    // normal-offset trick kills the acne the depth bias alone leaves on
    // slopes (~2 texels of the 30 m shadow window).
    lightSpacePos = u.lightVP * vec4(inPos + inNormal * 0.03, 1.0);
    materialIndex = uint(gl_InstanceIndex);
}
