#version 450

// The mirror object. Push carries the MVP and the world-space camera position.
// We hand the fragment shader the world position and normal so it can do a
// box-projected (parallax-corrected) cubemap lookup — ShaderStage is single-stage
// in bazalt, so the fragment stage needs no push block of its own.

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;

layout(push_constant) uniform Push {
    mat4 mvp;
    vec4 camPos;
} pc;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec3 vViewDir;
layout(location = 2) out vec3 vWorldPos;

void main() {
    vNormal = inNormal;
    vWorldPos = inPos;                 // object is at the origin, so world pos == inPos
    vViewDir = inPos - pc.camPos.xyz;
    gl_Position = pc.mvp * vec4(inPos, 1.0);
}
