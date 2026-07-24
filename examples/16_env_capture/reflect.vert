#version 450

// The mirror object. Push carries the MVP and the world-space camera position;
// the view direction (and thus everything the reflection needs) is computed here
// and handed to the fragment shader, so the fragment stage needs no push block —
// ShaderStage is single-stage in bazalt, one push range per stage.

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;

layout(push_constant) uniform Push {
    mat4 mvp;
    vec4 camPos;
} pc;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec3 vViewDir;

void main() {
    vNormal = inNormal;
    vViewDir = inPos - pc.camPos.xyz;  // object is at the origin, so world pos == inPos
    gl_Position = pc.mvp * vec4(inPos, 1.0);
}
