#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;

layout(push_constant) uniform Push { mat4 cameraVP; } pc;

layout(location = 0) out vec3 vWorld;
layout(location = 1) out vec3 vNormal;

void main() {
    vWorld = inPos;       // model is identity in this demo
    vNormal = inNormal;
    gl_Position = pc.cameraVP * vec4(inPos, 1.0);
}
