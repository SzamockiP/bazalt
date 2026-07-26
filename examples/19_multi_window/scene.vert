#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(set = 0, binding = 0) uniform Camera {
    mat4 mvp;
} cam;

layout(location = 0) out vec3 fragNormal;

void main() {
    gl_Position = cam.mvp * vec4(inPosition, 1.0);
    fragNormal = inNormal;
}
