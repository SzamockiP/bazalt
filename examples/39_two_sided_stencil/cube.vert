#version 450

layout(location = 0) in vec3 aPos;

layout(push_constant) uniform Push {
    mat4 view_proj;
} pc;

void main() {
    gl_Position = pc.view_proj * vec4(aPos, 1.0);
}
