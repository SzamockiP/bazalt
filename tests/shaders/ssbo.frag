#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout(std430, set = 0, binding = 0) readonly buffer Color {
    vec4 color;
};

void main() {
    outColor = color;
}
