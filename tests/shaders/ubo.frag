#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout(std140, set = 0, binding = 0) uniform Color {
    vec4 color;
};

void main() {
    outColor = color;
}
