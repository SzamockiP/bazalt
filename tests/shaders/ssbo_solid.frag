#version 450

// ssbo.frag without the uv varying, for pipelines whose last pre-rasterization
// stage is a tessellation evaluation shader that outputs position only.
layout(location = 0) out vec4 outColor;

layout(std430, set = 0, binding = 0) readonly buffer Color {
    vec4 color;
};

void main() {
    outColor = color;
}
