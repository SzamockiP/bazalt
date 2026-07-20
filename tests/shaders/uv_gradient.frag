#version 450

// GLSL reference for the HLSL pair (fullscreen_vs.hlsl + uv_gradient_ps.hlsl):
// same fullscreen triangle, same gradient — the two must render identically.

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(uv, 0.25, 1.0);
}
