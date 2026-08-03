#version 450

// Painted only where the stencil counter came out non-zero, which is where the
// near plane is INSIDE the volume. The stencil test does the deciding; this
// shader only says what that looks like.

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

void main() {
    float vignette = 1.0 - 0.6 * length(uv - 0.5);
    outColor = vec4(vec3(0.95, 0.45, 0.15) * vignette, 1.0);
}
