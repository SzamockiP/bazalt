#version 450

// No transform at all: the vertices are already in clip space. The example is
// about which of them form a triangle, so anything else would be noise.

layout(location = 0) in vec2 aPos;

layout(location = 0) out float height;

void main() {
    // 0 at the root, 1 at the tip, for the colour ramp.
    height = clamp((aPos.y + 0.75) / 1.4, 0.0, 1.0);
    gl_Position = vec4(aPos.x, -aPos.y, 0.0, 1.0);
}
