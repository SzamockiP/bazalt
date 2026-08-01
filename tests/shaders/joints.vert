#version 450

// A skinning-shaped vertex layout: FLOAT2 position + UINT4 joint indices
// (location 1, `in uvec4` — an integer attribute, no conversion). The indices
// come out as a colour so a readback proves they arrived intact.

layout(location = 0) in vec2 pos;
layout(location = 1) in uvec4 joints;
layout(location = 0) out vec4 color;

void main() {
    color = vec4(joints) / 255.0;
    gl_Position = vec4(pos, 0.0, 1.0);
}
