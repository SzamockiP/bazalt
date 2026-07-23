#version 450

// Samples a texture at an explicit mip level pushed as a constant. Used to
// inspect a generated mip chain: a big lod clamps to the smallest level.

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D tex;

layout(push_constant) uniform Push {
    float lod;
} pc;

void main() {
    outColor = textureLod(tex, uv, pc.lod);
}
