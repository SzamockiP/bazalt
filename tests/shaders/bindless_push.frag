#version 450

// A texture array indexed by a push constant. The index is the same for every
// invocation of the draw, so this is the DYNAMICALLY UNIFORM case and needs no
// nonuniformEXT.

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D textures[4];

layout(push_constant) uniform Push {
    int index;
} pc;

void main() {
    outColor = texture(textures[pc.index], uv);
}
