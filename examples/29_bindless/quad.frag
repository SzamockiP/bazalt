#version 450
#extension GL_EXT_nonuniform_qualifier : require

// One binding, eight textures. Which one this fragment reads comes from its
// instance, so the index differs inside a single draw — that is what
// nonuniformEXT says, and leaving it out is undefined behaviour rather than a
// compile error.

layout(location = 0) in vec2 uv;
layout(location = 1) flat in uint texIndex;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D textures[8];

void main() {
    outColor = texture(textures[nonuniformEXT(texIndex)], uv);
}
