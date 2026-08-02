#version 450
#extension GL_EXT_nonuniform_qualifier : require
// A fragment shader on a depth-only target is legal and here it earns its
// keep: without the discard, every leaf would cast the shadow of its full
// quad. No colour outputs — only the depth test survives this stage.

layout(location = 0) in vec2 uv;
layout(location = 1) flat in uint materialIndex;

layout(set = 1, binding = 0) uniform sampler2D materials[];

void main() {
    if (texture(materials[nonuniformEXT(materialIndex)], uv).a < 0.5) {
        discard;
    }
}
