#version 450

// One texture for everything ImGui draws. The font atlas carries the glyphs in
// its alpha channel and a white block in one corner, and every solid rectangle
// samples that block — so a panel and a letter go through the same draw call.

layout(location = 0) in vec2 uv;
layout(location = 1) in vec4 color;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D atlas;

void main() {
    outColor = color * texture(atlas, uv);
}
