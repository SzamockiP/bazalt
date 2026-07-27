#version 450

// Samples well outside 0..1, which is where the address mode and the border
// colour are the only things that decide what comes back.

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D tex;

void main() {
    outColor = texture(tex, uv + vec2(4.0, 4.0));
}
