#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

// The streamed texture. Its pixels are rewritten from numpy every frame by
// image.update(); nothing here knows or cares.
layout(set = 0, binding = 0) uniform sampler2D tex;

void main() {
    outColor = texture(tex, uv);
}
