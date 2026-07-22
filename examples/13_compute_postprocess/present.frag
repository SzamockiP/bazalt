#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

// The compute-generated image, sampled. bazalt inserts the
// GENERAL -> SHADER_READ_ONLY transition between the dispatch and this sample.
layout(set = 0, binding = 0) uniform sampler2D tex;

void main() {
    outColor = texture(tex, uv);
}
