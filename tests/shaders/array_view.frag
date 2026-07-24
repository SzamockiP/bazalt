#version 450

// Samples one layer of a colour texture ARRAY (push-constant layer index). The
// colour counterpart of depth_array_view.frag — used to read back a specific
// layer of a layered / MSAA-resolved render target.

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2DArray tex;
layout(push_constant) uniform Push { int layer; } pc;

void main() {
    outColor = texture(tex, vec3(uv, float(pc.layer)));
}
