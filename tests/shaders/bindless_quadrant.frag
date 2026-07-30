#version 450
#extension GL_EXT_nonuniform_qualifier : require

// The same texture array, indexed by which quadrant of the framebuffer this
// fragment is in. The index now differs between invocations of one draw, which
// is the NON-UNIFORM case: without nonuniformEXT the result is undefined, and
// without shaderSampledImageArrayNonUniformIndexing the pipeline is illegal.
//
// One draw paints four different textures, which is the whole point of bindless.

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D textures[4];

void main() {
    int index = (uv.x < 0.5 ? 0 : 1) + (uv.y < 0.5 ? 0 : 2);
    outColor = texture(textures[nonuniformEXT(index)], uv);
}
