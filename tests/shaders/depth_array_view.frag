#version 450

// Visualizes one layer of a depth texture ARRAY: raw depth becomes grayscale.
// The layer to sample comes from a push constant — this is how a test proves
// render-to-layer routed geometry into the layer it asked for and no other.

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2DArray depthTex;
layout(push_constant) uniform Push { int layer; } pc;

void main() {
    float d = texture(depthTex, vec3(uv, float(pc.layer))).r;
    outColor = vec4(d, d, d, 1.0);
}
