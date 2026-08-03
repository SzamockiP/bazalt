#version 450

// A custom resolve: reach the individual samples of a multisampled attachment,
// which is what a sampler2DMS is for. texelFetch takes an integer pixel and a
// sample index — no filtering, no UV, no mip.
//
// Red keeps sample 0 by itself, so it stays hard-edged. Green averages the four
// samples, which is what the driver's own resolve does. The two disagree exactly
// on the pixels the triangle covers partially, and that difference is the proof
// the samples arrived one at a time.

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2DMS msaaTex;

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    float first = texelFetch(msaaTex, coord, 0).r;
    float sum = 0.0;
    for (int s = 0; s < 4; ++s) {
        sum += texelFetch(msaaTex, coord, s).r;
    }
    outColor = vec4(first, sum * 0.25, 0.0, 1.0);
}
