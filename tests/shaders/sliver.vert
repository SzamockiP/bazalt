#version 450

// A triangle far smaller than one pixel of a 16x16 target, tucked into a corner
// of pixel (8, 8) so that it misses that pixel's centre. NDC 0.125 is one pixel
// there, the pixel spans 0.0 to 0.125 on both axes and its centre sits at
// 0.0625, and nothing here reaches past 0.045.
//
// Ordinary rasterization asks whether the primitive covers the sample point, so
// this triangle produces NO fragment at all. ConservativeRaster.OVERESTIMATE
// asks whether it touches the pixel, so the same triangle produces one. That
// difference is the whole subject of test_conservative_raster.py.
//
// No vertex buffer: these three positions are the constants the test's
// arithmetic depends on, so they belong beside the reasoning above.

void main() {
    const vec2 corners[3] = vec2[3](vec2(0.005, 0.005), vec2(0.045, 0.005), vec2(0.005, 0.045));
    gl_Position = vec4(corners[gl_VertexIndex], 0.0, 1.0);
}
