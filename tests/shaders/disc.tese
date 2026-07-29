#version 450

// Tessellation evaluation: interpolate the patch, then push every generated
// vertex out onto a circle of radius 0.8.
//
// This is what makes the test two-sided. The three input corners already sit on
// that circle, so at level 1 — where the tessellator generates nothing but the
// corners — the result is exactly the input triangle. At a higher level the edge
// midpoints exist too, and they get pushed outward, so the triangle fills out
// toward a disc and covers strictly more pixels. A tessellator that silently did
// nothing would paint the same triangle both times.
layout(triangles, equal_spacing, cw) in;

void main() {
    vec4 p = gl_TessCoord.x * gl_in[0].gl_Position
           + gl_TessCoord.y * gl_in[1].gl_Position
           + gl_TessCoord.z * gl_in[2].gl_Position;

    vec2 d = p.xy;
    float len = length(d);
    if (len > 0.0001) {
        d = d / len * 0.8;
    }
    gl_Position = vec4(d, 0.0, 1.0);
}
