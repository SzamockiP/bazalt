#version 450

// The thing tessellation cannot do: change the primitive type. One point in,
// four vertices of a triangle strip out.
//
// This is the two-sided test for the geometry stage. The same POINT_LIST draw
// without this shader paints one pixel per point (gl_PointSize is 1); with it,
// each point becomes a quad. Nothing else in bazalt can turn a point into a
// surface, which is the argument for keeping the stage at all.
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

void main() {
    vec2 c = gl_in[0].gl_Position.xy;
    const float s = 0.25;

    gl_Position = vec4(c + vec2(-s, -s), 0.0, 1.0);
    EmitVertex();
    gl_Position = vec4(c + vec2(+s, -s), 0.0, 1.0);
    EmitVertex();
    gl_Position = vec4(c + vec2(-s, +s), 0.0, 1.0);
    EmitVertex();
    gl_Position = vec4(c + vec2(+s, +s), 0.0, 1.0);
    EmitVertex();
    EndPrimitive();
}
