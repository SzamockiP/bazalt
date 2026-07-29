#version 450

// The vertex shader of a tessellation pipeline does almost nothing: it hands
// control points to the tessellation control stage, and the real geometry is
// generated after it. The buffer holds only the corners of flat quad patches on
// the xz plane -- 4 vertices per patch, no height and no normals.
layout(location = 0) in vec2 inXZ;

void main() {
    gl_Position = vec4(inXZ.x, 0.0, inXZ.y, 1.0);
}
