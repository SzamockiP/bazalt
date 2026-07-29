#version 450

// A quad from gl_VertexIndex, so there is no vertex buffer to keep in sync with
// whatever image was dropped.
layout(push_constant) uniform Push {
    mat4 mvp;
} push;

layout(location = 0) out vec2 uv;

void main() {
    // Two triangles over the unit square.
    const vec2 corners[6] = vec2[6](
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
        vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(1.0, 1.0)
    );
    vec2 c = corners[gl_VertexIndex];

    // The position and the texture coordinate need OPPOSITE y, and reusing one
    // value for both is why this drew every image upside down.
    //
    // A texture's v = 0 is its top row. This quad goes through a projection matrix
    // built with proj[1][1] *= -1, so world +y ends up pointing UP on screen — and
    // then position.y = 0 is the BOTTOM of the quad. tests/shaders/fullscreen.vert
    // can use one value for both only because it writes clip space directly and
    // never meets that flip.
    uv = vec2(c.x, 1.0 - c.y);
    gl_Position = push.mvp * vec4(c * 2.0 - 1.0, 0.0, 1.0);
}
