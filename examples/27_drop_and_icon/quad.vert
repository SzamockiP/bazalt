#version 450

// A quad from gl_VertexIndex, so there is no vertex buffer to keep in sync with
// whatever image was dropped.
layout(push_constant) uniform Push {
    mat4 mvp;
} push;

layout(location = 0) out vec2 uv;

void main() {
    // Two triangles: 0,1,2 and 2,1,3 expressed as six indices.
    const vec2 corners[6] = vec2[6](
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
        vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(1.0, 1.0)
    );
    uv = corners[gl_VertexIndex];
    gl_Position = push.mvp * vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
