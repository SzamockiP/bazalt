#version 450

// A fullscreen triangle at z = -0.5, i.e. BEHIND the near plane. Without
// depth_clamp the primitive is clipped away entirely; with it the depth is
// clamped into 0..1 and the triangle still rasterizes. That difference is the
// whole test.

layout(location = 0) out vec2 uv;

void main() {
    uv = vec2(gl_VertexIndex & 2, (gl_VertexIndex << 1) & 2);
    gl_Position = vec4(uv * 2.0 - 1.0, -0.5, 1.0);
}
