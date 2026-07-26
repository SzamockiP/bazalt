#version 450

// fullscreen.vert at z = 0.9 instead of 0.0, so two passes (or two draws) can
// disagree about depth and the test can say which one won. Kept as a second
// file rather than a push constant because push.frag already claims the one
// push-constant range, and this needs no new binding.

layout(location = 0) out vec2 uv;

void main() {
    uv = vec2(gl_VertexIndex & 2, (gl_VertexIndex << 1) & 2);
    gl_Position = vec4(uv * 2.0 - 1.0, 0.9, 1.0);
}
