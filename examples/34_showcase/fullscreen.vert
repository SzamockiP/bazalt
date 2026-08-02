#version 450
// One triangle covers the screen: draw(3), no vertex buffer. Every post pass
// and the sky share this stage; uv spans 0..1 over the visible area.
layout(location = 0) out vec2 uv;

void main() {
    // Bit order matters: this winding is front-facing under the default
    // cull BACK + COUNTER_CLOCKWISE state (same as examples 13/14).
    uv = vec2(gl_VertexIndex & 2, (gl_VertexIndex << 1) & 2);
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
