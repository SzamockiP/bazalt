#version 450

// Something with knobs worth turning. Every value here comes from a widget in
// the overlay, which is the point of the example: the picture is the thing being
// tuned, and the UI is how you tune it without editing a file.

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform Push {
    vec4 tint;
    float time;
    float scale;
    float speed;
    float warp;
} pc;

void main() {
    vec2 p = (uv - 0.5) * pc.scale;
    float t = pc.time * pc.speed;

    // Two rotating wave fields, one warped by the other.
    float a = sin(p.x * 3.0 + t);
    float b = sin(p.y * 3.0 - t * 0.7);
    float warped = sin((p.x + p.y) * 2.0 + pc.warp * (a + b) + t * 0.3);

    float v = 0.5 + 0.5 * (a * b + warped) * 0.5;
    outColor = vec4(pc.tint.rgb * v, 1.0);
}
