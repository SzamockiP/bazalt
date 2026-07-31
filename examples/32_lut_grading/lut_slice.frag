#version 450

// One slice of the grading LUT. The framebuffer's (u, v) are the red and green
// inputs, the push constant is the blue input, and the output is where the
// grade sends that colour. Rendering 32 of these through target.layer(z) IS
// the render-to-slice feature.

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
layout(push_constant) uniform Push { float blue; } push;

vec3 grade(vec3 c) {
    // A teal-and-orange look: cool the shadows, warm the highlights, lift the
    // contrast a little.
    float luma = dot(c, vec3(0.299, 0.587, 0.114));
    vec3 shadows = vec3(0.85, 1.0, 1.15);
    vec3 highlights = vec3(1.15, 1.02, 0.85);
    c *= mix(shadows, highlights, smoothstep(0.2, 0.8, luma));
    c = clamp((c - 0.5) * 1.15 + 0.5, 0.0, 1.0);
    return pow(c, vec3(0.95));
}

void main() {
    color = vec4(grade(vec3(uv, push.blue)), 1.0);
}
