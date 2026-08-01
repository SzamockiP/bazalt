#version 450

// The post pass: sample the scene, use its colour AS THE COORDINATE into the
// 3D LUT. That indirection is the whole trick of LUT grading — any colour
// transform, however nonlinear, becomes one filtered texture lookup.

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
layout(binding = 0) uniform sampler2D scene;
layout(binding = 1) uniform sampler3D lut;
layout(push_constant) uniform Push { float split; } push;

void main() {
    vec3 c = texture(scene, uv).rgb;
    // Left of the split shows the original, right of it the graded picture.
    if (uv.x > push.split) {
        c = texture(lut, c).rgb;
    }
    // A thin marker line on the split itself.
    if (abs(uv.x - push.split) < 0.002) {
        c = vec3(1.0);
    }
    color = vec4(c, 1.0);
}
