#version 450

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec4 v_tint;
layout(location = 0) out vec4 out_color;

void main() {
    // One fixed light, so the faces read as a solid and the instance tint is
    // still the thing you see.
    float light = 0.35 + 0.65 * max(dot(normalize(v_normal), normalize(vec3(0.4, 0.8, 0.5))), 0.0);
    out_color = vec4(v_tint.rgb * light, 1.0);
}
