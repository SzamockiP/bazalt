#version 450

// The ungraded footage: an animated plasma with a full range of colours, so
// the grade has something to bite on.

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
layout(push_constant) uniform Push { float time; } push;

void main() {
    float t = push.time * 0.6;
    vec2 p = uv * 4.0;
    float a = sin(p.x + t) + sin(p.y + t * 1.3);
    float b = sin(length(p - vec2(2.0 + sin(t), 2.0 + cos(t))) * 2.0);
    vec3 c = 0.5 + 0.5 * cos(vec3(a + b) + vec3(0.0, 2.1, 4.2));
    color = vec4(c, 1.0);
}
