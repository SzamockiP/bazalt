// Additive, so a million points build up density rather than overwriting each
// other, and the fragment cost stays honest: one blend per point.
#version 450

layout(location = 0) in float depth01;
layout(location = 0) out vec4 colour;

void main() {
    vec3 warm = vec3(1.0, 0.55, 0.2);
    vec3 cool = vec3(0.2, 0.5, 1.0);
    colour = vec4(mix(cool, warm, depth01) * 0.05, 1.0);
}
