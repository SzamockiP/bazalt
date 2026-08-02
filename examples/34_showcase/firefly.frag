#version 450

layout(location = 0) in float brightness;
layout(location = 0) out vec4 outColor;

void main() {
    vec2 d = gl_PointCoord * 2.0 - 1.0;
    float falloff = max(1.0 - dot(d, d), 0.0);
    // Additive blend: the colour IS the contribution. Bright enough that the
    // bloom pass picks the sprites up and they glow for free.
    outColor = vec4(vec3(1.0, 0.85, 0.35) * (brightness * falloff * falloff * 4.0), 1.0);
}
