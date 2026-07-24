#version 450

// Samples the captured environment cubemap along the reflection vector. Both the
// reflection vector and the cubemap are in world space (the cube was captured
// from the object's centre), so they line up.

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vViewDir;

layout(set = 0, binding = 0) uniform samplerCube envMap;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 r = reflect(normalize(vViewDir), normalize(vNormal));
    vec3 reflection = texture(envMap, r).rgb;
    // A constant base tint so the cube reads as a distinct (copper-tinted mirror)
    // object against the walls it reflects, instead of blending into them.
    vec3 base = vec3(0.75, 0.45, 0.2);
    outColor = vec4(mix(base, reflection, 0.6), 1.0);
}
