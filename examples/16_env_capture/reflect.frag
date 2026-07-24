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
    outColor = texture(envMap, r);
}
