#version 450

// Depth-only shadow pass, once per cascade: transform by that cascade's
// light view-projection (pushed). No fragment shader — the target is a
// depth-only layer of the shadow array.

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;  // unused; the vertex format carries it

layout(push_constant) uniform Push { mat4 lightVP; } pc;

void main() {
    gl_Position = pc.lightVP * vec4(inPos, 1.0);
}
