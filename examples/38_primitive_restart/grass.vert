#version 450

// A blade sways more the further it is from its root, which is what makes the
// strip visibly one connected ribbon of triangles rather than a stack of quads.

layout(location = 0) in vec2 aPos;
layout(location = 1) in float aBlade;

layout(push_constant) uniform Push {
    float time;
    float sway;
} pc;

layout(location = 0) out float height;

void main() {
    // 0 at the root, 1 at the tip. The bases all sit at y = -0.8.
    height = clamp((aPos.y + 0.8) / 1.5, 0.0, 1.0);
    float bend = height * height * pc.sway * sin(pc.time * 1.7 + aBlade * 2.1);
    gl_Position = vec4(aPos.x + bend, -aPos.y, 0.0, 1.0);
}
