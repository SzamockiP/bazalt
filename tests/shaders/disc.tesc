#version 450

// Tessellation control: pass the three patch corners through and set the
// subdivision level from a push constant, so ONE pipeline can render both sides
// of the two-sided test. That the level arrives as push data is the point — it
// proves the fixed-function tessellator is really reading what this stage wrote.
layout(vertices = 3) out;

layout(push_constant) uniform Push {
    float level;
} push;

void main() {
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;

    // The levels are per patch, not per control point.
    if (gl_InvocationID == 0) {
        gl_TessLevelOuter[0] = push.level;
        gl_TessLevelOuter[1] = push.level;
        gl_TessLevelOuter[2] = push.level;
        gl_TessLevelInner[0] = push.level;
    }
}
