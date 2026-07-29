#version 450

// Tessellation control: decide how finely each patch is subdivided.
//
// This is the stage that makes adaptive LOD possible, and it is the reason a
// fixed vertex buffer cannot do the same job: the triangle count is chosen per
// frame, on the GPU, from where the camera happens to be.
layout(vertices = 4) out;

layout(push_constant) uniform Push {
    vec4 camera_lod;  // xyz = camera position, w = LOD multiplier
    vec4 height_time; // x = height scale, y = time
} push;

// The level for one edge, derived from that EDGE'S MIDPOINT and not from the
// patch centre. That detail is the whole difference between a terrain and a
// terrain full of cracks: two neighbouring patches share an edge, so they must
// agree on how many times to split it. A midpoint is shared; a patch centre is
// not, so centre-based levels leave gaps along every seam where the two sides
// disagreed.
float level_for(vec3 a, vec3 b) {
    float d = distance(0.5 * (a + b), push.camera_lod.xyz);
    return clamp(push.camera_lod.w * 24.0 / max(d, 0.5), 1.0, 32.0);
}

void main() {
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;

    // The levels are per patch, so one invocation writes them.
    if (gl_InvocationID == 0) {
        vec3 p0 = gl_in[0].gl_Position.xyz;
        vec3 p1 = gl_in[1].gl_Position.xyz;
        vec3 p2 = gl_in[2].gl_Position.xyz;
        vec3 p3 = gl_in[3].gl_Position.xyz;

        gl_TessLevelOuter[0] = level_for(p3, p0);
        gl_TessLevelOuter[1] = level_for(p0, p1);
        gl_TessLevelOuter[2] = level_for(p1, p2);
        gl_TessLevelOuter[3] = level_for(p2, p3);

        // The interior follows the outside it has to meet.
        gl_TessLevelInner[0] = max(gl_TessLevelOuter[1], gl_TessLevelOuter[3]);
        gl_TessLevelInner[1] = max(gl_TessLevelOuter[0], gl_TessLevelOuter[2]);
    }
}
