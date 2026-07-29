#version 450

// Tessellation evaluation: runs once per vertex the fixed-function tessellator
// generated, and decides where that vertex goes. This is where displacement
// happens -- the input buffer is a flat grid, and the hills exist only here.
layout(quads, fractional_odd_spacing, cw) in;

layout(set = 0, binding = 0) uniform Camera {
    mat4 view_proj;
} cam;

layout(push_constant) uniform Push {
    vec4 camera_lod;
    vec4 height_time; // x = height scale, y = time
} push;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out float outHeight;

float height(vec2 p) {
    float t = push.height_time.y;
    float h = sin(p.x * 0.7 + t * 0.3) * cos(p.y * 0.6 - t * 0.2);
    h += 0.5 * sin(p.x * 1.7 - t * 0.15) * cos(p.y * 1.9);
    h += 0.25 * sin((p.x + p.y) * 3.1);
    return h * push.height_time.x;
}

void main() {
    // Bilinear across the quad patch. gl_TessCoord is the (u, v) the tessellator
    // handed this vertex; the four control points are the patch corners.
    vec3 a = mix(gl_in[0].gl_Position.xyz, gl_in[1].gl_Position.xyz, gl_TessCoord.x);
    vec3 b = mix(gl_in[3].gl_Position.xyz, gl_in[2].gl_Position.xyz, gl_TessCoord.x);
    vec3 p = mix(a, b, gl_TessCoord.y);

    p.y = height(p.xz);

    // The normal comes from finite differences of the same function, not from the
    // input mesh. A normal carried through from the flat grid would light the
    // terrain as if it were still flat, which is the usual way a displacement
    // demo ends up looking wrong for a reason that is not the displacement.
    const float e = 0.05;
    float hx = height(p.xz + vec2(e, 0.0)) - height(p.xz - vec2(e, 0.0));
    float hz = height(p.xz + vec2(0.0, e)) - height(p.xz - vec2(0.0, e));
    outNormal = normalize(vec3(-hx, 2.0 * e, -hz));
    outHeight = p.y;

    gl_Position = cam.view_proj * vec4(p, 1.0);
}
