#version 450
// Screen-space ambient occlusion from the depth buffer alone. View-space
// position comes back through the projection constants, the normal from
// position derivatives — no G-buffer, no extra attachment. Twelve rotated
// Poisson taps in a world-sized hemisphere; the gaussian blur pass smooths
// the grain. The composite multiplies the scene colour by the result.

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D depthTex;

layout(push_constant) uniform PC {
    float tanHalfFov;
    float aspect;
    float near;
    float far;
    float radius;    // world metres
    float strength;
    float pad0, pad1;
} pc;

vec3 view_pos(vec2 t) {
    float d = texture(depthTex, t).r;
    // perspectiveRH_ZO linearization: d=0 at near, d=1 at far.
    float dist = pc.near * pc.far / (pc.far + d * (pc.near - pc.far));
    vec2 ndc = t * 2.0 - 1.0;
    return vec3(ndc.x * pc.aspect * pc.tanHalfFov, -ndc.y * pc.tanHalfFov, -1.0) * dist;
}

void main() {
    const vec2 DISK[12] = vec2[](
        vec2(-0.326, -0.406), vec2(-0.840, -0.074), vec2(-0.696,  0.457),
        vec2(-0.203,  0.621), vec2( 0.962, -0.195), vec2( 0.473, -0.480),
        vec2( 0.519,  0.767), vec2( 0.185, -0.893), vec2( 0.507,  0.064),
        vec2( 0.896,  0.412), vec2(-0.322, -0.933), vec2(-0.792, -0.598));

    vec3 pos = view_pos(uv);
    float dist = -pos.z;
    if (dist > 200.0) {  // sky
        outColor = vec4(1.0);
        return;
    }

    vec3 n = normalize(cross(dFdx(pos), dFdy(pos)));
    if (dot(n, pos) > 0.0) {
        n = -n;  // face the camera whatever the derivative handedness
    }

    float angle = 6.2832 * fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
    mat2 rot = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));

    // The world-space radius projected to uv units at this depth.
    float screenR = pc.radius / dist * (0.5 / pc.tanHalfFov);
    float r2 = pc.radius * pc.radius;

    float occ = 0.0;
    for (int i = 0; i < 12; ++i) {
        vec2 off = rot * DISK[i] * screenR;
        off.x /= pc.aspect;
        vec3 diff = view_pos(uv + off) - pos;
        float d2 = dot(diff, diff);
        // Range check keeps a distant background from occluding a silhouette.
        occ += max(dot(n, diff * inversesqrt(max(d2, 1e-6))) - 0.10, 0.0)
             * (r2 / (r2 + d2));
    }
    float ao = clamp(1.0 - occ * (pc.strength / 12.0), 0.0, 1.0);
    outColor = vec4(vec3(ao), 1.0);
}
