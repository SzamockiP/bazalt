#version 450
#include "frame.glsl"
// Procedural sky: a Rayleigh-flavoured gradient by day, stars and a moon by
// night, and an HDR-bright sun disc. The disc's brightness is the point — the
// bloom threshold catches it and the god-ray mask sees it through sky pixels,
// so "bloom from the sun" needs no extra plumbing. Hot-reload friendly:
// edit this file while the demo runs.

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PC {
    vec4 right;   // camera basis (xyz)
    vec4 up;
    vec4 forward;
    vec4 params;  // x: tan(fov/2), y: aspect
} pc;

float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

vec3 sky_day(vec3 dir, vec3 sun) {
    float h = clamp(dir.y, 0.0, 1.0);
    vec3 col = mix(vec3(0.60, 0.71, 0.85), vec3(0.11, 0.26, 0.58), pow(h, 0.6));
    // Warmth pools at the horizon while the sun is low.
    float lowSun = 1.0 - clamp(sun.y * 4.0, 0.0, 1.0);
    col = mix(col, vec3(1.00, 0.48, 0.22), lowSun * pow(1.0 - h, 3.0) * 0.85);
    // Mie-ish glow around the sun direction.
    col += vec3(1.0, 0.75, 0.45) * pow(max(dot(dir, sun), 0.0), 24.0) * 0.35;
    return col;
}

vec3 sky_night(vec3 dir, vec3 moon, float time) {
    float h = clamp(dir.y, 0.0, 1.0);
    vec3 col = mix(vec3(0.020, 0.028, 0.055), vec3(0.004, 0.007, 0.018), pow(h, 0.5));
    // Stars: one hash per direction cell, thresholded, twinkling slowly.
    float star = hash13(floor(dir * 220.0));
    if (star > 0.997 && dir.y > 0.02) {
        float twinkle = 0.7 + 0.3 * sin(time * 3.0 + star * 400.0);
        col += vec3(smoothstep(0.997, 1.0, star)) * twinkle;
    }
    // Moon disc with a soft halo.
    float cosMoon = dot(dir, moon);
    col += vec3(0.90, 0.95, 1.00) * smoothstep(0.99970, 0.99995, cosMoon) * 2.0;
    col += vec3(0.25, 0.30, 0.45) * pow(max(cosMoon, 0.0), 64.0) * 0.15;
    return col;
}

void main() {
    vec2 ndc = uv * 2.0 - 1.0;
    // World-space ray from the camera basis — Vulkan NDC y points down, so
    // negate it to get world-up (example 14's convention).
    vec3 dir = normalize(pc.forward.xyz
                       + ndc.x * pc.params.y * pc.params.x * pc.right.xyz
                       - ndc.y * pc.params.x * pc.up.xyz);

    float night = u.lightColor.w;
    vec3 col = mix(sky_day(dir, u.sunDir.xyz),
                   sky_night(dir, -u.sunDir.xyz, u.ambient.w),
                   night);

    // The HDR sun disc.
    float cosSun = dot(dir, u.sunDir.xyz);
    col += vec3(1.0, 0.9, 0.7) * smoothstep(0.99980, 0.99995, cosSun)
         * u.sunDir.w * (1.0 - night);

    outColor = vec4(col, 1.0);
}
