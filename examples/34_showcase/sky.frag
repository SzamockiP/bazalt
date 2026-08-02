#version 450
#include "frame.glsl"
// Procedural sky: a Rayleigh-flavoured gradient, fbm clouds that drift and
// catch the sunset, stars and a moon by night, and an HDR-bright sun disc.
// The disc's brightness is the point — the bloom threshold catches it and
// the god-ray mask sees it through sky pixels, so "bloom from the sun" needs
// no extra plumbing. Clouds mix in AFTER the disc, so they dim it and the
// god rays react to them. Hot-reload friendly: edit this file while the demo
// runs.

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PC {
    vec4 right;   // camera basis (xyz)
    vec4 up;
    vec4 forward;
    vec4 params;  // x: tan(fov/2), y: aspect
} pc;

float hash12(vec2 p) {
    vec3 q = fract(vec3(p.xyx) * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

vec3 hash33(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.xxy + p.yxx) * p.zyx);
}

float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 s = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash12(i), hash12(i + vec2(1, 0)), s.x),
               mix(hash12(i + vec2(0, 1)), hash12(i + vec2(1, 1)), s.x), s.y);
}

float fbm(vec2 p) {
    float acc = 0.0;
    float amp = 0.5;
    for (int i = 0; i < 4; ++i) {
        acc += vnoise(p) * amp;
        p = p * 2.13 + 17.0;
        amp *= 0.5;
    }
    return acc;
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
    // Stars: one jittered point per cell, round through a distance falloff
    // (a thresholded cell alone is a square the size of the cell).
    vec3 sp = dir * 500.0;
    vec3 cell = floor(sp);
    vec3 rnd = hash33(cell);
    if (rnd.z > 0.85 && dir.y > 0.02) {
        vec3 centre = cell + 0.5 + (rnd - 0.5) * 0.5;
        float d = length(sp - centre);
        float twinkle = 0.75 + 0.25 * sin(time * 3.0 + rnd.x * 40.0);
        col += vec3(smoothstep(0.5, 0.0, d)) * (twinkle * smoothstep(0.85, 1.0, rnd.z));
    }
    // Moon disc with a soft halo.
    float cosMoon = dot(dir, moon);
    col += vec3(0.90, 0.95, 1.00) * smoothstep(0.99970, 0.99995, cosMoon) * 2.0;
    col += vec3(0.25, 0.30, 0.45) * pow(max(cosMoon, 0.0), 64.0) * 0.15;
    return col;
}

// A cloud layer projected onto a plane overhead, drifting with time.
float cloud_cover(vec3 dir, float time) {
    if (dir.y <= 0.02) {
        return 0.0;
    }
    vec2 p = dir.xz / (dir.y + 0.15) * 0.9 + vec2(time * 0.006, time * 0.002);
    return smoothstep(0.52, 0.78, fbm(p)) * smoothstep(0.02, 0.18, dir.y);
}

void main() {
    vec2 ndc = uv * 2.0 - 1.0;
    // World-space ray from the camera basis — Vulkan NDC y points down, so
    // negate it to get world-up (example 14's convention).
    vec3 dir = normalize(pc.forward.xyz
                       + ndc.x * pc.params.y * pc.params.x * pc.right.xyz
                       - ndc.y * pc.params.x * pc.up.xyz);

    float night = u.lightColor.w;
    float time = u.ambient.w;
    vec3 col = mix(sky_day(dir, u.sunDir.xyz),
                   sky_night(dir, -u.sunDir.xyz, time),
                   night);

    // The HDR sun disc.
    float cosSun = dot(dir, u.sunDir.xyz);
    col += vec3(1.0, 0.9, 0.7) * smoothstep(0.99980, 0.99995, cosSun)
         * u.sunDir.w * (1.0 - night);

    // Clouds cover whatever is behind them, the sun disc included.
    float cover = cloud_cover(dir, time);
    float lowSun = 1.0 - clamp(u.sunDir.y * 4.0, 0.0, 1.0);
    vec3 sunlit = mix(vec3(1.05, 1.00, 0.96), vec3(1.00, 0.55, 0.30), lowSun);
    vec3 cloudCol = mix(sunlit * (0.55 + 0.45 * (1.0 - cover)),
                        vec3(0.015, 0.020, 0.035), night);
    col = mix(col, cloudCol, cover * 0.9);

    outColor = vec4(col, 1.0);
}
