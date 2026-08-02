#version 450
#include "frame.glsl"
// One point per firefly, fetched straight from the buffer the compute pass
// integrates. The flicker here must match scene.frag's firefly_light — the
// sprite and the light it casts are the same creature.

layout(location = 0) in vec4 inPos; // xyz; w: flicker phase
layout(location = 1) in vec4 inVel; // xyz; w: seed

layout(location = 0) out float brightness;

void main() {
    gl_Position = u.viewProj * vec4(inPos.xyz, 1.0);
    float glow = 0.6 + 0.4 * sin(u.ambient.w * 3.0 + inPos.w);
    brightness = u.lightColor.w * glow;

    float dist = max(length(u.camPos.xyz - inPos.xyz), 0.1);
    float size = 1.0 + 0.3 * sin(inVel.w); // the seed varies the body size
    gl_PointSize = clamp(18.0 * size / dist, 1.0, 12.0) * step(0.001, brightness);
}
