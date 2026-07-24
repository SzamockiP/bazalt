#version 450

// Samples the captured environment cubemap along the reflection vector, with
// parallax correction for the box-shaped room.

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vViewDir;
layout(location = 2) in vec3 vWorldPos;

layout(set = 0, binding = 0) uniform samplerCube envMap;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 r = reflect(normalize(vViewDir), normalize(vNormal));

    // Parallax-correct the cubemap for this box-shaped room (captured from the
    // origin). A plain cubemap sample has the parallax of a point-sampled
    // environment — sampled from the cube's centre, not the surface point — so the
    // reflected wall corners don't line up with the real ones. Intersect the
    // reflection ray with the room box and sample toward the hit point from the
    // capture centre (the origin) instead.
    const vec3 HALF = vec3(8.0);       // room half-extents (matches room(8.0) in main.py)
    vec3 invR = 1.0 / r;
    vec3 t1 = (-HALF - vWorldPos) * invR;
    vec3 t2 = ( HALF - vWorldPos) * invR;
    vec3 tmax = max(t1, t2);
    float t = min(min(tmax.x, tmax.y), tmax.z);
    vec3 hit = vWorldPos + r * t;       // capture centre is the origin, so the
    vec3 reflection = texture(envMap, hit).rgb;  // direction to the hit is just `hit`

    // A constant base tint so the cube reads as a distinct (copper-tinted mirror)
    // object against the walls it reflects, instead of blending into them.
    vec3 base = vec3(0.75, 0.45, 0.2);
    outColor = vec4(mix(base, reflection, 0.6), 1.0);
}
