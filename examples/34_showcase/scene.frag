#version 450
#extension GL_EXT_nonuniform_qualifier : require
#include "frame.glsl"

layout(location = 0) in vec3 worldPos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec4 lightSpacePos;
layout(location = 4) flat in uint materialIndex;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform sampler2DShadow shadowMap;

struct Firefly {
    vec4 pos; // xyz; w: flicker phase
    vec4 vel; // xyz; w: seed
};
layout(std430, set = 0, binding = 2) readonly buffer Fireflies { Firefly flies[]; };

layout(set = 1, binding = 0) uniform sampler2D materials[];

// 3x3 taps over the hardware 2x2 PCF — nine compares, each already filtered.
// The sampler clamps to an OPAQUE_WHITE border, so outside the map means lit
// and no uv guard is needed. Depth bias lives on the caster pipeline.
float shadow_factor() {
    vec3 proj = lightSpacePos.xyz / lightSpacePos.w;
    vec2 suv = proj.xy * 0.5 + 0.5;
    vec2 texel = 1.0 / vec2(textureSize(shadowMap, 0));
    float lit = 0.0;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            lit += texture(shadowMap, vec3(suv + vec2(x, y) * texel, proj.z));
        }
    }
    // lightDir.w fades to zero around the sun/moon handover, hiding the flip.
    return mix(1.0, lit / 9.0, u.lightDir.w);
}

// Every firefly is a small warm point light. The night factor gates the whole
// loop, so daylight pays one uniform branch and nothing else.
vec3 firefly_light(vec3 n) {
    vec3 acc = vec3(0.0);
    uint count = uint(u.camPos.w);
    for (uint i = 0u; i < count; ++i) {
        vec3 d = flies[i].pos.xyz - worldPos;
        float d2 = dot(d, d);
        float atten = max(1.0 - d2 / 9.0, 0.0);  // 3 m cutoff radius
        if (atten <= 0.0) {
            continue;
        }
        atten = atten * atten / (1.0 + d2 * 4.0);
        float glow = 0.6 + 0.4 * sin(u.ambient.w * 3.0 + flies[i].pos.w);
        acc += vec3(1.0, 0.85, 0.35) * (atten * glow * max(dot(n, normalize(d)), 0.0));
    }
    return acc * (u.lightColor.w * 0.5);
}

void main() {
    vec4 albedo = texture(materials[nonuniformEXT(materialIndex)], uv);

    vec3 n = normalize(normal);
    if (!gl_FrontFacing) {
        n = -n; // leaves are single-sided geometry lit from either side
    }

    float ndl = max(dot(n, u.lightDir.xyz), 0.0);
    vec3 color = albedo.rgb * (u.ambient.rgb + u.lightColor.rgb * (ndl * shadow_factor()));
    if (u.lightColor.w > 0.001) {
        color += albedo.rgb * firefly_light(n);
    }

    // The alpha feeds alpha-to-coverage: MSAA dithers the leaf edges instead
    // of a hard discard cutting them.
    outColor = vec4(color, albedo.a);
}
