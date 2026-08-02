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
layout(set = 1, binding = 1) uniform sampler2D normalMaps[];

// Cotangent frame from screen-space derivatives (Schüler) — normal mapping
// with no tangent attribute, so the vertex format and the OBJ parse never
// heard of it. Slots without a bump map hold a flat 1x1 and fall through
// unchanged.
vec3 perturb_normal(vec3 n, vec3 viewVec, vec2 texUv, vec3 mapN) {
    vec3 dp1 = dFdx(viewVec);
    vec3 dp2 = dFdy(viewVec);
    vec2 duv1 = dFdx(texUv);
    vec2 duv2 = dFdy(texUv);
    vec3 dp2perp = cross(dp2, n);
    vec3 dp1perp = cross(n, dp1);
    vec3 t = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 b = dp2perp * duv1.y + dp1perp * duv2.y;
    float invmax = inversesqrt(max(dot(t, t), dot(b, b)));
    return normalize(mat3(t * invmax, b * invmax, n) * mapN);
}

// Twelve Poisson-disk taps over the hardware 2x2 PCF, the disk rotated by a
// per-pixel hash — the noise trades the blocky texel staircase for grain the
// eye forgives. The sampler clamps to an OPAQUE_WHITE border, so outside the
// map means lit and no uv guard is needed. Depth bias lives on the caster
// pipeline.
float shadow_factor() {
    const vec2 POISSON[12] = vec2[](
        vec2(-0.326, -0.406), vec2(-0.840, -0.074), vec2(-0.696,  0.457),
        vec2(-0.203,  0.621), vec2( 0.962, -0.195), vec2( 0.473, -0.480),
        vec2( 0.519,  0.767), vec2( 0.185, -0.893), vec2( 0.507,  0.064),
        vec2( 0.896,  0.412), vec2(-0.322, -0.933), vec2(-0.792, -0.598));

    vec3 proj = lightSpacePos.xyz / lightSpacePos.w;
    vec2 suv = proj.xy * 0.5 + 0.5;
    vec2 texel = 1.0 / vec2(textureSize(shadowMap, 0));

    float angle = 6.2832 * fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
    mat2 rot = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));

    float lit = 0.0;
    for (int i = 0; i < 12; ++i) {
        vec2 offset = rot * POISSON[i] * (texel * 2.5);
        lit += texture(shadowMap, vec3(suv + offset, proj.z));
    }
    // lightDir.w fades to zero around the sun/moon handover, hiding the flip.
    return mix(1.0, lit / 12.0, u.lightDir.w);
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
    vec3 viewVec = u.camPos.xyz - worldPos;
    vec3 mapN = texture(normalMaps[nonuniformEXT(materialIndex)], uv).rgb * 2.0 - 1.0;
    n = perturb_normal(n, viewVec, uv, mapN);

    float shadow = shadow_factor();
    float ndl = max(dot(n, u.lightDir.xyz), 0.0);
    // Blinn-Phong off the sun/moon — the asset carries almost no specular
    // maps, so one uniform gloss stands in for all of them.
    vec3 halfVec = normalize(u.lightDir.xyz + normalize(viewVec));
    float spec = pow(max(dot(n, halfVec), 0.0), 48.0) * 0.25;
    vec3 color = albedo.rgb * (u.ambient.rgb + u.lightColor.rgb * (ndl * shadow))
               + u.lightColor.rgb * (spec * shadow);
    if (u.lightColor.w > 0.001) {
        color += albedo.rgb * firefly_light(n);
    }

    // The alpha feeds alpha-to-coverage: MSAA dithers the leaf edges instead
    // of a hard discard cutting them.
    outColor = vec4(color, albedo.a);
}
