#version 450

// Picks a cascade by how far the fragment is from the scene centre (concentric
// boxes: the smallest that contains it), projects into that cascade's light
// space, and does a hardware-PCF shadow test against the matching layer of the
// shadow array (sampler2DArrayShadow). The cascade also tints the surface so the
// split is visible.

layout(location = 0) in vec3 vWorld;
layout(location = 1) in vec3 vNormal;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform Cascades {
    mat4 lightVP[3];
    vec4 cascadeExtent;  // half-size per cascade in .xyz
    vec4 lightDir;       // .xyz, the direction the light RAYS travel (downward-ish)
} u;

layout(set = 0, binding = 1) uniform sampler2DArrayShadow shadowMap;

const vec3 TINT[3] = vec3[3](vec3(1.0, 0.3, 0.3), vec3(0.3, 1.0, 0.3), vec3(0.3, 0.3, 1.0));

int pick_cascade(vec3 p) {
    for (int c = 0; c < 3; ++c) {
        float e = u.cascadeExtent[c];
        if (abs(p.x) < e && abs(p.z) < e) return c;
    }
    return 2;
}

void main() {
    int c = pick_cascade(vWorld);
    vec4 lp = u.lightVP[c] * vec4(vWorld, 1.0);
    vec3 proj = lp.xyz / lp.w;
    vec2 uv = proj.xy * 0.5 + 0.5;            // xy: NDC -> [0,1]; z is already [0,1] (Vulkan ZO)

    // Direction TO the light is -lightDir (lightDir is the ray travel direction),
    // so an up-facing ground lit from above gets full diffuse.
    float ndl = max(dot(normalize(vNormal), -normalize(u.lightDir.xyz)), 0.0);

    // Slope-scaled depth bias: a surface angled to the light needs more bias, or it
    // self-shadows (acne) — false shadows on open floor with nothing above it.
    float bias = max(0.0015, 0.006 * (1.0 - ndl));

    float lit;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || proj.z > 1.0) {
        lit = 1.0;    // outside this cascade's shadow map -> lit, not spuriously dark
    } else {
        lit = texture(shadowMap, vec4(uv, float(c), proj.z - bias));
    }

    float light = 0.3 + 0.7 * lit * ndl;      // ambient + shadowed diffuse

    // A subtle cascade tint on a bright base makes the split bands visible without
    // muddying the scene.
    vec3 tint = mix(vec3(1.0), TINT[c], 0.3);
    outColor = vec4(tint * light, 1.0);
}
