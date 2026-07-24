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
    vec4 lightDir;       // .xyz, points from surface toward the light
} u;

layout(set = 0, binding = 1) uniform sampler2DArrayShadow shadowMap;

const vec3 TINT[3] = vec3[3](vec3(1.0, 0.7, 0.7), vec3(0.7, 1.0, 0.7), vec3(0.7, 0.7, 1.0));

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
    float lit = texture(shadowMap, vec4(uv, float(c), proj.z - 0.002));  // 0.002: constant depth bias

    float ndl = max(dot(normalize(vNormal), normalize(u.lightDir.xyz)), 0.0);
    float shade = 0.25 + 0.75 * lit * ndl;    // ambient + lit diffuse
    outColor = vec4(TINT[c] * shade, 1.0);
}
