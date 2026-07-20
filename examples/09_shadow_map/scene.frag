#version 450

layout(location = 0) in vec3 normal;
layout(location = 1) in vec4 lightSpacePos;

layout(location = 0) out vec4 outColor;

// shadow.depth sampled through a COMPARE sampler (create_sampler(compare=...)):
// texture() takes a reference depth as the third coordinate and returns the
// comparison result — with CompareOp.LESS, 1.0 where the fragment is closer
// than the map (lit), 0.0 where it is behind (shadowed). LINEAR filtering on a
// compare sampler averages the four comparison RESULTS: free hardware PCF.
layout(set = 0, binding = 1) uniform sampler2DShadow shadowMap;

const vec3 LIGHT_DIR = normalize(vec3(0.5, 1.0, 0.25));

void main() {
    vec3 proj = lightSpacePos.xyz / lightSpacePos.w;
    vec2 uv = proj.xy * 0.5 + 0.5;

    float shadow = 1.0;
    if (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0 && proj.z <= 1.0) {
        float lit = texture(shadowMap, vec3(uv, proj.z - 0.003));
        shadow = mix(0.35, 1.0, lit);
    }

    float diffuse = max(dot(normalize(normal), LIGHT_DIR), 0.0);
    vec3 base = vec3(0.85, 0.8, 0.7);
    outColor = vec4(base * (0.25 + 0.75 * diffuse) * shadow, 1.0);
}
