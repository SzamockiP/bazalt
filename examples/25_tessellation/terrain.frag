#version 450

layout(location = 0) in vec3 inNormal;
layout(location = 1) in float inHeight;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform Push {
    vec4 camera_lod;
    vec4 height_time;
} push;

void main() {
    const vec3 light = normalize(vec3(0.4, 0.9, 0.2));
    float diffuse = max(dot(normalize(inNormal), light), 0.0);

    // Height bands, so the displacement reads even where the lighting is flat.
    float band = clamp(inHeight / max(push.height_time.x, 0.001) * 0.5 + 0.5, 0.0, 1.0);
    vec3 low = vec3(0.16, 0.34, 0.42);
    vec3 high = vec3(0.85, 0.80, 0.62);
    vec3 albedo = mix(low, high, band);

    outColor = vec4(albedo * (0.25 + 0.75 * diffuse), 1.0);
}
