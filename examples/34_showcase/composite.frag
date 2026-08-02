#version 450
// The last pass: HDR sum, lens flare, ACES tonemap, time-of-day grade and a
// vignette, straight to the swapchain. This runs at window resolution and the
// inputs stay at the internal resolution — resizing the window only
// stretches, nothing reallocates.

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D sceneTex;
layout(set = 0, binding = 1) uniform sampler2D bloomTex;
layout(set = 0, binding = 2) uniform sampler2D godrayTex;
layout(set = 0, binding = 3) uniform sampler2D aoTex;

layout(push_constant) uniform PC {
    float night;    // 0..1
    float exposure;
    float warmth;   // dawn/dusk grade amount
    float pad0, pad1, pad2, pad3, pad4;
} pc;

vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
    // AO darkens the scene only — bloom and god rays are light added on top
    // and stay unoccluded.
    vec3 hdr = texture(sceneTex, uv).rgb * texture(aoTex, uv).r
             + texture(bloomTex, uv).rgb * 0.45
             + texture(godrayTex, uv).rgb;

    // The float target is headroom, not a display mode: light sums past 1.0
    // so the bloom pass has something to find, and this maps it back into
    // the 0-1 range the swapchain shows.
    vec3 col = aces(hdr * pc.exposure);

    // Punch: a saturation lift and a gentle S-curve, or everything reads as
    // bare albedo.
    float grey0 = dot(col, vec3(0.299, 0.587, 0.114));
    col = clamp(mix(vec3(grey0), col, 1.22), 0.0, 1.0);
    col = mix(col, col * col * (3.0 - 2.0 * col), 0.35);

    // Grade: warm dawns and dusks, cool desaturated nights.
    col = mix(col, col * vec3(1.08, 0.97, 0.88), pc.warmth);
    float grey = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(col, mix(col, vec3(grey) * vec3(0.75, 0.85, 1.10), 0.35), pc.night);

    vec2 v = uv - 0.5;
    col *= 1.0 - dot(v, v) * 0.55;

    outColor = vec4(col, 1.0);
}
