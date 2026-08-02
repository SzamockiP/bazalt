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

layout(push_constant) uniform PC {
    vec2  sunPos;   // sun in uv space
    float sunVis;   // 0..1, zero when the sun is off-frame or set
    float night;    // 0..1
    float exposure;
    float warmth;   // dawn/dusk grade amount
    float pad0, pad1;
} pc;

vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

// Pseudo lens flare (John Chapman's trick): ghosts are the blurred bright
// buffer sampled at positions mirrored through the screen centre. They slide
// as the camera turns and cost no extra pass.
vec3 lens_flare() {
    vec2 ghostDir = (vec2(0.5) - uv) * 0.6;
    vec3 acc = vec3(0.0);
    for (int i = 1; i <= 4; ++i) {
        vec2 p = uv + ghostDir * float(i);
        float w = max(1.0 - length(vec2(0.5) - p) * 2.0, 0.0);
        acc += texture(bloomTex, p).rgb * (w * w);
    }
    // A wide halo ring around the centre.
    vec2 haloP = uv + normalize(ghostDir) * 0.35;
    acc += texture(bloomTex, haloP).rgb
         * max(1.0 - length(vec2(0.5) - haloP) * 4.0, 0.0);
    return acc * pc.sunVis * 0.10 * vec3(0.9, 0.7, 1.0);
}

void main() {
    vec3 hdr = texture(sceneTex, uv).rgb
             + texture(bloomTex, uv).rgb * 0.55
             + texture(godrayTex, uv).rgb;
    hdr += lens_flare();

    vec3 col = aces(hdr * pc.exposure);

    // Grade: warm dawns and dusks, cool desaturated nights.
    col = mix(col, col * vec3(1.08, 0.97, 0.88), pc.warmth);
    float grey = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(col, mix(col, vec3(grey) * vec3(0.75, 0.85, 1.10), 0.35), pc.night);

    vec2 v = uv - 0.5;
    col *= 1.0 - dot(v, v) * 0.55;

    outColor = vec4(col, 1.0);
}
