#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

// One bar per mode. `index` is which bar lights up, `count` is how many there
// are, and `glow` fades the whole strip in and out.
layout(push_constant) uniform Bar {
    float index;
    float count;
    float glow;
} bar;

void main() {
    float slot = floor(uv.x * bar.count);
    float lit = slot == bar.index ? 1.0 : 0.12;

    // A gap between the bars, so the strip reads as segments and not a block.
    float edge = fract(uv.x * bar.count);
    float gap = min(edge, 1.0 - edge) < 0.03 ? 0.0 : 1.0;

    outColor = vec4(vec3(0.25, 0.65, 1.0) * lit * gap * bar.glow, 1.0);
}
