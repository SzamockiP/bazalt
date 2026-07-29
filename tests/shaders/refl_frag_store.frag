#version 450
layout(location = 0) out vec4 outColor;
layout(set = 0, binding = 0, rgba8) uniform writeonly image2D dst;
void main() {
    imageStore(dst, ivec2(gl_FragCoord.xy), vec4(0.0, 1.0, 0.0, 1.0));
    outColor = vec4(1.0, 0.0, 0.0, 1.0);
}
