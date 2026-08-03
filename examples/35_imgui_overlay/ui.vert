#version 450

// ImGui hands out vertices in FRAMEBUFFER PIXELS, so the whole transform is a
// scale and a translate into clip space. No projection matrix and no camera.
//
// No Y flip: ImGui's Y points down and so does Vulkan's clip space, which is one
// of the few places those two agree.

layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
layout(location = 2) in vec4 aColor;

layout(push_constant) uniform Push {
    vec2 scale;
    vec2 translate;
} pc;

layout(location = 0) out vec2 uv;
layout(location = 1) out vec4 color;

void main() {
    uv = aUV;
    color = aColor;
    gl_Position = vec4(aPos * pc.scale + pc.translate, 0.0, 1.0);
}
