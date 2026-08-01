#version 450

// Samples one Z plane of a 3D texture across the framebuffer. The plane is a
// push constant so one pipeline can probe every slice.

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
layout(binding = 0) uniform sampler3D vol;
layout(push_constant) uniform Push { float z; } push;

void main() {
    color = texture(vol, vec3(uv, push.z));
}
