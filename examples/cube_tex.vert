#version 450
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inUV_TexIndex; // x=u, y=v, z=texIndex

layout(binding = 0) uniform UBO {
    mat4 mvp;
} ubo;

layout(location = 0) out vec2 fragUV;
layout(location = 1) out float fragTexIndex;

void main() {
    gl_Position = ubo.mvp * vec4(inPos, 1.0);
    fragUV = inUV_TexIndex.xy;
    fragTexIndex = inUV_TexIndex.z;
}
