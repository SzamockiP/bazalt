#version 450
layout(location = 0) in vec2 fragUV;
layout(location = 1) in float fragTexIndex;

layout(binding = 1) uniform sampler2D texSampler1;
layout(binding = 2) uniform sampler2D texSampler2;

layout(location = 0) out vec4 outColor;

void main() {
    if (fragTexIndex < 0.5) {
        outColor = texture(texSampler1, fragUV);
    } else {
        outColor = texture(texSampler2, fragUV);
    }
}
