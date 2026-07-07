#version 450

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in vec3 fragColor;

layout(set = 1, binding = 0) uniform sampler2D texSampler;

layout(location = 0) out vec4 outColor;

void main() {
    // Proste światło kierunkowe i kolor bazujący na wektorze normalnym
    vec3 normal = normalize(fragNormal);
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    float diff = max(dot(normal, lightDir), 0.2); // 0.2 to ambient
    
    vec4 texColor = texture(texSampler, fragUV);
    
    // Alpha test - odrzucamy niewidoczne piksele (np. tła liści)
    if (texColor.a < 0.1) {
        discard;
    }
    
    outColor = vec4(texColor.rgb * fragColor * diff, texColor.a);
}
