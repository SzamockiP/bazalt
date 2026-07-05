#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragPos;
layout(location = 3) flat out int instanceID;

layout(binding = 0) uniform UBO {
    mat4 view;
    mat4 proj;
    mat4 models[11];
    vec3 lightPos;
    float pad1;
    vec3 viewPos;
    float pad2;
    vec3 lightColor;
    float pad3;
} ubo;

void main() {
    mat4 model = ubo.models[gl_InstanceIndex];
    vec4 worldPos = model * vec4(inPosition, 1.0);
    gl_Position = ubo.proj * ubo.view * worldPos;
    
    fragPos = worldPos.xyz;
    fragNormal = mat3(model) * inNormal;
    fragColor = inColor;
    instanceID = gl_InstanceIndex;
}
