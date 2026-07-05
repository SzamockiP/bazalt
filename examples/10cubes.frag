#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragPos;
layout(location = 3) flat in int instanceID;

layout(location = 0) out vec4 outColor;

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
    if (instanceID == 10) {
        // Render the light source itself as an unlit, fully bright object
        outColor = vec4(ubo.lightColor, 1.0);
        return;
    }

    float ambientStrength = 0.2;
    vec3 ambient = ambientStrength * ubo.lightColor;
    
    vec3 norm = normalize(fragNormal);
    vec3 lightDir = normalize(ubo.lightPos - fragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * ubo.lightColor;
    
    float specularStrength = 0.5;
    vec3 viewDir = normalize(ubo.viewPos - fragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * ubo.lightColor;  
        
    vec3 result = (ambient + diffuse + specular) * fragColor;
    outColor = vec4(result, 1.0);
}
