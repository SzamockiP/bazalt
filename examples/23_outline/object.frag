#version 450

layout(location = 0) in vec3 v_normal;
layout(location = 0) out vec4 out_color;

layout(push_constant) uniform Push {
    vec4 color;
    float inflate;
} pc;

void main() {
    // The outline pass passes a flat colour (inflate > 0); the object pass gets
    // the same colour shaded by one light.
    if (pc.inflate > 0.0) {
        out_color = pc.color;
        return;
    }
    float light = 0.3 + 0.7 * max(dot(normalize(v_normal), normalize(vec3(0.5, 0.8, 0.4))), 0.0);
    out_color = vec4(pc.color.rgb * light, 1.0);
}
