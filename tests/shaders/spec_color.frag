#version 450

// Specialization constants: the values are supplied when the pipeline is built,
// not when the shader is compiled, so one module serves several pipelines.

layout(constant_id = 0) const float RED = 0.0;
layout(constant_id = 1) const int STEPS = 1;
layout(constant_id = 2) const bool BLUE = false;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

void main() {
    // A loop the driver can unroll once STEPS is known.
    float green = 0.0;
    for (int i = 0; i < STEPS; ++i) {
        green += 0.25;
    }
    outColor = vec4(RED, green, BLUE ? 1.0 : 0.0, 1.0);
}
