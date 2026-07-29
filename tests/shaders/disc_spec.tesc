#version 450

// disc.tesc with the level as a specialization constant instead of push data.
// It exists to prove that a constant declared for TESS_CONTROL reaches the
// tessellation control shader: before 0.19 the builder sorted constants with an
// if(FRAGMENT)/else, so this value would have been baked into the vertex shader
// and this stage would have kept the default below.
layout(vertices = 3) out;

layout(constant_id = 0) const int LEVEL = 1;

void main() {
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;

    if (gl_InvocationID == 0) {
        gl_TessLevelOuter[0] = float(LEVEL);
        gl_TessLevelOuter[1] = float(LEVEL);
        gl_TessLevelOuter[2] = float(LEVEL);
        gl_TessLevelInner[0] = float(LEVEL);
    }
}
