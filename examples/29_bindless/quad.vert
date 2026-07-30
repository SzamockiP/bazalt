#version 450

// The mesh (locations 0-1) and the per-instance row (locations 2-4). Locations
// continue across the two bindings, which is why the instance data starts at 2.

layout(location = 0) in vec2 inPos;
layout(location = 1) in vec2 inUv;

layout(location = 2) in vec2 inOffset;
layout(location = 3) in float inScale;
layout(location = 4) in uint inTexture;

layout(location = 0) out vec2 uv;
// flat: an index must not be interpolated. It still differs between instances,
// so the fragment shader has to treat it as non-uniform.
layout(location = 1) flat out uint texIndex;

void main() {
    uv = inUv;
    texIndex = inTexture;
    gl_Position = vec4(inPos * inScale + inOffset, 0.0, 1.0);
}
