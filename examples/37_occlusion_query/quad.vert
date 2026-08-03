#version 450

// One axis-aligned rectangle from gl_VertexIndex, placed and sized by push
// constants. Two of these are the whole scene: a wall, and a shape that passes
// behind it.

layout(push_constant) uniform Push {
    vec2 centre;
    vec2 half_size;
    vec3 color;
    float depth;
} pc;

layout(location = 0) out vec3 color;

void main() {
    vec2 corner = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2) - 1.0;
    color = pc.color;
    gl_Position = vec4(pc.centre + corner * pc.half_size, pc.depth, 1.0);
}
