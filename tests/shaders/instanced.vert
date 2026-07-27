#version 450

// Per-vertex data comes from binding 0 (vertex_format), per-instance data from
// binding 1 (instance_format). The locations continue across the two, which is
// the rule instance_format documents.

layout(location = 0) in vec2 pos;    // per vertex
layout(location = 1) in vec2 offset; // per instance
layout(location = 2) in vec4 tint;   // per instance

layout(location = 0) out vec4 color;

void main() {
    color = tint;
    gl_Position = vec4(pos + offset, 0.0, 1.0);
}
