#version 450

// pos2.vert plus gl_PointSize, which POINT_LIST requires of the last
// pre-rasterization stage (VUID-VkGraphicsPipelineCreateInfo-topology-08773).
// It is a separate file rather than a change to pos2.vert because writing
// PointSize from a shader feeding a geometry stage is the wrong thing: there the
// geometry shader is the last pre-rasterization stage and it emits triangles.
layout(location = 0) in vec2 pos;

void main() {
    gl_Position = vec4(pos, 0.0, 1.0);
    gl_PointSize = 1.0;
}
