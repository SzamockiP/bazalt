#version 450
#extension GL_EXT_multiview : require

// Cube capture in ONE pass: multiview runs this shader once per face, with
// gl_ViewIndex telling it which face (= cube layer) is being rendered. It picks
// that face's view-projection from a UBO array of six. This is the whole point
// of multiview — a single draw fans out to all six faces instead of six passes.

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inColor;

layout(set = 0, binding = 0) uniform Faces { mat4 viewProj[6]; } u;

layout(location = 0) out vec3 vColor;

void main() {
    vColor = inColor;
    gl_Position = u.viewProj[gl_ViewIndex] * vec4(inPos, 1.0);
}
