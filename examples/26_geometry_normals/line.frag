#version 450

// No inputs at all: the geometry shader emits gl_Position and nothing else, and a
// fragment shader that declared varyings the previous stage does not write would
// fail to link.
layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(1.0, 0.82, 0.2, 1.0);
}
