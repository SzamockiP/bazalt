#version 450

// The reference picture: a plain dark cube, so you can see where the volume is
// while the camera is outside it.

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(0.20, 0.24, 0.34, 1.0);
}
