#version 450

// Solid white on black, so every grey pixel in the final image is coverage and
// nothing else.

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(1.0);
}
