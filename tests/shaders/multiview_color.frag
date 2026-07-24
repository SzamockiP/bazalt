#version 450
#extension GL_EXT_multiview : require

// One pass, every layer: gl_ViewIndex is the layer being rendered under
// multiview, so each layer gets its own colour from a single draw. This is how
// a real multiview pass would index a per-view matrix; here it just proves the
// draw fanned out to all layers with the right index per layer.

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(float(gl_ViewIndex) / 4.0, 0.25, 0.5, 1.0);
}
