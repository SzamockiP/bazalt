#version 450

// One full-height stripe per instance, so the painted pixel count is exactly
// proportional to instanceCount — which is what makes an indirect draw testable
// without reading the argument buffer back and trusting it.
//
// Instance i covers x in [-1 + i*w, -1 + (i+1)*w) where w = 2/total, so N
// instances paint the leftmost N stripes and nothing else.
layout(push_constant) uniform Push {
    uint total;
} push;

void main() {
    // Wound to match fullscreen.vert: front-facing under the pipeline default
    // (cull BACK, front face COUNTER_CLOCKWISE). The mirror order compiles and
    // draws nothing at all, which looks exactly like a broken indirect draw.
    const vec2 corners[6] = vec2[6](
        vec2(0.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 0.0),
        vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(1.0, 0.0)
    );
    float w = 2.0 / float(push.total);
    float x0 = -1.0 + float(gl_InstanceIndex) * w;
    vec2 p = corners[gl_VertexIndex];
    gl_Position = vec4(x0 + p.x * w, -1.0 + p.y * 2.0, 0.0, 1.0);
}
