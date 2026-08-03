#version 450

// A spinning star: thin triangles meeting at the centre, which is the shape that
// aliases worst and therefore shows MSAA best.

layout(location = 0) in vec2 aPos;

layout(push_constant) uniform Push {
    float angle;
    float aspect;
} pc;

void main() {
    float s = sin(pc.angle);
    float c = cos(pc.angle);
    vec2 p = mat2(c, -s, s, c) * aPos;
    p.x /= pc.aspect;
    gl_Position = vec4(p, 0.0, 1.0);
}
