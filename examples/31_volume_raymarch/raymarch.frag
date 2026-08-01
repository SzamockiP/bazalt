#version 450

// Raymarches the 3D texture: a ray per pixel from an orbiting camera, clipped
// to the unit cube the volume fills, then stepped front to back accumulating
// density. This is what sampler3D buys over a stack of 2D slices — one
// filtered lookup anywhere in the field, mip chain included.

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
layout(binding = 0) uniform sampler3D volume;
layout(push_constant) uniform Push { float time; } push;

// Slab method against the unit cube [0,1]^3. Returns (t_enter, t_exit).
vec2 intersect_box(vec3 origin, vec3 dir) {
    vec3 inv = 1.0 / dir;
    vec3 t0 = (vec3(0.0) - origin) * inv;
    vec3 t1 = (vec3(1.0) - origin) * inv;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    return vec2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}

void main() {
    // Orbiting camera looking at the cube's centre.
    float angle = push.time * 0.4;
    vec3 eye = vec3(0.5) + vec3(1.8 * cos(angle), 0.9, 1.8 * sin(angle));
    vec3 forward = normalize(vec3(0.5) - eye);
    vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
    vec3 up = cross(right, forward);

    vec2 ndc = uv * 2.0 - 1.0;
    vec3 dir = normalize(forward + ndc.x * right - ndc.y * up);

    vec2 hit = intersect_box(eye, dir);
    hit.x = max(hit.x, 0.0);

    vec3 sky = mix(vec3(0.09, 0.11, 0.18), vec3(0.02, 0.03, 0.06), uv.y);
    if (hit.x >= hit.y) {
        color = vec4(sky, 1.0);
        return;
    }

    const int steps = 96;
    float step_len = (hit.y - hit.x) / float(steps);
    vec3 light_dir = normalize(vec3(0.6, 1.0, 0.3));

    float transmittance = 1.0;
    vec3 lit = vec3(0.0);
    for (int i = 0; i < steps; ++i) {
        vec3 p = eye + dir * (hit.x + (float(i) + 0.5) * step_len);
        float density = texture(volume, p).r;
        if (density <= 0.001) {
            continue;
        }
        // One cheap shadow tap towards the light instead of a nested march.
        float towards_light = texture(volume, p + light_dir * 0.08).r;
        float shading = clamp(density - towards_light, 0.0, 1.0) * 2.0 + 0.35;
        vec3 sample_color = mix(vec3(0.25, 0.35, 0.55), vec3(1.0, 0.98, 0.92), shading);

        float alpha = 1.0 - exp(-density * step_len * 24.0);
        lit += transmittance * alpha * sample_color;
        transmittance *= 1.0 - alpha;
        if (transmittance < 0.01) {
            break;
        }
    }

    color = vec4(lit + transmittance * sky, 1.0);
}
