// The one per-frame UBO, shared by every pass that needs the camera or the
// light. Declared once here and #included (the hot-reload watcher follows
// includes, so editing this file reloads every shader that uses it).
//
// Direction convention: lightDir and sunDir point TOWARD the light source.
layout(std140, set = 0, binding = 0) uniform Frame {
    mat4 viewProj;   // camera view-projection
    mat4 lightVP;    // shadow light view-projection (sun by day, moon by night)
    vec4 lightDir;   // xyz: toward the active light; w: shadow strength 0..1
    vec4 lightColor; // rgb: light colour * intensity; w: night factor 0..1
    vec4 ambient;    // rgb: ambient colour; w: time in seconds
    vec4 camPos;     // xyz: camera position; w: firefly count
    vec4 sunDir;     // xyz: toward the sun (may be below horizon); w: sun disc HDR intensity
} u;
