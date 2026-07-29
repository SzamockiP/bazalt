#version 450

// One shader for both passes. `inflate` is 0 for the object itself and a small
// positive number for the outline pass, which grows the silhouette so the stencil
// test keeps only the ring that was not covered the first time.
//
// The inflation direction is `normalize(position)` — outward from the model's
// centre — and NOT the vertex normal, which is the obvious thing to reach for and
// is wrong for this mesh.
//
// A cube has hard FACE normals: 24 vertices, four per face, each carrying its own
// face direction so the faces shade flat. Pushing those along the normal moves all
// six faces apart as six independent quads, which detach at every edge and leave
// the silhouette full of gaps. Normal inflation needs SMOOTH (averaged) normals,
// where the vertices an edge shares agree on one direction.
//
// Outward-from-the-centre needs no second normal set and stays closed for any mesh
// whose surface surrounds its origin, which is what an example wants. It is only
// undefined at a vertex exactly at the centre, and a surface has none.

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(set = 0, binding = 0) uniform Camera {
    mat4 view_proj;
    mat4 model;
} cam;

layout(push_constant) uniform Push {
    vec4 color;
    float inflate;
} pc;

layout(location = 0) out vec3 v_normal;

void main() {
    vec3 inflated = position + normalize(position) * pc.inflate;
    // Lighting still uses the face normal: flat faces are what makes a cube read
    // as a cube, and only the inflation direction had to change.
    v_normal = mat3(cam.model) * normal;
    gl_Position = cam.view_proj * cam.model * vec4(inflated, 1.0);
}
