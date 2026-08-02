"""Geometry shader — drawing normals as lines, from the mesh's own buffer.

This is the one thing the geometry stage does that nothing else in bazalt can:
it changes the primitive type. `normals.geom` takes each triangle and emits line
strips running along the surface normals, so the debug view comes out of the same
vertex buffer as the shaded sphere, in the same frame, with no second CPU-built
line buffer to keep in sync.

Tessellation cannot do this. It subdivides a patch and hands back the same kind of
primitive, so "triangles in, lines out" is outside what it can express. What DID
take geometry's most famous job is elsewhere in bazalt: routing one draw into every
layer of an array attachment used to need `gl_Layer` from a geometry shader, and
`target.all_layers()` (multiview, 0.13) does it without one.

Geometry shaders are slow on modern hardware and absent on MoltenVK, which is why
`Feature.GEOMETRY_SHADER` has to be asked for and why a debug view like this is a
much better use of the stage than a hot path.

Keys: SPACE toggles the normals, UP/DOWN change their length.
"""

import math
import struct
import time

import glm
import numpy as np

import bazalt as bz

logger = bz.Logger()
logger.on_message(lambda msg: print(f"[{msg.severity}] {msg.text}"))

window = bz.Window(1024, 720, "Bazalt Demo - Normals from a geometry shader", logger=logger)
ctx = bz.Context(logger, features=[bz.Feature.GEOMETRY_SHADER])
renderer = ctx.create_renderer(window)

vert = ctx.compile_shader("mesh.vert", bz.ShaderStage.VERTEX)
geom = ctx.compile_shader("normals.geom", bz.ShaderStage.GEOMETRY)
shade = ctx.compile_shader("shade.frag", bz.ShaderStage.FRAGMENT)
line = ctx.compile_shader("line.frag", bz.ShaderStage.FRAGMENT)

def with_camera(builder):
    """Declare set 0 the same way on every pipeline that shares its descriptor set.

    One descriptor set serves both pipelines below, and a set is allocated against
    a layout — so if `solid` declared binding 0 for VERTEX only while `normals`
    declared it for VERTEX and GEOMETRY, the two layouts would differ and binding
    the set to the second pipeline is a validation error about compatibility, not
    a wrong picture. Declaring a stage a pipeline does not have is legal and free;
    bazalt ORs the flags for a repeated binding index.
    """
    return (builder
            .uniform_buffer(0, bz.ShaderStage.VERTEX)
            .uniform_buffer(0, bz.ShaderStage.GEOMETRY))


# The shaded sphere: an ordinary vertex + fragment pipeline.
solid = (with_camera(ctx.graphics_pipeline()
                     .vertex_shader(vert)
                     .fragment_shader(shade)
                     .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3]))
         .depth_test(True)
         .cull_mode(bz.CullMode.NONE, bz.FrontFace.COUNTER_CLOCKWISE)
         .build(renderer))

# The normals: same vertex shader, same buffer, one extra stage. The camera also
# reaches the geometry stage, because that is where the projection now happens.
normals = (with_camera(ctx.graphics_pipeline()
                       .vertex_shader(vert)
                       .geometry_shader(geom)
                       .fragment_shader(line)
                       .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3]))
           .push_constant(4, bz.ShaderStage.GEOMETRY)
           .depth_test(True)
           .cull_mode(bz.CullMode.NONE, bz.FrontFace.COUNTER_CLOCKWISE)
           .build(renderer))

# ── a UV sphere ───────────────────────────────────────────────────────────
# On a unit sphere the position IS the normal, which keeps this to a few lines.
RINGS, SEGMENTS = 14, 22
positions = []
indices = []
for r in range(RINGS + 1):
    phi = math.pi * r / RINGS
    for s in range(SEGMENTS + 1):
        theta = 2.0 * math.pi * s / SEGMENTS
        n = (math.sin(phi) * math.cos(theta), math.cos(phi), math.sin(phi) * math.sin(theta))
        positions += [n[0], n[1], n[2], n[0], n[1], n[2]]
for r in range(RINGS):
    for s in range(SEGMENTS):
        a = r * (SEGMENTS + 1) + s
        b = a + SEGMENTS + 1
        indices += [a, b, a + 1, a + 1, b, b + 1]

vbuf = ctx.create_buffer(np.array(positions, dtype=np.float32),
                         bz.BufferType.VERTEX, bz.MemoryUsage.STATIC)
ibuf = ctx.create_buffer(np.array(indices, dtype=np.uint32),
                         bz.BufferType.INDEX, bz.MemoryUsage.STATIC)
index_count = len(indices)

# view_proj + model
ubuf = ctx.create_buffer(32 * 4, bz.BufferType.UNIFORM, bz.MemoryUsage.DYNAMIC)
pool = ctx.create_descriptor_pool()
desc_set = pool.allocate_frame_set(solid)
desc_set.set_buffer(0, ubuf)

show_normals = True
normal_length = 0.25

proj = glm.perspectiveRH_ZO(glm.radians(45.0), 1024.0 / 720.0, 0.1, 100.0)
proj[1][1] *= -1
start = time.time()
frames = 0
fps_timer = start

while window.is_open():
    bz.poll_events()

    if window.was_key_pressed(bz.Key.SPACE):
        show_normals = not show_normals
    if window.was_key_pressed(bz.Key.UP):
        normal_length = min(normal_length + 0.05, 1.0)
    if window.was_key_pressed(bz.Key.DOWN):
        normal_length = max(normal_length - 0.05, 0.0)

    ctx.begin_frame()
    if not renderer.acquire():
        continue

    t = time.time() - start
    view = glm.lookAt(glm.vec3(0.0, 1.2, 4.0), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
    model = glm.rotate(glm.mat4(1.0), t * 0.4, glm.vec3(0.2, 1.0, 0.1))
    # DYNAMIC means one buffer per frame in flight, so this has to be written
    # every frame and not once at startup.
    ubuf.update(bytes(glm.transpose(proj * view)) + bytes(glm.transpose(model)))

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(renderer, clear_color=[0.04, 0.05, 0.08, 1.0]) as c:
        c.bind_pipeline(solid)
        c.bind_descriptor_set(desc_set, solid)
        c.bind_vertex_buffer(vbuf).bind_index_buffer(ibuf).draw_indexed(index_count)
    if show_normals and normal_length > 0.0:
        # A second pass that preserves what the first drew: the lines sit on top of
        # the shaded surface and share its depth buffer, so the ones facing away
        # are hidden by the sphere.
        with cmd.rendering(renderer, clear_color=None) as c:
            c.bind_pipeline(normals)
            c.bind_descriptor_set(desc_set, normals)
            c.push_constants(normals, 0, struct.pack("f", normal_length))
            c.bind_vertex_buffer(vbuf).bind_index_buffer(ibuf).draw_indexed(index_count)
    renderer.present(cmd)

    frames += 1
    if time.time() - fps_timer >= 1.0:
        window.set_title(
            f"Bazalt Demo - Normals from a geometry shader | "
            f"normals {'on' if show_normals else 'off'} | "
            f"length {normal_length:.2f} | {frames} FPS")
        frames = 0
        fps_timer = time.time()
