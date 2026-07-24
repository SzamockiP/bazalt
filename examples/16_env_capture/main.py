"""Dynamic environment capture — render-to-layer, the 0.13 headline.

A mirror cube sits in a coloured room. Every frame the room is rasterized into
the six faces of a cubemap FROM the cube's centre — one graphics pass per face,
each into `env.layer(i)` — and then the cube samples that cubemap as a
`samplerCube` to reflect its surroundings. This is real-time environment capture:
the reflection is rendered, not baked, so a moving scene would reflect live.

The whole feature is the two new bits of API:
  * env = bz.RenderTarget(ctx, ENV, ENV, color=..., depth=..., cube=True)
      → a cube colour target (target.color[0] samples as a cubemap) with a
        matching 6-layer depth buffer;
  * cmd.rendering(env.layer(i)) → a pass that rasterizes into cube face i.

Cube face order is Vulkan's +X, -X, +Y, -Y, +Z, -Z (face i == layer i).

NOTE: windowed demo, verified by eye on a real GPU (like the shadow/MSAA
examples). The cube-capture up-vectors + the Vulkan Y-flip are the fiddly part
to eyeball first: a correct run reflects red on one side, green on the opposite,
blue/yellow front/back, white above, grey below.
"""

import math
import struct
import time

import bazalt as bz
import glm
import numpy as np

W, H = 960, 640
ENV = 512  # cubemap face resolution

logger = bz.Logger()
logger.on_message(lambda msg: print(f"[{msg.severity}] {msg.text}"))

window = bz.Window(W, H, "Bazalt Demo - Environment Capture (render-to-layer)", logger=logger)
ctx = bz.Context(logger)
renderer = bz.SwapchainRenderer(window, ctx)

# The environment probe: a cube colour target + a matching cube-shaped depth
# buffer. color[0] ends up sampleable as a samplerCube.
env = bz.RenderTarget(ctx, ENV, ENV, color=bz.Format.RGBA8, depth=bz.Format.D32F, cube=True)

solid_vert = ctx.compile_shader("solid.vert", bz.ShaderStage.VERTEX)
solid_frag = ctx.compile_shader("solid.frag", bz.ShaderStage.FRAGMENT)
reflect_vert = ctx.compile_shader("reflect.vert", bz.ShaderStage.VERTEX)
reflect_frag = ctx.compile_shader("reflect.frag", bz.ShaderStage.FRAGMENT)


def solid_pipeline(target):
    """Vertex-coloured geometry with a pushed view-projection. Built once per
    target (the cube face and the window have different formats). No culling: the
    room is viewed from inside, so both windings are visible."""
    return (ctx.graphics_pipeline()
            .vertex_shader(solid_vert)
            .fragment_shader(solid_frag)
            .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
            .cull_mode(bz.CullMode.NONE, bz.FrontFace.COUNTER_CLOCKWISE)
            .depth_test(True)
            .push_constant(64, bz.ShaderStage.VERTEX)
            .build(target))


capture_pipe = solid_pipeline(env.layer(0))  # cube-face format; reused for all 6 faces
room_pipe = solid_pipeline(renderer)

reflect_pipe = (ctx.graphics_pipeline()
                .vertex_shader(reflect_vert)
                .fragment_shader(reflect_frag)
                .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                .depth_test(True)
                .push_constant(80, bz.ShaderStage.VERTEX)  # mat4 mvp + vec4 camPos
                .texture(0, bz.ShaderStage.FRAGMENT, set=0)
                .build(renderer))

pool = ctx.create_descriptor_pool(max_sets=2, samplers=2)
reflect_set = pool.allocate_set(reflect_pipe, set=0)
reflect_set.set_image(0, env.color[0], sampler=ctx.create_sampler(filter=bz.Filter.LINEAR))


# ── geometry ───────────────────────────────────────────────────────────────

def room(half):
    """Six inward-facing coloured walls: +X red, -X green, +Y white, -Y grey,
    +Z blue, -Z yellow — so a reflection is easy to read by colour."""
    R = half
    walls = [
        ((1.0, 0.2, 0.2), [(R, -R, -R), (R, -R, R), (R, R, R), (R, R, -R)]),      # +X red
        ((0.2, 1.0, 0.2), [(-R, -R, R), (-R, -R, -R), (-R, R, -R), (-R, R, R)]),  # -X green
        ((0.9, 0.9, 0.9), [(-R, R, -R), (-R, R, R), (R, R, R), (R, R, -R)]),      # +Y white
        ((0.3, 0.3, 0.3), [(-R, -R, -R), (-R, -R, R), (R, -R, R), (R, -R, -R)]),  # -Y grey
        ((0.2, 0.4, 1.0), [(-R, -R, R), (R, -R, R), (R, R, R), (-R, R, R)]),      # +Z blue
        ((1.0, 0.9, 0.2), [(-R, -R, -R), (R, -R, -R), (R, R, -R), (-R, R, -R)]),  # -Z yellow
    ]
    verts, idx = [], []
    for color, corners in walls:
        base = len(verts) // 6
        for (x, y, z) in corners:
            verts += [x, y, z, *color]
        idx += [base, base + 1, base + 2, base + 2, base + 3, base]
    return verts, idx


def cube(s):
    """A pos+normal cube centred at the origin — the mirror object."""
    faces = [
        ((0, 0, 1), [(-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s)]),
        ((0, 0, -1), [(s, -s, -s), (-s, -s, -s), (-s, s, -s), (s, s, -s)]),
        ((-1, 0, 0), [(-s, -s, -s), (-s, -s, s), (-s, s, s), (-s, s, -s)]),
        ((1, 0, 0), [(s, -s, s), (s, -s, -s), (s, s, -s), (s, s, s)]),
        ((0, 1, 0), [(-s, s, s), (s, s, s), (s, s, -s), (-s, s, -s)]),
        ((0, -1, 0), [(-s, -s, -s), (s, -s, -s), (s, -s, s), (-s, -s, s)]),
    ]
    verts, idx = [], []
    for normal, corners in faces:
        base = len(verts) // 6
        for (x, y, z) in corners:
            verts += [x, y, z, *normal]
        idx += [base, base + 1, base + 2, base + 2, base + 3, base]
    return verts, idx


room_v, room_i = room(8.0)
room_vbuf = ctx.create_buffer(np.array(room_v, np.float32), bz.BufferType.VERTEX, bz.MemoryUsage.STATIC)
room_ibuf = ctx.create_buffer(np.array(room_i, np.uint32), bz.BufferType.INDEX, bz.MemoryUsage.STATIC)
room_count = len(room_i)

cube_v, cube_i = cube(1.0)
cube_vbuf = ctx.create_buffer(np.array(cube_v, np.float32), bz.BufferType.VERTEX, bz.MemoryUsage.STATIC)
cube_ibuf = ctx.create_buffer(np.array(cube_i, np.uint32), bz.BufferType.INDEX, bz.MemoryUsage.STATIC)
cube_count = len(cube_i)


# ── cube-capture matrices ────────────────────────────────────────────────────
# Vulkan face order +X, -X, +Y, -Y, +Z, -Z, captured from the origin with a 90°
# frustum. GL up-vectors + the Vulkan Y-flip on the projection; eyeball on first
# run (see module docstring).
FACE_DIRS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
FACE_UPS = [(0, -1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (0, -1, 0), (0, -1, 0)]

capture_proj = glm.perspectiveRH_ZO(glm.radians(90.0), 1.0, 0.05, 50.0)
capture_proj[1][1] *= -1
FACE_VP = []
for d, u in zip(FACE_DIRS, FACE_UPS):
    view = glm.lookAt(glm.vec3(0.0), glm.vec3(*d), glm.vec3(*u))
    FACE_VP.append(bytes(glm.transpose(capture_proj * view)))


def record(cmd, eye, camera_vp):
    cmd.begin()

    # Capture: rasterize the room into all six cube faces this frame.
    for i in range(6):
        with cmd.rendering(env.layer(i), clear_color=[0, 0, 0, 1]) as c:
            (c.bind_pipeline(capture_pipe)
              .push_constants(capture_pipe, 0, FACE_VP[i])
              .bind_vertex_buffer(room_vbuf)
              .bind_index_buffer(room_ibuf)
              .draw_indexed(room_count))

    # Window: the room, then the mirror cube reflecting the freshly captured env.
    with cmd.rendering(renderer, clear_color=[0.02, 0.02, 0.03, 1.0]) as c:
        (c.bind_pipeline(room_pipe)
          .push_constants(room_pipe, 0, bytes(glm.transpose(camera_vp)))
          .bind_vertex_buffer(room_vbuf)
          .bind_index_buffer(room_ibuf)
          .draw_indexed(room_count))
        (c.bind_pipeline(reflect_pipe)
          .bind_descriptor_set(reflect_set, reflect_pipe, set=0)
          .push_constants(reflect_pipe, 0,
                          bytes(glm.transpose(camera_vp)) + struct.pack("4f", eye.x, eye.y, eye.z, 0.0))
          .bind_vertex_buffer(cube_vbuf)
          .bind_index_buffer(cube_ibuf)
          .draw_indexed(cube_count))


cmd = ctx.create_command_buffer()
proj = glm.perspectiveRH_ZO(glm.radians(60.0), W / H, 0.1, 100.0)
proj[1][1] *= -1

start = time.time()
while window.is_open():
    window.poll_events()
    frame = renderer.begin_frame()
    if frame is None:
        continue
    t = (time.time() - start) * 0.5
    eye = glm.vec3(5.0 * math.cos(t), 2.5, 5.0 * math.sin(t))
    view = glm.lookAt(eye, glm.vec3(0.0), glm.vec3(0.0, 1.0, 0.0))
    record(cmd, eye, proj * view)
    frame.submit(cmd)
