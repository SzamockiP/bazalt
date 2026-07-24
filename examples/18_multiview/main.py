"""Multiview environment capture — the six cube faces in ONE pass.

This is example 16 (dynamic reflections) with the six capture passes collapsed
into one via multiview: cmd.rendering(env.all_layers()) renders the room into
every cube face in a single draw, and the vertex shader picks each face's matrix
with gl_ViewIndex. Same result as the six-pass loop, fewer submissions and less
CPU — the whole point of multiview.

The new API is just:
  * ctx.supports_multiview()  → is it available on this GPU
  * env.all_layers()          → a render target covering every layer at once
  * gl_ViewIndex in the shader → which layer this invocation is drawing

NOTE: windowed demo, verified by eye on a real GPU. Cube-capture up-vectors +
the Vulkan Y-flip are the fiddly part (see example 16).
"""

import math
import struct
import sys
import time

import bazalt as bz
import glm
import numpy as np

W, H = 960, 640
ENV = 512

logger = bz.Logger()
logger.on_message(lambda msg: print(f"[{msg.severity}] {msg.text}"))

window = bz.Window(W, H, "Bazalt Demo - Multiview Environment Capture", logger=logger)
ctx = bz.Context(logger)
renderer = bz.SwapchainRenderer(window, ctx)

if not ctx.supports_multiview():
    print("This GPU does not support multiview; see example 16 for the six-pass version.")
    sys.exit(0)

env = bz.RenderTarget(ctx, ENV, ENV, color=bz.Format.RGBA8, depth=bz.Format.D32F, cube=True)

capture_vert = ctx.compile_shader("capture_mv.vert", bz.ShaderStage.VERTEX)
solid_vert = ctx.compile_shader("solid.vert", bz.ShaderStage.VERTEX)
solid_frag = ctx.compile_shader("solid.frag", bz.ShaderStage.FRAGMENT)
reflect_vert = ctx.compile_shader("reflect.vert", bz.ShaderStage.VERTEX)
reflect_frag = ctx.compile_shader("reflect.frag", bz.ShaderStage.FRAGMENT)

# One multiview pass into every cube face; the pipeline picks up the view mask
# from env.all_layers().
capture_pipe = (ctx.graphics_pipeline()
                .vertex_shader(capture_vert)
                .fragment_shader(solid_frag)
                .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                .cull_mode(bz.CullMode.NONE, bz.FrontFace.COUNTER_CLOCKWISE)
                .depth_test(True)
                .uniform_buffer(0, bz.ShaderStage.VERTEX, set=0)
                .build(env.all_layers()))

room_pipe = (ctx.graphics_pipeline()
             .vertex_shader(solid_vert)
             .fragment_shader(solid_frag)
             .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
             .cull_mode(bz.CullMode.NONE, bz.FrontFace.COUNTER_CLOCKWISE)
             .depth_test(True)
             .push_constant(64, bz.ShaderStage.VERTEX)
             .build(renderer))

reflect_pipe = (ctx.graphics_pipeline()
                .vertex_shader(reflect_vert)
                .fragment_shader(reflect_frag)
                .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                .depth_test(True)
                .push_constant(80, bz.ShaderStage.VERTEX)
                .texture(0, bz.ShaderStage.FRAGMENT, set=0)
                .build(renderer))


# ── geometry (same coloured room + mirror cube as example 16) ────────────────

def room(half):
    R = half
    walls = [
        ((1.0, 0.2, 0.2), [(R, -R, -R), (R, -R, R), (R, R, R), (R, R, -R)]),
        ((0.2, 1.0, 0.2), [(-R, -R, R), (-R, -R, -R), (-R, R, -R), (-R, R, R)]),
        ((0.9, 0.9, 0.9), [(-R, R, -R), (-R, R, R), (R, R, R), (R, R, -R)]),
        ((0.3, 0.3, 0.3), [(-R, -R, -R), (-R, -R, R), (R, -R, R), (R, -R, -R)]),
        ((0.2, 0.4, 1.0), [(-R, -R, R), (R, -R, R), (R, R, R), (-R, R, R)]),
        ((1.0, 0.9, 0.2), [(-R, -R, -R), (R, -R, -R), (R, R, -R), (-R, R, -R)]),
    ]
    verts, idx = [], []
    for color, corners in walls:
        base = len(verts) // 6
        for (x, y, z) in corners:
            verts += [x, y, z, *color]
        idx += [base, base + 1, base + 2, base + 2, base + 3, base]
    return verts, idx


def cube(s):
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


# ── the six face matrices, in a UBO the multiview vertex shader indexes ───────
FACE_DIRS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
FACE_UPS = [(0, -1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (0, -1, 0), (0, -1, 0)]

capture_proj = glm.perspectiveRH_ZO(glm.radians(90.0), 1.0, 0.05, 50.0)
capture_proj[1][1] *= -1
face_bytes = b""
for d, u in zip(FACE_DIRS, FACE_UPS):
    view = glm.lookAt(glm.vec3(0.0), glm.vec3(*d), glm.vec3(*u))
    face_bytes += bytes(glm.transpose(capture_proj * view))

# The six face matrices are constant, but uniform buffers ride the per-frame ring
# (allocate_frame_set), so keep it DYNAMIC and re-upload the (tiny) block each
# frame — the current frame's slot must hold it.
faces_ubo = ctx.create_buffer(np.frombuffer(face_bytes, np.float32).copy(),
                              bz.BufferType.UNIFORM, bz.MemoryUsage.DYNAMIC)

pool = ctx.create_descriptor_pool(max_sets=4, uniform_buffers=4, samplers=2)
capture_set = pool.allocate_frame_set(capture_pipe, set=0)
capture_set.set_buffer(0, faces_ubo)
reflect_set = pool.allocate_set(reflect_pipe, set=0)
reflect_set.set_image(0, env.color[0], sampler=ctx.create_sampler(filter=bz.Filter.LINEAR))


def record(cmd, eye, camera_vp):
    cmd.begin()
    faces_ubo.update(face_bytes)
    # Capture: ONE pass, all six faces (multiview).
    with cmd.rendering(env.all_layers(), clear_color=[0, 0, 0, 1]) as c:
        (c.bind_pipeline(capture_pipe)
          .bind_descriptor_set(capture_set, capture_pipe, set=0)
          .bind_vertex_buffer(room_vbuf)
          .bind_index_buffer(room_ibuf)
          .draw_indexed(room_count))

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

TITLE = "Bazalt Demo - Multiview Environment Capture"
start = time.time()
last_time = start
frame_count = 0
fps_timer = 0.0
while window.is_open():
    window.poll_events()
    frame = renderer.begin_frame()
    if frame is None:
        continue

    now = time.time()
    frame_count += 1
    fps_timer += now - last_time
    last_time = now
    if fps_timer >= 1.0:
        fps = frame_count / fps_timer
        window.set_title(f"{TITLE} | {1000.0 / fps:.2f} ms/frame | {fps:.1f} FPS")
        frame_count = 0
        fps_timer = 0.0

    t = (now - start) * 0.5
    eye = glm.vec3(5.0 * math.cos(t), 2.5, 5.0 * math.sin(t))
    view = glm.lookAt(eye, glm.vec3(0.0), glm.vec3(0.0, 1.0, 0.0))
    record(cmd, eye, proj * view)
    frame.submit(cmd)
