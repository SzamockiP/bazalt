"""Deferred-style rendering with MRT — a g-buffer in one pass.

Pass 1 renders a spinning cube into a RenderTarget with TWO colour
attachments of different formats (RGBA16F normals + RGBA8 albedo) and depth.
Pass 2 samples both attachments — they are ordinary bz.Image objects — and
lights the result into the window. One command buffer, recorded once.
"""

import time

import bazalt as bz
import glm
import numpy as np

WIDTH, HEIGHT = 1024, 720

logger = bz.Logger()
logger.on_message(lambda msg: print(f"[{msg.severity}] {msg.text}"))

window = bz.Window(WIDTH, HEIGHT, "Bazalt Demo - G-Buffer MRT")
ctx = bz.Context(logger)
renderer = ctx.create_renderer(window)

# The g-buffer: two colour formats in one target, plus depth.
gbuffer = ctx.create_render_target(WIDTH, HEIGHT,
                                   color=[bz.Format.RGBA16F, bz.Format.RGBA8],
                                   depth=bz.Format.D32F)

gbuf_vert = ctx.compile_shader("gbuffer.vert", bz.ShaderStage.VERTEX)
gbuf_frag = ctx.compile_shader("gbuffer.frag", bz.ShaderStage.FRAGMENT)
comp_vert = ctx.compile_shader("composite.vert", bz.ShaderStage.VERTEX)
comp_frag = ctx.compile_shader("composite.frag", bz.ShaderStage.FRAGMENT)

gbuf_pipe = (ctx.graphics_pipeline()
             .vertex_shader(gbuf_vert)
             .fragment_shader(gbuf_frag)
             .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
             .depth_test(True)
             .uniform_buffer(0, bz.ShaderStage.VERTEX, set=0)
             .build(gbuffer))

comp_pipe = (ctx.graphics_pipeline()
             .vertex_shader(comp_vert)
             .fragment_shader(comp_frag)
             .texture(0, bz.ShaderStage.FRAGMENT, set=0)
             .texture(1, bz.ShaderStage.FRAGMENT, set=0)
             .build(renderer))

# Cube geometry: pos + normal (same layout as example 04).
s = 0.6
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

vbuf = ctx.create_buffer(np.array(verts, dtype=np.float32),
                         bz.BufferType.VERTEX, bz.MemoryUsage.STATIC)
ibuf = ctx.create_buffer(np.array(idx, dtype=np.uint32),
                         bz.BufferType.INDEX, bz.MemoryUsage.STATIC)

# UBO: { mat4 mvp; mat4 model; }
ubuf = ctx.create_buffer(np.zeros(32, dtype=np.float32),
                         bz.BufferType.UNIFORM, bz.MemoryUsage.DYNAMIC)

pool = ctx.create_descriptor_pool(max_sets=4, uniform_buffers=4, textures=8)

gbuf_set = pool.allocate_frame_set(gbuf_pipe, set=0)
gbuf_set.set_buffer(0, ubuf)

comp_set = pool.allocate_set(comp_pipe, set=0)
comp_set.set_image(0, gbuffer.color[0])  # normals
comp_set.set_image(1, gbuffer.color[1])  # albedo

# Record once: g-buffer pass, then composite pass.
cmd = ctx.create_command_buffer()
cmd.begin()

with cmd.rendering(gbuffer, clear_color=[0.0, 0.0, 0.0, 0.0]) as c:
    (c.bind_pipeline(gbuf_pipe)
      .bind_descriptor_set(gbuf_set, gbuf_pipe, set=0)
      .bind_vertex_buffer(vbuf)
      .bind_index_buffer(ibuf)
      .draw_indexed(len(idx)))

with cmd.rendering(renderer, clear_color=[0.05, 0.07, 0.1, 1.0]) as c:
    (c.bind_pipeline(comp_pipe)
      .bind_descriptor_set(comp_set, comp_pipe, set=0)
      .draw(3))

proj = glm.perspectiveRH_ZO(glm.radians(45.0), WIDTH / HEIGHT, 0.1, 100.0)
proj[1][1] *= -1
view = glm.lookAt(glm.vec3(0.0, 1.2, 3.0), glm.vec3(0.0), glm.vec3(0.0, 1.0, 0.0))

start = time.time()
last_time = start
frame_count = 0
fps_timer = 0.0
while window.is_open():
    bz.poll_events()
    ctx.begin_frame()
    if renderer.acquire():
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        frame_count += 1
        fps_timer += dt
        if fps_timer >= 1.0:
            avg_fps = frame_count / fps_timer
            window.set_title(f"Bazalt Demo - G-Buffer MRT | {1000.0 / avg_fps:.2f} ms/frame | {avg_fps:.1f} FPS")
            frame_count = 0
            fps_timer = 0.0

        t = current_time - start
        model = glm.rotate(glm.mat4(1.0), t * 0.8, glm.vec3(0.0, 1.0, 0.0))
        model = glm.rotate(model, t * 0.3, glm.vec3(1.0, 0.0, 0.0))
        mvp = proj * view * model

        ubuf.update(bytes(glm.transpose(mvp)) + bytes(glm.transpose(model)))
        renderer.present(cmd)
