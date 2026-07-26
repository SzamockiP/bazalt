"""Shadow mapping — two passes, one command buffer, zero shadow-specific API.

Pass 1 renders the scene into a depth-only RenderTarget from the light's view
(no fragment shader). Pass 2 renders to the window and samples `shadow.depth`
like any other texture. The command buffer is recorded once and replays both
passes every frame; only the camera UBO changes.
"""

import math
import time

import bazalt as bz
import glm
import numpy as np

SHADOW_SIZE = 2048

logger = bz.Logger()
logger.on_message(lambda msg: print(f"[{msg.severity}] {msg.text}"))

window = bz.Window(1024, 720, "Bazalt Demo - Shadow Map")
ctx = bz.Context(logger)
renderer = bz.SwapchainRenderer(window, ctx)

# The shadow map: a depth-only target. Its .depth is an ordinary bz.Image.
shadow = bz.RenderTarget(ctx, SHADOW_SIZE, SHADOW_SIZE, color=None, depth=bz.Format.D32F)

shadow_vert = ctx.compile_shader("shadow.vert", bz.ShaderStage.VERTEX)
scene_vert = ctx.compile_shader("scene.vert", bz.ShaderStage.VERTEX)
scene_frag = ctx.compile_shader("scene.frag", bz.ShaderStage.FRAGMENT)

# Depth-only pipeline: no fragment shader (legal because the target has no
# colour attachments).
shadow_pipe = (ctx.graphics_pipeline()
               .vertex_shader(shadow_vert)
               .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
               .depth_test(True)
               .uniform_buffer(0, bz.ShaderStage.VERTEX, set=0)
               .build(shadow))

scene_pipe = (ctx.graphics_pipeline()
              .vertex_shader(scene_vert)
              .fragment_shader(scene_frag)
              .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
              .depth_test(True)
              .uniform_buffer(0, bz.ShaderStage.VERTEX, set=0)
              .texture(1, bz.ShaderStage.FRAGMENT, set=0)
              .build(renderer))


def cube(cx, cy, cz, s):
    """24 pos+normal vertices and 36 indices for an axis-aligned cube."""
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
            verts += [x + cx, y + cy, z + cz, *normal]
        idx += [base, base + 1, base + 2, base + 2, base + 3, base]
    return verts, idx


# Scene: a ground plane and a floating cube that casts the shadow.
verts, idx = cube(0.0, 0.75, 0.0, 0.5)
ground_base = len(verts) // 6
verts += [
    -3.0, -0.25, -3.0, 0.0, 1.0, 0.0,
    -3.0, -0.25, +3.0, 0.0, 1.0, 0.0,
    +3.0, -0.25, +3.0, 0.0, 1.0, 0.0,
    +3.0, -0.25, -3.0, 0.0, 1.0, 0.0,
]
idx += [ground_base, ground_base + 1, ground_base + 2,
        ground_base + 2, ground_base + 3, ground_base]

vbuf = ctx.create_buffer(np.array(verts, dtype=np.float32),
                         bz.BufferType.VERTEX, bz.MemoryUsage.STATIC)
ibuf = ctx.create_buffer(np.array(idx, dtype=np.uint32),
                         bz.BufferType.INDEX, bz.MemoryUsage.STATIC)
index_count = len(idx)

# One UBO feeds both pipelines: { mat4 cameraMvp; mat4 lightMvp; }.
ubuf = ctx.create_buffer(np.zeros(32, dtype=np.float32),
                         bz.BufferType.UNIFORM, bz.MemoryUsage.DYNAMIC)

pool = ctx.create_descriptor_pool(max_sets=4, uniform_buffers=8, samplers=4)

shadow_set = pool.allocate_frame_set(shadow_pipe, set=0)
shadow_set.set_buffer(0, ubuf)

scene_set = pool.allocate_frame_set(scene_pipe, set=0)
scene_set.set_buffer(0, ubuf)
# NEAREST: linear filtering of depth formats is not universally supported.
# LINEAR + compare = hardware PCF: the sampler compares the reference depth
# against the four nearest texels and filters the RESULTS, softening the edge.
# (Linear filtering of depth formats is a format feature — fine on desktop GPUs.)
scene_set.set_image(1, shadow.depth, sampler=ctx.create_sampler(
    filter=bz.Filter.LINEAR, compare=bz.CompareOp.LESS))

# The light never moves; its matrix is computed once.
light_proj = glm.orthoRH_ZO(-3.0, 3.0, -3.0, 3.0, 0.1, 10.0)
light_proj[1][1] *= -1
light_view = glm.lookAt(glm.vec3(2.0, 4.0, 1.0), glm.vec3(0.0), glm.vec3(0.0, 1.0, 0.0))
light_mvp = light_proj * light_view

# Record ONCE: both passes in one command buffer. begin_rendering captures its
# target per call, which is exactly what makes this possible.
cmd = ctx.create_command_buffer()
cmd.begin()

with cmd.rendering(shadow) as c:
    (c.bind_pipeline(shadow_pipe)
      .bind_descriptor_set(shadow_set, shadow_pipe, set=0)
      .bind_vertex_buffer(vbuf)
      .bind_index_buffer(ibuf)
      .draw_indexed(index_count))

with cmd.rendering(renderer, clear_color=[0.05, 0.07, 0.1, 1.0]) as c:
    (c.bind_pipeline(scene_pipe)
      .bind_descriptor_set(scene_set, scene_pipe, set=0)
      .bind_vertex_buffer(vbuf)
      .bind_index_buffer(ibuf)
      .draw_indexed(index_count))

# Main loop: the camera orbits; the shadow map re-renders every frame.
start = time.time()
last_time = start
frame_count = 0
fps_timer = 0.0
while window.is_open():
    window.poll_events()
    ctx.begin_frame()
    if renderer.acquire():
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        frame_count += 1
        fps_timer += dt
        if fps_timer >= 1.0:
            avg_fps = frame_count / fps_timer
            window.set_title(f"Bazalt Demo - Shadow Map | {1000.0 / avg_fps:.2f} ms/frame | {avg_fps:.1f} FPS")
            frame_count = 0
            fps_timer = 0.0

        t = (current_time - start) * 0.4
        eye = glm.vec3(4.0 * math.cos(t), 2.5, 4.0 * math.sin(t))
        view = glm.lookAt(eye, glm.vec3(0.0, 0.25, 0.0), glm.vec3(0.0, 1.0, 0.0))
        proj = glm.perspectiveRH_ZO(glm.radians(45.0), 1024.0 / 720.0, 0.1, 100.0)
        proj[1][1] *= -1
        camera_mvp = proj * view

        ubuf.update(bytes(glm.transpose(camera_mvp)) + bytes(glm.transpose(light_mvp)))
        renderer.present(cmd)
