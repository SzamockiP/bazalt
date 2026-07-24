"""MSAA — hardware multisampling.

A spinning cube on a dark background: its silhouette edges alias hard at
samples=1 and smooth out at samples=4. MSAA lives entirely on the renderer —
`SwapchainRenderer(window, ctx, samples=4)` — and the pipeline picks the sample
count off the target it builds against, so nothing else in the frame changes.

If the GPU exposes SAMPLE_RATE_SHADING we also turn on per-sample shading, which
cleans up interior/specular aliasing that plain edge MSAA leaves behind.
"""

import math
import time

import glm
import numpy as np

import bazalt as bz

logger = bz.Logger()
logger.on_message(lambda msg: print(f"[{msg.severity}] {msg.text}"))

window = bz.Window(1024, 720, "Bazalt Demo - MSAA")
# Ask for sample-rate shading; it's optional, so a GPU without it just skips it.
ctx = bz.Context(logger, optional=[bz.Feature.SAMPLE_RATE_SHADING])

samples = min(4, ctx.max_samples())
renderer = bz.SwapchainRenderer(window, ctx, samples=samples)
print(f"MSAA: {samples}x (GPU max {ctx.max_samples()}x)")

vert_spv = ctx.compile_shader("cube.vert", bz.ShaderStage.VERTEX)
frag_spv = ctx.compile_shader("cube.frag", bz.ShaderStage.FRAGMENT)

builder = (ctx.graphics_pipeline()
    .vertex_shader(vert_spv)
    .fragment_shader(frag_spv)
    .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
    .depth_test(True)
    .cull_mode(bz.CullMode.BACK, bz.FrontFace.COUNTER_CLOCKWISE)
    .uniform_buffer(0, bz.ShaderStage.VERTEX, set=0))

sample_shading = ctx.supports(bz.Feature.SAMPLE_RATE_SHADING) and samples > 1
if sample_shading:
    builder = builder.sample_shading(True)
print(f"Sample-rate shading: {'on' if sample_shading else 'off'}")

pipeline = builder.build(renderer)

# pos (x,y,z), normal (nx,ny,nz) — normal doubles as the face colour.
vertices = np.array([
    -0.5, -0.5,  0.5,   0.0, 0.0, 1.0,
     0.5, -0.5,  0.5,   0.0, 0.0, 1.0,
     0.5,  0.5,  0.5,   0.0, 0.0, 1.0,
    -0.5,  0.5,  0.5,   0.0, 0.0, 1.0,
    -0.5, -0.5, -0.5,   0.0, 0.0, -1.0,
     0.5, -0.5, -0.5,   0.0, 0.0, -1.0,
     0.5,  0.5, -0.5,   0.0, 0.0, -1.0,
    -0.5,  0.5, -0.5,   0.0, 0.0, -1.0,
    -0.5, -0.5, -0.5,  -1.0, 0.0, 0.0,
    -0.5, -0.5,  0.5,  -1.0, 0.0, 0.0,
    -0.5,  0.5,  0.5,  -1.0, 0.0, 0.0,
    -0.5,  0.5, -0.5,  -1.0, 0.0, 0.0,
     0.5, -0.5, -0.5,   1.0, 0.0, 0.0,
     0.5, -0.5,  0.5,   1.0, 0.0, 0.0,
     0.5,  0.5,  0.5,   1.0, 0.0, 0.0,
     0.5,  0.5, -0.5,   1.0, 0.0, 0.0,
    -0.5, -0.5, -0.5,   0.0, -1.0, 0.0,
     0.5, -0.5, -0.5,   0.0, -1.0, 0.0,
     0.5, -0.5,  0.5,   0.0, -1.0, 0.0,
    -0.5, -0.5,  0.5,   0.0, -1.0, 0.0,
    -0.5,  0.5, -0.5,   0.0, 1.0, 0.0,
     0.5,  0.5, -0.5,   0.0, 1.0, 0.0,
     0.5,  0.5,  0.5,   0.0, 1.0, 0.0,
    -0.5,  0.5,  0.5,   0.0, 1.0, 0.0,
], dtype=np.float32)
vbuf = ctx.create_buffer(vertices, bz.BufferType.VERTEX, bz.MemoryUsage.STATIC)

indices = np.array([
    0, 1, 2, 2, 3, 0,
    5, 4, 7, 7, 6, 5,
    8, 9, 10, 10, 11, 8,
    13, 12, 15, 15, 14, 13,
    16, 17, 18, 18, 19, 16,
    23, 22, 21, 21, 20, 23,
], dtype=np.uint32)
ibuf = ctx.create_buffer(indices, bz.BufferType.INDEX, bz.MemoryUsage.STATIC)

ubuf = ctx.create_buffer(np.zeros(16, dtype=np.float32), bz.BufferType.UNIFORM, bz.MemoryUsage.DYNAMIC)

pool = ctx.create_descriptor_pool(max_sets=2, uniform_buffers=2)
desc_set = pool.allocate_frame_set(pipeline, set=0)
desc_set.set_buffer(0, ubuf)

cmd = ctx.create_command_buffer()
cmd.begin()
with cmd.rendering(renderer, clear_color=[0.02, 0.02, 0.04, 1.0]) as c:
    (c.bind_pipeline(pipeline)
      .bind_descriptor_set(desc_set, pipeline, set=0)
      .bind_vertex_buffer(vbuf)
      .bind_index_buffer(ibuf)
      .draw_indexed(36))

proj = glm.perspectiveRH_ZO(glm.radians(45.0), 1024.0 / 720.0, 0.1, 100.0)
proj[1][1] *= -1
view = glm.lookAt(glm.vec3(0, 0, 3), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))

start = time.time()
frame_count = 0
fps_timer = time.time()

while window.is_open():
    window.poll_events()
    if frame := renderer.begin_frame():
        t = time.time() - start
        model = glm.rotate(glm.mat4(1.0), t * 0.7, glm.vec3(0.3, 1.0, 0.2))
        mvp = proj * view * model
        ubuf.update(bytes(glm.transpose(mvp)))
        frame.submit(cmd)

        frame_count += 1
        if time.time() - fps_timer >= 1.0:
            window.set_title(f"Bazalt Demo - MSAA {samples}x | {frame_count} FPS")
            frame_count = 0
            fps_timer = time.time()
