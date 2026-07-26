"""Compute-driven particles.

One STORAGE buffer is both the simulation state and the vertex buffer: the
compute shader integrates {pos, vel} pairs in place, and the draw fetches
them directly (STORAGE buffers carry VERTEX usage since 0.6). The barrier
between the dispatch's writes and the vertex fetch is recorded automatically
— the command buffer is recorded once and replayed every frame.
"""

import math
import random
import struct
import time

import bazalt as bz

N = 4096

logger = bz.Logger()

@logger.on_message
def on_message(msg):
    print(f"[{msg.severity}] {msg.text}")

window = bz.Window(1024, 720, "Bazalt Demo - Compute Particles", logger=logger)
ctx = bz.Context(logger)
renderer = bz.SwapchainRenderer(window, ctx)

# Simulation: a compute pipeline. No stage arguments, no target.
sim_shader = ctx.compile_shader("particles.comp", bz.ShaderStage.COMPUTE)
sim = (ctx.compute_pipeline()
    .shader(sim_shader)
    .storage_buffer(0)
    .push_constant(4)
    .build())

# Presentation: points, one per particle.
vert = ctx.compile_shader("particles.vert", bz.ShaderStage.VERTEX)
frag = ctx.compile_shader("particles.frag", bz.ShaderStage.FRAGMENT)
draw = (ctx.graphics_pipeline()
    .vertex_shader(vert)
    .fragment_shader(frag)
    .vertex_format([bz.VertexFormat.FLOAT2, bz.VertexFormat.FLOAT2])
    .topology(bz.Topology.POINT_LIST)
    .build(renderer))

# Interleaved {pos.xy, vel.xy} per particle.
state = []
for _ in range(N):
    angle = random.uniform(0.0, math.tau)
    speed = random.uniform(0.05, 0.6)
    state += [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0),
              math.cos(angle) * speed, math.sin(angle) * speed]
particles = ctx.create_buffer(state, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC,
                              bz.DataType.FLOAT)

pool = ctx.create_descriptor_pool(max_sets=4, storage_buffers=4)
sim_set = pool.allocate_set(sim, set=0)
sim_set.set_buffer(0, particles)

# Recorded once. The dispatch -> vertex-fetch barrier is hoisted before the
# rendering scope automatically; replay-to-replay ordering is handled too.
cmd = ctx.create_command_buffer()
cmd.begin()
(cmd.bind_pipeline(sim)
    .bind_descriptor_set(sim_set, sim, set=0)
    .push_constants(sim, 0, struct.pack("<f", 1.0 / 60.0))
    .dispatch((N + 63) // 64))
with cmd.rendering(renderer, clear_color=[0.02, 0.02, 0.05, 1.0]) as c:
    c.bind_pipeline(draw).bind_vertex_buffer(particles).draw(N)

last_time = time.time()
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
            window.set_title(f"Bazalt Demo - Compute Particles | {1000.0 / avg_fps:.2f} ms/frame | {avg_fps:.1f} FPS")
            frame_count = 0
            fps_timer = 0.0

        renderer.present(cmd)
