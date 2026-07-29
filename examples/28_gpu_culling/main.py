"""Indirect draw — GPU frustum culling, with no readback.

20,000 cubes in a slab, and the CPU issues exactly one draw call per frame without
ever learning how many of them are on screen. Each frame:

  1. `cmd.fill_buffer(args, 0)` zeroes the draw arguments. This is the prerequisite
     that landed in 0.18 for exactly this job — a counter an atomic increments has
     to start each frame at a known value.
  2. A compute pass tests every cube against the frustum planes it pulls out of the
     view-projection matrix, atomically increments `instanceCount`, and writes the
     survivors into a compacted buffer.
  3. `cmd.draw_indexed_indirect(args)` draws whatever that came to.

There is no readback anywhere in the loop, which is the whole point. A CPU-side
version has to either stall for the count or upload a per-instance buffer it cannot
size — and the moment the culling result comes back a frame late, it is wrong.

One draw command whose instanceCount the GPU accumulates, rather than N commands,
because a GPU-decided draw COUNT needs vkCmdDrawIndexedIndirectCount and a feature
bit in a pNext struct bazalt's feature table cannot reach yet. This shape needs no
extension at all.

Keys: SPACE pauses the camera, C toggles culling off (watch the frame time).
"""

import struct
import time

import glm
import numpy as np

import bazalt as bz

logger = bz.Logger()
logger.on_message(lambda msg: print(f"[{msg.severity}] {msg.text}"))

window = bz.Window(1024, 720, "Bazalt Demo - GPU culling")
ctx = bz.Context(logger, gpu_timing=True)
renderer = bz.SwapchainRenderer(window, ctx)

COUNT = 20000
INDEX_COUNT = 36

# ── the candidates ────────────────────────────────────────────────────────
# xyz = centre, w = radius. A wide flat slab, so turning the camera changes how
# many are inside the frustum by a lot.
rng = np.random.default_rng(7)
spheres = np.empty((COUNT, 4), dtype=np.float32)
spheres[:, 0] = rng.uniform(-120.0, 120.0, COUNT)
spheres[:, 1] = rng.uniform(-6.0, 6.0, COUNT)
spheres[:, 2] = rng.uniform(-120.0, 120.0, COUNT)
spheres[:, 3] = rng.uniform(0.35, 0.9, COUNT)

candidates = ctx.create_buffer(spheres, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
# The compacted survivors. Sized for the worst case (everything visible), because
# the CPU cannot know the real number — that is the point of the example.
visible = ctx.create_buffer(COUNT * 16, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
# One VkDrawIndexedIndirectCommand: 5 words.
args = ctx.create_buffer(20, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)

# ── a unit cube ───────────────────────────────────────────────────────────
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
indices = np.array([
    0, 1, 2, 2, 3, 0,
    5, 4, 7, 7, 6, 5,
    8, 9, 10, 10, 11, 8,
    13, 12, 15, 15, 14, 13,
    16, 17, 18, 18, 19, 16,
    23, 22, 21, 21, 20, 23,
], dtype=np.uint32)
vbuf = ctx.create_buffer(vertices, bz.BufferType.VERTEX, bz.MemoryUsage.STATIC)
ibuf = ctx.create_buffer(indices, bz.BufferType.INDEX, bz.MemoryUsage.STATIC)

# ── pipelines ─────────────────────────────────────────────────────────────
comp = ctx.compile_shader("cull.comp", bz.ShaderStage.COMPUTE)
cull = (ctx.compute_pipeline()
        .shader(comp)
        .storage_buffer(0, set=0)   # draw arguments
        .storage_buffer(1, set=0)   # candidates
        .storage_buffer(2, set=0)   # compacted survivors
        .push_constant(72)          # mat4 + 2 uints
        .build())

vert = ctx.compile_shader("cube.vert", bz.ShaderStage.VERTEX)
frag = ctx.compile_shader("cube.frag", bz.ShaderStage.FRAGMENT)
draw = (ctx.graphics_pipeline()
        .vertex_shader(vert)
        .fragment_shader(frag)
        .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
        .storage_buffer(0, bz.ShaderStage.VERTEX, set=0)
        .push_constant(64, bz.ShaderStage.VERTEX)
        .depth_test(True)
        .cull_mode(bz.CullMode.BACK, bz.FrontFace.COUNTER_CLOCKWISE)
        .build(renderer))

pool = ctx.create_descriptor_pool(max_sets=4, storage_buffers=8)
cull_set = pool.allocate_set(cull, set=0)
cull_set.set_buffer(0, args)
cull_set.set_buffer(1, candidates)
cull_set.set_buffer(2, visible)
draw_set = pool.allocate_set(draw, set=0)
draw_set.set_buffer(0, visible)

# Culling off: every candidate is "visible", so the compute pass compacts nothing
# and the draw runs over all of them. Uploaded once, since it never changes.
all_visible = ctx.create_buffer(spheres, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
no_cull_args = ctx.create_buffer(
    np.array([INDEX_COUNT, COUNT, 0, 0, 0], dtype=np.uint32),
    bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
no_cull_set = pool.allocate_set(draw, set=0)
no_cull_set.set_buffer(0, all_visible)

culling = True
paused = False

proj = glm.perspectiveRH_ZO(glm.radians(60.0), 1024.0 / 720.0, 0.5, 150.0)
proj[1][1] *= -1
start = time.time()
angle = 0.0
last = start
frames = 0
fps_timer = start

while window.is_open():
    bz.poll_events()

    if window.was_key_pressed(bz.KEY_SPACE):
        paused = not paused
    if window.was_key_pressed(bz.KEY_C):
        culling = not culling

    now = time.time()
    if not paused:
        angle += (now - last) * 0.25
    last = now

    ctx.begin_frame()
    if not renderer.acquire():
        continue

    eye = glm.vec3(0.0, 3.0, 0.0)
    look = glm.vec3(glm.cos(angle), 0.0, glm.sin(angle))
    view = glm.lookAt(eye, eye + look, glm.vec3(0, 1, 0))
    view_proj = proj * view
    # Transposed once: GLSL reads column-major, and the compute shader pulls the
    # frustum planes out of the same matrix the vertex shader transforms with.
    vp_bytes = bytes(glm.transpose(view_proj))

    cmd = ctx.create_command_buffer()
    cmd.begin()

    if culling:
        # 1. Zero the arguments so instanceCount starts from nothing.
        cmd.fill_buffer(args, 0)
        # 2. Decide. No barrier by hand: the tracker orders the compute write
        #    against the command processor's read of the same buffer, and hoists
        #    that barrier out of the rendering scope on its own.
        cmd.bind_pipeline(cull)
        cmd.bind_descriptor_set(cull_set, cull, set=0)
        cmd.push_constants(cull, 0, vp_bytes + struct.pack("II", COUNT, INDEX_COUNT))
        cmd.dispatch((COUNT + 63) // 64)

    # 3. Obey. One call, whatever the count turned out to be.
    with cmd.rendering(renderer, clear_color=[0.03, 0.04, 0.07, 1.0]) as c:
        c.bind_pipeline(draw)
        c.bind_descriptor_set(draw_set if culling else no_cull_set, draw, set=0)
        c.push_constants(draw, 0, vp_bytes)
        c.bind_vertex_buffer(vbuf).bind_index_buffer(ibuf)
        c.draw_indexed_indirect(args if culling else no_cull_args)
    renderer.present(cmd)

    frames += 1
    if time.time() - fps_timer >= 1.0:
        # memory_stats stays flat: nothing in the loop allocates, which is the
        # claim an indirect frame makes. A version that resized a per-instance
        # buffer per frame would climb here.
        mb = ctx.memory_stats().used / (1024 * 1024)
        window.set_title(
            f"Bazalt Demo - GPU culling | {COUNT} cubes | "
            f"culling {'on' if culling else 'OFF'} | {frames} FPS | {mb:.1f} MB")
        frames = 0
        fps_timer = time.time()
