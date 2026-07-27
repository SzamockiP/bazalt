"""Instancing — 20000 cubes in one draw call.

The mesh sits in one vertex buffer and the per-object data in a second one that
advances once per INSTANCE instead of once per vertex: `instance_format(...)`
declares it, `bind_vertex_buffer(buf, binding=1)` feeds it, and
`draw_indexed(36, instances=N)` runs the same 36 indices N times.

The instance buffer is built once and never touched again — the animation is a
sine in the vertex shader keyed off each cube's own position, so the CPU sends
one matrix and a clock per frame no matter how many cubes there are.

Keys: SPACE cycles the instance count, W toggles the wireframe view.
"""

import struct
import time

import glm
import numpy as np

import bazalt as bz

COUNTS = [20000, 5000, 1, 0]

logger = bz.Logger()
logger.on_message(lambda msg: print(f"[{msg.severity}] {msg.text}"))

window = bz.Window(1024, 720, "Bazalt Demo - Instancing")
ctx = bz.Context(logger, optional=[bz.Feature.WIREFRAME], gpu_timing=True)
renderer = bz.SwapchainRenderer(window, ctx, samples=min(4, ctx.max_samples()))

vert = ctx.compile_shader("cube.vert", bz.ShaderStage.VERTEX)
frag = ctx.compile_shader("cube.frag", bz.ShaderStage.FRAGMENT)


def build(polygon_mode):
    builder = (ctx.graphics_pipeline()
        .vertex_shader(vert)
        .fragment_shader(frag)
        # Two bindings: the mesh, then the per-instance row.
        .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
        .instance_format([bz.VertexFormat.FLOAT3,       # offset
                          bz.VertexFormat.FLOAT,        # scale
                          bz.VertexFormat.UBYTE4_NORM]) # tint, 4 bytes
        .depth_test(True)
        .cull_mode(bz.CullMode.BACK, bz.FrontFace.COUNTER_CLOCKWISE)
        .uniform_buffer(0, bz.ShaderStage.VERTEX, set=0))
    if polygon_mode is not None:
        builder = builder.polygon_mode(polygon_mode)
    return builder.build(renderer)


solid = build(None)
wireframe = build(bz.PolygonMode.LINE) if ctx.supports(bz.Feature.WIREFRAME) else solid

# ── the mesh: one cube, position + normal ─────────────────────────────────
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

# ── the instances: a grid, packed by hand ─────────────────────────────────
# 20 bytes each: 3 floats of offset, 1 float of scale, 4 bytes of colour. The
# tint costs four bytes instead of the sixteen a FLOAT4 would take, which is the
# reason UBYTE4_NORM exists.
rng = np.random.default_rng(7)
side = int(np.ceil(np.sqrt(COUNTS[0])))
rows = []
for i in range(COUNTS[0]):
    x = (i % side - side / 2) * 1.6
    z = (i // side - side / 2) * 1.6
    r, g, b = (rng.integers(60, 256), rng.integers(60, 256), rng.integers(60, 256))
    rows.append(struct.pack("<4f4B", x, 0.0, z, float(rng.uniform(0.3, 0.8)), r, g, b, 255))
instances = ctx.create_buffer(np.frombuffer(b"".join(rows), dtype=np.uint8),
                              bz.BufferType.VERTEX, bz.MemoryUsage.STATIC,
                              name="instances")
print(f"instance buffer: {len(rows) * 20 / 1024:.0f} KiB for {len(rows)} cubes")

ubuf = ctx.create_buffer(20 * 4, bz.BufferType.UNIFORM, bz.MemoryUsage.DYNAMIC)
pool = ctx.create_descriptor_pool(max_sets=2, uniform_buffers=2)
desc_set = pool.allocate_frame_set(solid, set=0)
desc_set.set_buffer(0, ubuf)


def record(pipeline, count):
    """One recording per (pipeline, count) pair — the instance count is baked
    into the recorded draw, so changing it re-records."""
    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(renderer, clear_color=[0.02, 0.02, 0.05, 1.0]) as c:
        (c.bind_pipeline(pipeline)
          .bind_descriptor_set(desc_set, pipeline, set=0)
          .bind_vertex_buffer(vbuf)
          .bind_vertex_buffer(instances, binding=1)
          .bind_index_buffer(ibuf)
          .draw_indexed(36, instances=count))
    return cmd


count_index = 0
wire = False
cmd = record(solid, COUNTS[0])

proj = glm.perspectiveRH_ZO(glm.radians(60.0), 1024.0 / 720.0, 0.1, 400.0)
proj[1][1] *= -1

start = time.time()
frames = 0
fps_timer = time.time()

while window.is_open():
    bz.poll_events()

    if window.was_key_pressed(bz.KEY_SPACE):
        count_index = (count_index + 1) % len(COUNTS)
        cmd = record(wireframe if wire else solid, COUNTS[count_index])
    if window.was_key_pressed(bz.KEY_W):
        wire = not wire
        cmd = record(wireframe if wire else solid, COUNTS[count_index])

    ctx.begin_frame()
    if renderer.acquire():
        t = time.time() - start
        eye = glm.vec3(np.sin(t * 0.15) * 60.0, 25.0, np.cos(t * 0.15) * 60.0)
        view_proj = proj * glm.lookAt(eye, glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
        ubuf.update(bytes(glm.transpose(view_proj)) + struct.pack("4f", t, 0.0, 0.0, 0.0))
        renderer.present(cmd)

        frames += 1
        if time.time() - fps_timer >= 1.0:
            gpu = renderer.gpu_time_ms
            gpu_text = f"{gpu:.2f} ms GPU" if gpu else "GPU time n/a"
            window.set_title(
                f"Bazalt Demo - Instancing | {COUNTS[count_index]} cubes, one draw | "
                f"{frames} FPS | {gpu_text}")
            frames = 0
            fps_timer = time.time()
