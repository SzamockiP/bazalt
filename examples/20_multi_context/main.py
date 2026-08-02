"""Multi-context — two GPUs (or two devices on one GPU), one process.

The 0.15 release feature. Until now a second Context raised: volk installs its
function pointers as process globals and bound them to a single VkDevice, so a
second Context would have silently redirected the first one's GPU calls at its
own device. Each Context now carries its own dispatch table, and the limit —
tech debt #1 since 0.5 — is gone.

What this shows is the shape the feature enables: **bake on one device, render
on the other.**

    baker  = bz.Context(device=devices[0])   # compute writes a texture
    viewer = bz.Context(device=devices[1])   # window draws with it
    texture = viewer.create_image(baked)     # the pixels cross over

`create_image(image)` is the fourth overload of the same function that already
makes images from a size, a numpy array or a list of them — a Context still just
makes images, the source is only where the pixels come from. It carries format,
layer count and cube-ness across, which `viewer.create_image(baked.read())` could
not: a numpy array has nowhere to put them.

It is a SETUP operation, not a per-frame one. Without external memory (waiting
room) the only portable route between two devices is host memory, so the call
reads back on the baker and uploads here — done once, before the loop.

Resources do not otherwise cross: hand a Context's image, pipeline or target to
the other Context's command buffer and you get a ResourceError naming the
mistake, instead of a driver crash. The bottom of this file demonstrates that.

On a single-GPU machine both Contexts land on the same card. That is not a
degraded demo — two VkDevices are two dispatch tables either way, which is
exactly what used to break.
"""

import time

import numpy as np

import bazalt as bz

logger = bz.Logger()
logger.on_message(lambda msg: print(f"[{msg.severity}] {msg.text}"))

SIZE = 512

# Two devices where the machine has them, otherwise the same one twice.
devices = bz.list_devices()
for d in devices:
    print(f"  {d.name} ({d.type}, {d.memory_mb} MB, Vulkan {d.api_version})")

baker = bz.Context(logger, device=devices[0])
viewer = bz.Context(logger, device=devices[1] if len(devices) > 1 else devices[0])
print(f"bake on:   {baker.device_name}")
print(f"render on: {viewer.device_name}")

# ── Context A: bake a procedural texture in compute ───────────────────────────

bake = baker.compile_shader("bake.comp", bz.ShaderStage.COMPUTE)
bake_pipeline = (baker.compute_pipeline()
    .shader(bake)
    .storage_image(0)
    .push_constant(4)
    .build())

baked = baker.create_image(SIZE, SIZE)
bake_pool = baker.create_descriptor_pool()
bake_set = bake_pool.allocate_set(bake_pipeline)
bake_set.set_storage_image(0, baked)

cmd = baker.create_command_buffer()
cmd.begin()
(cmd.bind_pipeline(bake_pipeline)
    .bind_descriptor_set(bake_set, bake_pipeline)
    .push_constants(bake_pipeline, 0, np.float32(1.7).tobytes())
    .dispatch(SIZE // 8, SIZE // 8))
baker.submit(cmd)

# ── The crossing ──────────────────────────────────────────────────────────────

start = time.perf_counter()
texture = viewer.create_image(baked)
texture.wait()
print(f"{SIZE}x{SIZE} moved between Contexts in {(time.perf_counter() - start) * 1000:.1f} ms")

# ── Context B: a window that draws with it ────────────────────────────────────

window = bz.Window(800, 600, "Bazalt - Multi-context", logger=logger)
renderer = viewer.create_renderer(window)

pipeline = (viewer.graphics_pipeline()
    .vertex_shader(viewer.compile_shader("quad.vert", bz.ShaderStage.VERTEX))
    .fragment_shader(viewer.compile_shader("quad.frag", bz.ShaderStage.FRAGMENT))
    .vertex_format([bz.VertexFormat.FLOAT2, bz.VertexFormat.FLOAT2])
    .texture(0, bz.ShaderStage.FRAGMENT)
    .build(renderer))

vertices = [
    -0.8, -0.8,  0.0, 0.0,
     0.8, -0.8,  1.0, 0.0,
     0.8,  0.8,  1.0, 1.0,
    -0.8,  0.8,  0.0, 1.0,
]
vbuf = viewer.create_buffer(vertices, bz.BufferType.VERTEX, bz.MemoryUsage.STATIC, bz.DataType.FLOAT)
ibuf = viewer.create_buffer([0, 3, 2, 2, 1, 0], bz.BufferType.INDEX,
                            bz.MemoryUsage.STATIC, bz.DataType.UINT32)

pool = viewer.create_descriptor_pool()
dset = pool.allocate_set(pipeline)
dset.set_image(0, texture)

draw = viewer.create_command_buffer()
draw.begin()
with draw.rendering(renderer, clear_color=[0.02, 0.02, 0.05, 1.0]) as c:
    (c.bind_pipeline(pipeline)
      .bind_descriptor_set(dset, pipeline)
      .bind_vertex_buffer(vbuf)
      .bind_index_buffer(ibuf)
      .draw_indexed(6))

# The guard, in the one line it takes to trip: `baked` lives on the baker, and
# the viewer's command buffer says so rather than handing a foreign VkImage to
# the driver.
try:
    probe = viewer.create_command_buffer()
    probe.begin()
    probe.barrier(baked, bz.Access.SHADER_READ, bz.Access.SHADER_READ)
except bz.ResourceError as error:
    print(f"as expected: {error}")

frames, fps_timer = 0, time.time()
while window.is_open():
    bz.poll_events()

    # Each Context opens its own frame. The viewer is the only one drawing per
    # frame here; the baker did its work once, before the loop.
    viewer.begin_frame()
    if renderer.acquire():
        renderer.present(draw)

    frames += 1
    if time.time() - fps_timer >= 1.0:
        window.set_title(f"Bazalt - Multi-context | {frames} FPS")
        frames, fps_timer = 0, time.time()
