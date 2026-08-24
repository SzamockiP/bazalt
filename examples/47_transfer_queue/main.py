"""The transfer queue: copies that run beside the frame.

The 0.30 headline. A discrete GPU has a transfer-only queue family — a DMA
engine that copies over PCIe while the graphics and compute queues keep
working. Since 0.30 bazalt's own uploads (create_buffer, load_image,
image.update) submit there, and `graph.add_pass(queue=bz.Queue.TRANSFER)`
puts your own copies there too.

Three things are on show:

  * `bz.Feature.ASYNC_TRANSFER` says whether the family is real. Without it
    the transfer queue is the graphics queue under its own timeline: the same
    program, the same order, no overlap. Set BAZALT_FORCE_SINGLE_QUEUE=1 to
    see that path on any machine.
  * A TRANSFER pass runs the copy vocabulary: update_buffer (a small patch
    with no staging buffer), copy_buffer_to_image (stream one mip of a
    texture), copy_buffer, fill_buffer, copy_image_to_buffer. Shaders, blits
    and clears are refused with the fix in the message.
  * The copy -> sample edge is a timeline wait the graph inserts itself: the
    sampling pass on GRAPHICS waits the TRANSFER batch, exactly as a compute
    producer is waited in example 45.

What the graph does NOT order: two different graphs, on any queues. That
stays submit(after=...) — see example 44.

Honesty box: the win is measured only on a discrete GPU, and only when the
frame is busy while the copy streams. On one family the frame costs the sum
either way (the example 46 lesson, one queue over). An upload pays no extra
submit for the move — it always had its own — which is why the uploads were
the first customer (DESIGN.md, "The queues after 0.29").

Headless. Run: python main.py
"""

import os
import sys

import numpy as np

import bazalt as bz

os.chdir(os.path.dirname(os.path.abspath(__file__)))

SIZE = 256

ctx = bz.Context()
real = ctx.supports(bz.Feature.ASYNC_TRANSFER)
print(f"device: {ctx.device_name}")
print("transfer queue:",
      "a separate transfer-only family (real overlap)" if real
      else "aliases the graphics queue (same program, no overlap)")

# The texture a TRANSFER pass streams into, and the tile it streams from. A
# procedural checkerboard, recomputed on the CPU each replay and patched into
# the staging buffer with update() — the DYNAMIC path — while the copy into
# the image rides the graph on the transfer queue.
image = ctx.create_image(SIZE, SIZE, bz.Format.RGBA8, name="streamed tile")
tile = ctx.create_buffer(SIZE * SIZE * 4, bz.BufferUsage.STORAGE,
                         bz.MemoryUsage.DYNAMIC, name="tile bytes")

target = ctx.create_render_target(512, 512)
present = (ctx.graphics_pipeline()
           .vertex_shader(ctx.compile_shader("fullscreen.vert", bz.ShaderStage.VERTEX))
           .fragment_shader(ctx.compile_shader("present.frag", bz.ShaderStage.FRAGMENT))
           .texture(0, bz.ShaderStage.FRAGMENT)
           .build(target))
dset = ctx.create_descriptor_pool().allocate_frame_set(present)
dset.set_image(0, image)


def checkerboard(step):
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    cells = ((xx // 32 + yy // 32 + step) % 2).astype(np.uint8)
    rgba = np.empty((SIZE, SIZE, 4), np.uint8)
    rgba[..., 0] = cells * 255
    rgba[..., 1] = (1 - cells) * 160
    rgba[..., 2] = 64
    rgba[..., 3] = 255
    return rgba.ravel()


# One graph, replayed. The TRANSFER pass copies the tile into the image; the
# GRAPHICS pass samples it. The edge between them crosses the queues, so the
# compile turns it into a timeline wait rather than a barrier.
g = ctx.graph()
stream = g.add_pass(name="stream", queue=bz.Queue.TRANSFER)
stream.copy_buffer_to_image(tile, image)
with g.add_pass(target, name="sample") as p:
    p.bind_pipeline(present).bind_descriptor_set(dset).draw(3)

for step in range(4):
    # begin_frame first: a DYNAMIC buffer's update() writes THIS frame's slot,
    # and the ring advances at begin_frame.
    ctx.begin_frame()
    tile.update(checkerboard(step))
    ctx.submit(g)

# The referee: the last replay's top-left cell is white when step is odd.
pixels = target.color[0].read()
corner = tuple(pixels[8, 8][:3])
assert corner == (255, 0, 64), corner
print("streamed", 4, "tiles; the copy -> sample edge held:", corner)
sys.exit(0)
