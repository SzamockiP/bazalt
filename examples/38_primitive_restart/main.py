"""Triangle strips, and the restart index that separates them (0.25).

Three blades of grass. Each one is a triangle STRIP: every new vertex extends the
blade by one triangle instead of starting a new one, so a blade of 7 triangles is
9 vertices rather than 21. That is what a strip is for, and it is also the problem
this example is about — a strip has no end.

`topology(TRIANGLE_STRIP, restart=True)` gives it one. The largest index value
(0xFFFFFFFF for 32-bit indices) stops the current strip and starts another, so all
three blades are ONE indexed draw.

Press R to turn it off. The same vertices, the same one draw, but with no
sentinel to stop at: the strip runs on from the tip of one blade into the root of
the next, and the ribbon between them is the thing restart exists to prevent. The
alternatives without it are three draws, or degenerate triangles nobody wants to
write by hand.

It is opt-in for a reason. With it on, an index of 0xFFFFFFFF is no longer an
index — a mesh that happens to use the last vertex of a 4-billion-vertex buffer
would be cut in two, silently. The caller who asked for restart is the one who
knows their indices do not go there.

    R    restart on/off
    W    wireframe (needs Feature.WIREFRAME)
    ESC  quit
"""

import struct
import time

import numpy as np

import bazalt as bz

logger = bz.Logger()
logger.on_message(lambda msg: print(f"[{msg.severity}] {msg.text}"))

window = bz.Window(900, 600, "Bazalt Demo - Primitive restart", logger=logger)
ctx = bz.Context(logger, optional=[bz.Feature.WIREFRAME])
renderer = ctx.create_renderer(window)

# ── three blades, each a strip ────────────────────────────────────────────
# A blade is SEGMENTS quads: two vertices per row, left then right, tapering to a
# point. In strip order that is exactly the order they are written, so the index
# buffer for one blade is just 0, 1, 2, 3, ...

BLADES = 3
SEGMENTS = 7
RESTART = 0xFFFFFFFF

positions = []
blade_ids = []
blade_ranges = []

for blade in range(BLADES):
    base_x = -0.45 + blade * 0.45
    start = len(positions) // 2
    for segment in range(SEGMENTS + 1):
        t = segment / SEGMENTS
        y = -0.8 + t * 1.5
        half_width = 0.055 * (1.0 - t) ** 1.4
        positions += [base_x - half_width, y,
                      base_x + half_width, y]
        blade_ids += [float(blade), float(blade)]
    blade_ranges.append((start, len(positions) // 2))

vertices = np.empty((len(blade_ids), 3), dtype=np.float32)
vertices[:, 0:2] = np.array(positions, dtype=np.float32).reshape(-1, 2)
vertices[:, 2] = np.array(blade_ids, dtype=np.float32)
vbuf = ctx.create_buffer(vertices, bz.BufferType.VERTEX, bz.MemoryUsage.STATIC)

# With the sentinel between blades: one draw, three separate strips.
separated = []
for i, (start, end) in enumerate(blade_ranges):
    if i:
        separated.append(RESTART)
    separated += list(range(start, end))
separated_ibuf = ctx.create_buffer(
    np.array(separated, dtype=np.uint32), bz.BufferType.INDEX, bz.MemoryUsage.STATIC)

# Without it: the same vertices in the same order, and the strip never stops.
joined = []
for start, end in blade_ranges:
    joined += list(range(start, end))
joined_ibuf = ctx.create_buffer(
    np.array(joined, dtype=np.uint32), bz.BufferType.INDEX, bz.MemoryUsage.STATIC)


def pipeline(restart, polygon_mode):
    return (ctx.graphics_pipeline()
            .vertex_shader(ctx.compile_shader("grass.vert", bz.ShaderStage.VERTEX))
            .fragment_shader(ctx.compile_shader("grass.frag", bz.ShaderStage.FRAGMENT))
            .vertex_format([bz.VertexFormat.FLOAT2, bz.VertexFormat.FLOAT])
            .topology(bz.Topology.TRIANGLE_STRIP, restart=restart)
            # A blade seen from behind is still a blade, and a swaying strip
            # changes which way it winds.
            .cull_mode(bz.CullMode.NONE)
            .polygon_mode(polygon_mode)
            .push_constant(8, bz.ShaderStage.VERTEX)
            .build(renderer))


HAS_WIREFRAME = ctx.supports(bz.Feature.WIREFRAME)
FILL = bz.PolygonMode.FILL
LINE = bz.PolygonMode.LINE

pipelines = {(True, False): pipeline(True, FILL), (False, False): pipeline(False, FILL)}
if HAS_WIREFRAME:
    pipelines[(True, True)] = pipeline(True, LINE)
    pipelines[(False, True)] = pipeline(False, LINE)

print(__doc__)
if not HAS_WIREFRAME:
    print("This GPU has no fillModeNonSolid, so the W key does nothing.")
print(f"{BLADES} blades, {SEGMENTS} segments each: "
      f"{len(separated)} indices with restart, {len(joined)} without.")

restart = True
wireframe = False
start_time = time.perf_counter()

while window.is_open():
    bz.poll_events()
    if window.is_key_pressed(bz.Key.ESCAPE):
        break
    if window.was_key_pressed(bz.Key.R):
        restart = not restart
        print("restart on: three blades" if restart else "restart off: one ribbon")
    if window.was_key_pressed(bz.Key.W) and HAS_WIREFRAME:
        wireframe = not wireframe

    elapsed = time.perf_counter() - start_time
    indices = separated_ibuf if restart else joined_ibuf

    with ctx.record() as cmd:
        with cmd.rendering(renderer, clear_color=[0.05, 0.07, 0.10, 1.0]):
            cmd.bind_pipeline(pipelines[(restart, wireframe and HAS_WIREFRAME)])
            cmd.push_constants(0, struct.pack("2f", elapsed, 0.18))
            cmd.bind_vertex_buffer(vbuf)
            cmd.bind_index_buffer(indices)
            # ONE draw either way. The only difference is whether the index
            # buffer carries a sentinel and whether the pipeline honours it.
            cmd.draw_indexed(len(separated) if restart else len(joined))

    ctx.begin_frame()
    if renderer.acquire():
        renderer.present(cmd)

    window.set_title(
        f"Bazalt Demo - Primitive restart | restart {'ON: 3 blades' if restart else 'OFF: 1 ribbon'}"
        + (" | wireframe" if wireframe and HAS_WIREFRAME else ""))

cmd = None
renderer = None
window = None
