"""Triangle strips and fans, and the restart index that separates them (0.25).

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

Press F for the other topology. A FAN shares its FIRST vertex with every triangle
it draws — 0,1,2 then 0,2,3 then 0,3,4 — which is a pie, not a ribbon, so the fan
mode draws three flower heads instead of three blades. Look at the wireframe: in
a strip the diagonals zigzag along the blade, and in a fan every edge runs back to
the middle.

Restart separates fans exactly as it separates strips, and the failure is louder:
with R off, the second and third flowers keep radiating from the FIRST flower's
centre, so the screen fills with spokes stretched across it.

A fan is the one topology that is not universal. Metal has none, so MoltenVK
reports `Feature.TRIANGLE_FANS` missing and the pipeline build says so instead of
drawing something else. Where it answers False, the same shape is an indexed
TRIANGLE_LIST.

    R    restart on/off
    F    strip / fan
    W    wireframe (needs Feature.WIREFRAME)
    ESC  quit
"""

import math
import struct
import time

import numpy as np

import bazalt as bz

logger = bz.Logger()
logger.on_message(lambda msg: print(f"[{msg.severity}] {msg.text}"))

window = bz.Window(900, 600, "Bazalt Demo - Primitive restart", logger=logger)
ctx = bz.Context(logger, optional=[bz.Feature.WIREFRAME])
renderer = ctx.create_renderer(window)

# ── the geometry ──────────────────────────────────────────────────────────
# Each topology gets the shape it is FOR, in one vertex buffer. A strip wants a
# ribbon and a fan wants a pie, and drawing one with the other's vertex order is
# how a demo teaches nothing.

COUNT = 3
SEGMENTS = 7
RIM = 12
RESTART = 0xFFFFFFFF

positions = []
group_ids = []
strip_ranges = []
fan_ranges = []


def add(x, y, group):
    positions.append((x, y))
    group_ids.append(float(group))


# A blade is SEGMENTS quads: two vertices per row, left then right, tapering to a
# point. That IS strip order, so its indices are just start..end.
for blade in range(COUNT):
    base_x = -0.45 + blade * 0.45
    start = len(positions)
    for segment in range(SEGMENTS + 1):
        t = segment / SEGMENTS
        y = -0.8 + t * 1.5
        half_width = 0.055 * (1.0 - t) ** 1.4
        add(base_x - half_width, y, blade)
        add(base_x + half_width, y, blade)
    strip_ranges.append((start, len(positions)))

# A flower head is a centre followed by its rim. Every triangle a fan makes uses
# that first vertex, which is exactly what a pie slice is.
for flower in range(COUNT):
    centre_x = -0.45 + flower * 0.45
    start = len(positions)
    add(centre_x, 0.15, flower)
    for point in range(RIM + 1):
        angle = (point / RIM) * math.tau
        add(centre_x + math.cos(angle) * 0.20, 0.15 + math.sin(angle) * 0.20, flower)
    fan_ranges.append((start, len(positions)))

vertices = np.empty((len(positions), 3), dtype=np.float32)
vertices[:, 0:2] = np.array(positions, dtype=np.float32)
vertices[:, 2] = np.array(group_ids, dtype=np.float32)
vbuf = ctx.create_buffer(vertices, bz.BufferType.VERTEX, bz.MemoryUsage.STATIC)


def index_buffer(ranges, separate):
    """One draw either way. With `separate`, a sentinel between the shapes ends
    each primitive; without it they run into one another, which is the picture
    restart exists to prevent."""
    indices = []
    for i, (start, end) in enumerate(ranges):
        if i and separate:
            indices.append(RESTART)
        indices += list(range(start, end))
    return (ctx.create_buffer(np.array(indices, dtype=np.uint32),
                              bz.BufferType.INDEX, bz.MemoryUsage.STATIC),
            len(indices))


INDICES = {
    (True, True): index_buffer(strip_ranges, True),
    (False, True): index_buffer(strip_ranges, False),
    (True, False): index_buffer(fan_ranges, True),
    (False, False): index_buffer(fan_ranges, False),
}


def pipeline(restart, topology, polygon_mode):
    return (ctx.graphics_pipeline()
            .vertex_shader(ctx.compile_shader("grass.vert", bz.ShaderStage.VERTEX))
            .fragment_shader(ctx.compile_shader("grass.frag", bz.ShaderStage.FRAGMENT))
            .vertex_format([bz.VertexFormat.FLOAT2, bz.VertexFormat.FLOAT])
            .topology(topology, restart=restart)
            # A blade seen from behind is still a blade, and a swaying strip
            # changes which way it winds.
            .cull_mode(bz.CullMode.NONE)
            .polygon_mode(polygon_mode)
            .push_constant(8, bz.ShaderStage.VERTEX)
            .build(renderer))


HAS_WIREFRAME = ctx.supports(bz.Feature.WIREFRAME)
HAS_FANS = ctx.supports(bz.Feature.TRIANGLE_FANS)
FILL = bz.PolygonMode.FILL
LINE = bz.PolygonMode.LINE
STRIP = bz.Topology.TRIANGLE_STRIP
FAN = bz.Topology.TRIANGLE_FAN

# One pipeline per combination, built up front: a topology and a polygon mode are
# both baked into a pipeline, so switching them at run time IS switching pipeline.
topologies = [STRIP] + ([FAN] if HAS_FANS else [])
pipelines = {
    (r, t, w): pipeline(r, t, LINE if w else FILL)
    for r in (True, False)
    for t in topologies
    for w in ((False, True) if HAS_WIREFRAME else (False,))
}

print(__doc__)
if not HAS_WIREFRAME:
    print("This GPU has no fillModeNonSolid, so the W key does nothing.")
if not HAS_FANS:
    print("This driver reports no triangleFans (Metal has none), so the F key does nothing.")
print(f"{COUNT} blades of {SEGMENTS} segments, and {COUNT} flowers of {RIM} slices: "
      f"{INDICES[(True, True)][1]} strip indices with restart, {INDICES[(False, True)][1]} without.")

restart = True
wireframe = False
topology = STRIP
start_time = time.perf_counter()

while window.is_open():
    bz.poll_events()
    if window.is_key_pressed(bz.Key.ESCAPE):
        break
    if window.was_key_pressed(bz.Key.R):
        restart = not restart
        print("restart on: three separate shapes" if restart
              else "restart off: they run into one another")
    if window.was_key_pressed(bz.Key.W) and HAS_WIREFRAME:
        wireframe = not wireframe
    if window.was_key_pressed(bz.Key.F) and HAS_FANS:
        topology = FAN if topology is STRIP else STRIP
        print("fan: three flowers, every triangle sharing its centre" if topology is FAN
              else "strip: three blades, every vertex extending the last triangle")

    elapsed = time.perf_counter() - start_time
    indices, index_count = INDICES[(restart, topology is STRIP)]

    with ctx.record() as cmd:
        with cmd.rendering(renderer, clear_color=[0.05, 0.07, 0.10, 1.0]):
            cmd.bind_pipeline(pipelines[(restart, topology, wireframe and HAS_WIREFRAME)])
            # A flower sways less than a blade: its root is its middle, not its
            # bottom, so the same bend would shear it in half.
            cmd.push_constants(0, struct.pack("2f", elapsed, 0.18 if topology is STRIP else 0.05))
            cmd.bind_vertex_buffer(vbuf)
            cmd.bind_index_buffer(indices)
            # ONE draw either way. The only difference is whether the index
            # buffer carries a sentinel and whether the pipeline honours it.
            cmd.draw_indexed(index_count)

    ctx.begin_frame()
    if renderer.acquire():
        renderer.present(cmd)

    shape = "strip" if topology is STRIP else "fan"
    window.set_title(
        f"Bazalt Demo - Primitive restart | {shape} | "
        f"restart {'ON: 3 shapes' if restart else 'OFF: they run together'}"
        + (" | wireframe" if wireframe and HAS_WIREFRAME else ""))

cmd = None
renderer = None
window = None
