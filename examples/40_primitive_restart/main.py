"""Primitive restart — many strips in ONE draw (0.25).

`examples/38_topologies` draws three blades with three topologies, which is three
pipelines and three draws. This one asks the opposite question: the three blades
are the same topology, so can they be one draw?

A strip has no end. Every new vertex extends it, which is what makes a strip
cheap, and it means the vertices of the second blade continue the first.

`topology(TRIANGLE_STRIP, restart=True)` gives it an end. The largest index value
— 0xFFFFFFFF for 32-bit indices — stops the current strip, and the next index
starts a new one. So all three blades become one `draw_indexed`.

Press R to drop the sentinel. Same vertices, same one draw, but the strip runs on
from the tip of one blade into the root of the next, and the ribbon it stretches
between them is the thing restart exists to prevent. The alternatives are three
draws, or degenerate triangles nobody wants to write by hand.

It is opt-in for a reason. With it on, 0xFFFFFFFF is no longer an index — a mesh
that happens to use the last vertex of a four-billion-vertex buffer would be cut
in two, silently. The caller who asks for restart is the one who knows their
indices do not go there.

    R    restart on/off
    W    wireframe (needs Feature.WIREFRAME)
    ESC  quit
"""

import numpy as np

import bazalt as bz

logger = bz.Logger()
logger.on_message(lambda msg: print(f"[{msg.severity}] {msg.text}"))

window = bz.Window(900, 600, "Bazalt Demo - Primitive restart", logger=logger)
ctx = bz.Context(logger, optional=[bz.Feature.WIREFRAME])
renderer = ctx.create_renderer(window)

HAS_WIREFRAME = ctx.supports(bz.Feature.WIREFRAME)
RESTART = 0xFFFFFFFF

# Three blades, in strip order: rows of two vertices, left then right.
ROWS = 6
positions = []
ranges = []
for blade in range(3):
    base_x = -0.5 + blade * 0.5
    start = len(positions)
    for row in range(ROWS):
        t = row / (ROWS - 1)
        y = -0.75 + t * 1.4
        half_width = 0.075 * (1.0 - t)
        positions += [(base_x - half_width, y), (base_x + half_width, y)]
    ranges.append((start, len(positions)))

vbuf = ctx.create_buffer(np.array(positions, dtype=np.float32),
                         bz.BufferType.VERTEX, bz.MemoryUsage.STATIC)


def index_buffer(separate):
    indices = []
    for i, (start, end) in enumerate(ranges):
        if i and separate:
            indices.append(RESTART)
        indices += list(range(start, end))
    data = np.array(indices, dtype=np.uint32)
    return ctx.create_buffer(data, bz.BufferType.INDEX, bz.MemoryUsage.STATIC), len(data)


def pipeline(restart, polygon_mode):
    return (ctx.graphics_pipeline()
            .vertex_shader(ctx.compile_shader("blade.vert", bz.ShaderStage.VERTEX))
            .fragment_shader(ctx.compile_shader("blade.frag", bz.ShaderStage.FRAGMENT))
            .vertex_format([bz.VertexFormat.FLOAT2])
            .topology(bz.Topology.TRIANGLE_STRIP, restart=restart)
            .cull_mode(bz.CullMode.NONE)
            .polygon_mode(polygon_mode)
            .build(renderer))


# Whether the pipeline honours the sentinel is pipeline state, so switching it at
# run time IS switching pipeline. The index buffer changes with it, because a
# 0xFFFFFFFF left in a buffer nothing honours would be read as a vertex index.
BUFFERS = {True: index_buffer(True), False: index_buffer(False)}
PIPELINES = {(r, w): pipeline(r, bz.PolygonMode.LINE if w else bz.PolygonMode.FILL)
             for r in (True, False)
             for w in ((False, True) if HAS_WIREFRAME else (False,))}

print(__doc__)
if not HAS_WIREFRAME:
    print("This GPU has no fillModeNonSolid, so the W key does nothing.")
print(f"{BUFFERS[True][1]} indices with the two sentinels, {BUFFERS[False][1]} without.")

restart = True
wireframe = False

while window.is_open():
    bz.poll_events()
    if window.is_key_pressed(bz.Key.ESCAPE):
        break
    if window.was_key_pressed(bz.Key.R):
        restart = not restart
        print("restart on: three blades" if restart else "restart off: one ribbon")
    if window.was_key_pressed(bz.Key.W) and HAS_WIREFRAME:
        wireframe = not wireframe

    buffer, count = BUFFERS[restart]

    with ctx.record() as cmd:
        with cmd.rendering(renderer, clear_color=[0.05, 0.07, 0.10, 1.0]):
            cmd.bind_pipeline(PIPELINES[(restart, wireframe and HAS_WIREFRAME)])
            cmd.bind_vertex_buffer(vbuf)
            cmd.bind_index_buffer(buffer)
            # ONE draw either way. The sentinel is the only difference.
            cmd.draw_indexed(count)

    ctx.begin_frame()
    if renderer.acquire():
        renderer.present(cmd)

    window.set_title(
        f"Bazalt Demo - Primitive restart | {'ON: 3 blades' if restart else 'OFF: 1 ribbon'}"
        + (" | wireframe" if wireframe and HAS_WIREFRAME else ""))

cmd = None
renderer = None
window = None
