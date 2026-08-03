"""Three topologies drawing the same blade of grass.

The same vertices, three times. What differs is the INDEX list and the topology
that reads it, and the point is that all three produce the identical picture:

    left    Topology.TRIANGLE_LIST    every triangle spelled out, 3 indices each
    middle  Topology.TRIANGLE_FAN     every triangle shares the first vertex
    right   Topology.TRIANGLE_STRIP   every new vertex extends the last triangle

Press W for the wireframe and the difference shows up at once. The list and the
strip cut the blade into the same zigzag of quads; the fan cuts it into slices
that all meet at the root corner. Same outline, same pixels, three ways of saying
which vertices belong to which triangle — and the index counts printed at startup
are what that costs.

A fan works here because a straight blade is convex: fanning from one corner is a
valid triangulation of it. Bend the blade and it stops being one, which is why
this example does not animate — a fan and a strip would then cover different
areas and the comparison would be a lie.

**A fan is the one topology that is not universal.** Metal has none, so MoltenVK
reports `Feature.TRIANGLE_FANS` missing and the pipeline build says so rather
than drawing something else. This example falls back to a LIST for the middle
blade there, which is the same shape and the portable answer.

    W    wireframe (needs Feature.WIREFRAME)
    ESC  quit
"""

import numpy as np

import bazalt as bz

logger = bz.Logger()
logger.on_message(lambda msg: print(f"[{msg.severity}] {msg.text}"))

window = bz.Window(900, 600, "Bazalt Demo - Topologies", logger=logger)
ctx = bz.Context(logger, optional=[bz.Feature.WIREFRAME])
renderer = ctx.create_renderer(window)

HAS_WIREFRAME = ctx.supports(bz.Feature.WIREFRAME)
HAS_FANS = ctx.supports(bz.Feature.TRIANGLE_FANS)

# ── one blade, three times ────────────────────────────────────────────────
# A blade is ROWS rows of two vertices, left and right, tapering LINEARLY to a
# point. Linear matters: it keeps the outline convex, and a fan is only a valid
# triangulation of a convex shape.

ROWS = 6
QUADS = ROWS - 1

positions = []
for blade in range(3):
    base_x = -0.5 + blade * 0.5
    for row in range(ROWS):
        t = row / (ROWS - 1)
        y = -0.75 + t * 1.4
        half_width = 0.075 * (1.0 - t)
        positions += [(base_x - half_width, y), (base_x + half_width, y)]

vbuf = ctx.create_buffer(np.array(positions, dtype=np.float32),
                         bz.BufferType.VERTEX, bz.MemoryUsage.STATIC)


# Local vertex numbering inside one blade: row r is 2r (left) and 2r+1 (right).

def strip_indices():
    """The vertices in the order they were written. A strip needs nothing else:
    each new vertex closes a triangle with the previous two."""
    return list(range(2 * ROWS))


def list_indices():
    """Every triangle spelled out. Three times the indices of a strip, and the
    only one of the three whose triangles can be reordered or dropped."""
    out = []
    for quad in range(QUADS):
        a, b, c, d = 2 * quad, 2 * quad + 1, 2 * quad + 2, 2 * quad + 3
        out += [a, b, c, c, b, d]
    return out


def fan_indices():
    """The outline, walked from the root-left corner: up the left edge, across
    the tip, back down the right edge. A fan then joins that first corner to
    every consecutive pair, which fills the blade exactly once."""
    left = [2 * row for row in range(ROWS)]
    right = [2 * row + 1 for row in reversed(range(ROWS))]
    return left + right


def indexed(local, blade):
    """The same local pattern, shifted onto blade `blade`'s vertices."""
    offset = blade * 2 * ROWS
    data = np.array([i + offset for i in local], dtype=np.uint32)
    return ctx.create_buffer(data, bz.BufferType.INDEX, bz.MemoryUsage.STATIC), len(data)


def pipeline(topology, polygon_mode):
    return (ctx.graphics_pipeline()
            .vertex_shader(ctx.compile_shader("blade.vert", bz.ShaderStage.VERTEX))
            .fragment_shader(ctx.compile_shader("blade.frag", bz.ShaderStage.FRAGMENT))
            .vertex_format([bz.VertexFormat.FLOAT2])
            .topology(topology)
            # The three index orders do not agree on a winding, and none of them
            # has to: a flat blade has no back to hide.
            .cull_mode(bz.CullMode.NONE)
            .polygon_mode(polygon_mode)
            .build(renderer))


# A topology is baked into the pipeline, so three topologies are three pipelines
# and three draws. That is the honest cost of this comparison, and it is also why
# drawing many shapes in ONE draw needs a single topology plus a restart index
# instead — see examples/40_primitive_restart.
BLADES = [
    ("left", bz.Topology.TRIANGLE_LIST, list_indices(), 0),
    ("middle", bz.Topology.TRIANGLE_FAN if HAS_FANS else bz.Topology.TRIANGLE_LIST,
     fan_indices() if HAS_FANS else list_indices(), 1),
    ("right", bz.Topology.TRIANGLE_STRIP, strip_indices(), 2),
]

draws = []
for name, topology, local, blade in BLADES:
    buffer, count = indexed(local, blade)
    draws.append((name, topology, buffer, count,
                  pipeline(topology, bz.PolygonMode.FILL),
                  pipeline(topology, bz.PolygonMode.LINE) if HAS_WIREFRAME else None))

print(__doc__)
if not HAS_WIREFRAME:
    print("This GPU has no fillModeNonSolid, so the W key does nothing.")
if not HAS_FANS:
    print("This driver reports no triangleFans (Metal has none), so the middle blade is a list.")
for name, topology, _, count, _, _ in draws:
    print(f"  {name:6} {topology.name:16} {count:2} indices for {QUADS * 2} triangles")

wireframe = False

while window.is_open():
    bz.poll_events()
    if window.is_key_pressed(bz.Key.ESCAPE):
        break
    if window.was_key_pressed(bz.Key.W) and HAS_WIREFRAME:
        wireframe = not wireframe

    with ctx.record() as cmd:
        with cmd.rendering(renderer, clear_color=[0.05, 0.07, 0.10, 1.0]):
            for _, _, buffer, count, filled, lined in draws:
                cmd.bind_pipeline(lined if wireframe and lined else filled)
                cmd.bind_vertex_buffer(vbuf)
                cmd.bind_index_buffer(buffer)
                cmd.draw_indexed(count)

    ctx.begin_frame()
    if renderer.acquire():
        renderer.present(cmd)

    window.set_title("Bazalt Demo - Topologies | list, fan, strip"
                     + (" | wireframe" if wireframe and HAS_WIREFRAME else ""))

cmd = None
renderer = None
window = None
