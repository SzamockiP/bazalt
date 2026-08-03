"""Occlusion queries — asking the GPU how much of something was drawn.

A blue square slides left and right behind a grey wall. `cmd.occlusion_query()`
wraps its draw, and `query.samples` is how many of its fragments survived the
depth test. Watch the title bar: the number falls to zero while the square is
hidden and climbs back as it comes out.

The CPU never has to guess what is visible, which is what the query is for —
skipping work behind a wall, fading a lens flare, deciding a level of detail.

**`Feature.PRECISE_OCCLUSION` (0.25) decides what the number MEANS.** With it the
query counts samples, so 4 800 means 4 800 covered samples. Without it the Vulkan
spec promises only that the value is non-zero when something passed, and a driver
is free to answer 1 — or 4 800 — for the same picture. A number that means two
things depending on the machine is not a number, so bazalt makes it a Feature you
can ask about rather than a promise it cannot keep. The title bar says which of
the two you are looking at.

The query is asked for as `optional=`, so a GPU without it still runs this: the
count is then "something or nothing", which is still enough to hide a lens flare.

    SPACE   pause the motion
    ESC     quit
"""

import math
import struct
import time

import bazalt as bz

logger = bz.Logger()
logger.on_message(lambda msg: print(f"[{msg.severity}] {msg.text}"))

window = bz.Window(900, 500, "Bazalt Demo - Occlusion query", logger=logger)
# optional=, not features=: a machine without occlusionQueryPrecise still runs
# the demo, it only gets a coarser answer.
ctx = bz.Context(logger, optional=[bz.Feature.PRECISE_OCCLUSION])
renderer = ctx.create_renderer(window)

PRECISE = ctx.supports(bz.Feature.PRECISE_OCCLUSION)

vert = ctx.compile_shader("quad.vert", bz.ShaderStage.VERTEX)
frag = ctx.compile_shader("quad.frag", bz.ShaderStage.FRAGMENT)

# Depth test on, and that is the whole mechanism: the wall is drawn nearer than
# the square, so the square's fragments fail the test where they overlap and the
# query never counts them.
pipeline = (ctx.graphics_pipeline()
            .vertex_shader(vert)
            .fragment_shader(frag)
            .depth_test(True)
            # Four vertices make one quad on a strip; on the default TRIANGLE_LIST
            # they would make one triangle and drop the fourth.
            .topology(bz.Topology.TRIANGLE_STRIP)
            # A screen-space quad has no inside, and the default cull would drop
            # it: Vulkan's Y points down, so the corner order below is clockwise.
            .cull_mode(bz.CullMode.NONE)
            .push_constant(32, bz.ShaderStage.VERTEX)
            .build(renderer))

SQUARE_HALF = 0.18


def quad(centre_x, half_w, half_h, color, depth):
    return struct.pack("2f2f3ff", centre_x, 0.0, half_w, half_h, *color, depth)


print(__doc__)
print(f"precise occlusion: {PRECISE}")

paused = False
elapsed = 0.0
last = time.perf_counter()
samples = None
report_timer = 0.0

while window.is_open():
    bz.poll_events()
    if window.is_key_pressed(bz.Key.ESCAPE):
        break
    if window.was_key_pressed(bz.Key.SPACE):
        paused = not paused

    now = time.perf_counter()
    dt = now - last
    last = now
    if not paused:
        elapsed += dt

    x = math.sin(elapsed * 0.9) * 0.75

    with ctx.record() as cmd:
        with cmd.rendering(renderer, clear_color=[0.06, 0.06, 0.09, 1.0]):
            cmd.bind_pipeline(pipeline)
            # The wall first, and nearer (a smaller depth is nearer).
            # Wider than the square, so "fully hidden" lasts long enough to read.
            cmd.push_constants(0, quad(0.0, 0.30, 0.9, (0.55, 0.55, 0.6), 0.2))
            cmd.draw(4)
            # The square, further away, wrapped in the query. Everything drawn
            # inside the block is counted, so it is exactly one object here.
            with cmd.occlusion_query() as query:
                cmd.push_constants(0, quad(x, SQUARE_HALF, SQUARE_HALF, (0.25, 0.6, 0.95), 0.6))
                cmd.draw(4)

    ctx.begin_frame()
    if not renderer.acquire():
        continue
    renderer.present(cmd)

    # None until the submit that recorded it has finished — the same handle rule
    # Timer.ms follows. Reading it every frame gives the previous frame's answer,
    # which is what a visibility test wants anyway.
    if query.samples is not None:
        samples = query.samples

    report_timer += dt
    if report_timer >= 0.15:
        report_timer = 0.0
        kind = "samples" if PRECISE else "non-zero means visible"
        hidden = "  HIDDEN" if samples == 0 else ""
        window.set_title(f"Bazalt Demo - Occlusion query | {samples} {kind}{hidden}")

cmd = None
renderer = None
window = None
