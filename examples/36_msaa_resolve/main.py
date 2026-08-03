"""Custom MSAA resolve — reading the samples one at a time.

MSAA renders N samples per pixel and the hardware averages them into one colour.
That average is almost always what you want, and it is also the end of the road:
once it has happened, the samples are gone. Per-sample shading, per-sample edge
detection and the history reject in a TAA resolve all need them before that.

`keep_samples=True` on the target stores the multisampled attachment instead of
discarding it, and `target.multisampled_color[0]` is then an ordinary Image that
goes into a `sampler2DMS`. It costs the bandwidth of a full multisample buffer,
which is why it is off by default: without it the samples never have to leave
tile memory on a tiled GPU.

SPACE cycles what you are looking at:

  0. the hardware resolve      — smooth edges, one colour per pixel
  1. sample 0 on its own       — the aliased picture, from inside the same buffer
  2. how much the samples disagree — a map of every pixel MSAA is working on

ESC quits.
"""

import math
import struct
import time

import numpy as np

import bazalt as bz

logger = bz.Logger()
logger.on_message(lambda msg: print(f"[{msg.severity}] {msg.text}"))

window = bz.Window(1000, 700, "Bazalt Demo - Custom MSAA resolve", logger=logger)
ctx = bz.Context(logger)
renderer = ctx.create_renderer(window)

SAMPLES = min(4, ctx.max_samples())
if SAMPLES < 2:
    raise SystemExit("this GPU reports no MSAA support (max_samples == 1)")

# The offscreen target is what carries the samples. keep_samples is the whole
# feature: without it multisampled_color is empty, because the pass throws the
# samples away and there would be nothing to hand out.
scene_target = ctx.create_render_target(
    window.width, window.height, color=bz.Format.RGBA8,
    samples=SAMPLES, keep_samples=True, name="star")

# ── the star ──────────────────────────────────────────────────────────────
# Thin triangles meeting at the centre: the shape that aliases worst.

POINTS = 9
vertices = []
for i in range(POINTS):
    a0 = (i / POINTS) * math.tau
    a1 = ((i + 0.12) / POINTS) * math.tau
    vertices += [0.0, 0.0,
                 math.cos(a0) * 0.9, math.sin(a0) * 0.9,
                 math.cos(a1) * 0.9, math.sin(a1) * 0.9]
star_buffer = ctx.create_buffer(np.array(vertices, dtype=np.float32),
                                bz.BufferType.VERTEX, bz.MemoryUsage.STATIC)

star = (ctx.graphics_pipeline()
        .vertex_shader(ctx.compile_shader("star.vert", bz.ShaderStage.VERTEX))
        .fragment_shader(ctx.compile_shader("star.frag", bz.ShaderStage.FRAGMENT))
        .vertex_format([bz.VertexFormat.FLOAT2])
        .push_constant(8, bz.ShaderStage.VERTEX)
        # A star fan winds both ways depending on the point.
        .cull_mode(bz.CullMode.NONE)
        .build(scene_target))

# ── the resolve pass ──────────────────────────────────────────────────────
# Two bindings of the same picture: the single-sample resolve and the
# multisampled attachment. The descriptor TYPE is identical — a sampler2DMS is a
# combined image sampler like any other — so nothing about the layout says which
# is which. The shader's declaration does.

resolve = (ctx.graphics_pipeline()
           .vertex_shader(ctx.compile_shader("fullscreen.vert", bz.ShaderStage.VERTEX))
           .fragment_shader(ctx.compile_shader("resolve.frag", bz.ShaderStage.FRAGMENT))
           .texture(0, bz.ShaderStage.FRAGMENT)
           .texture(1, bz.ShaderStage.FRAGMENT)
           .push_constant(8, bz.ShaderStage.FRAGMENT)
           .build(renderer))

pool = ctx.create_descriptor_pool()
dset = pool.allocate_set(resolve)
# NEAREST on the multisampled image because texelFetch does no filtering at all;
# the sampler is bound because the descriptor type says so, not because it works.
nearest = ctx.create_sampler(filter=bz.Filter.NEAREST)
dset.set_image(0, scene_target.color[0], sampler=ctx.create_sampler(filter=bz.Filter.LINEAR))
dset.set_image(1, scene_target.multisampled_color[0], sampler=nearest)

MODES = ["hardware resolve", "sample 0 only", "sample disagreement"]
mode = 0
start = time.perf_counter()

print(f"{SAMPLES}x MSAA. SPACE cycles the view, ESC quits.")

while window.is_open():
    bz.poll_events()
    if window.is_key_pressed(bz.Key.ESCAPE):
        break
    if window.was_key_pressed(bz.Key.SPACE):
        mode = (mode + 1) % len(MODES)
        print(f"  {mode}: {MODES[mode]}")

    angle = (time.perf_counter() - start) * 0.4
    window.set_title(f"Bazalt Demo - Custom MSAA resolve | {MODES[mode]}")

    # The short forms of push_constants and bind_descriptor_set: the pipeline is
    # bound two lines up, and the set knows which index it was allocated for, so
    # naming either again is a chance to disagree with the truth rather than
    # information.
    with ctx.record() as cmd:
        # Pass one: the star, multisampled, resolved AND kept.
        with cmd.rendering(scene_target, clear_color=[0.02, 0.02, 0.04, 1.0]):
            cmd.bind_pipeline(star)
            cmd.push_constants(0, struct.pack("2f", angle, scene_target.width / scene_target.height))
            cmd.bind_vertex_buffer(star_buffer)
            cmd.draw(POINTS * 3)
        # Pass two: look at it three ways. No barrier here — the target contract
        # leaves both images readable, and the tracker knows it.
        with cmd.rendering(renderer):
            cmd.bind_pipeline(resolve)
            cmd.bind_descriptor_set(dset)
            cmd.push_constants(0, struct.pack("2i", mode, SAMPLES))
            cmd.draw(3)

    ctx.begin_frame()
    if renderer.acquire():
        renderer.present(cmd)
