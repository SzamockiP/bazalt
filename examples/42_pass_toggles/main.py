"""Switching passes off, and taking one out of the graph (0.28).

A post-processing chain is what a prototype fiddles with most: generate an
image, run some effects over it, show it. Before the graph, comparing "with
the effect" against "without it" meant re-recording the frame around an `if`.

A pass is a handle, so there are two better verbs, and this example is built
to show them: the graph is built ONCE, before the loop, and the frame loop
never touches it except to flip a boolean.

  * `p.enabled = False` skips a pass. The graph computes the barriers again
    for the passes that DO run, so switching one off never leaves a stale
    barrier behind. Nothing is re-recorded — the pass keeps its commands, and
    `p.enabled = True` brings it back.
  * `graph.remove(p)` takes a pass out for good. The handle is dead
    afterwards: every verb on it raises StateError.

The animation is a DYNAMIC uniform buffer rather than a push constant, which
is what lets the graph be built once. `buf.update()` writes the slot the next
frame reads, so an animated frame needs no re-recording.

The effects read and write the same image, one texel per invocation, so each
pass is safe on its own. What is NOT safe on its own is one pass reading what
the pass before it wrote — and that is the graph's job. Toggle an effect and
the chain changes shape; the barriers follow, and the validation layers stay
quiet either way.

Keys:
  1   tint on/off
  2   vignette on/off
  D   remove the vignette pass from the graph (permanent)
  ESC quit
"""

import time

import bazalt as bz

W, H = 640, 640

logger = bz.Logger()


@logger.on_message
def on_message(msg):
    print(f"[{msg.severity}] {msg.text}")


window = bz.Window(W, H, "Bazalt Demo - Pass Toggles", logger=logger)
# gpu_timing=True adds a timestamp pair around every windowed submit, so the
# title can report what the frame costs on the GPU. Switching a pass off is
# supposed to make the frame cheaper, and this is where you see it.
ctx = bz.Context(logger, gpu_timing=True)
renderer = ctx.create_renderer(window)

pattern = (ctx.compute_pipeline()
           .shader(ctx.compile_shader("pattern.comp", bz.ShaderStage.COMPUTE))
           .storage_image(0)
           .uniform_buffer(1)
           .build())


def effect(path):
    return (ctx.compute_pipeline()
            .shader(ctx.compile_shader(path, bz.ShaderStage.COMPUTE))
            .storage_image(0)
            .build())


tint = effect("tint.comp")
vignette = effect("vignette.comp")

present = (ctx.graphics_pipeline()
           .vertex_shader(ctx.compile_shader("fullscreen.vert", bz.ShaderStage.VERTEX))
           .fragment_shader(ctx.compile_shader("present.frag", bz.ShaderStage.FRAGMENT))
           .texture(0, bz.ShaderStage.FRAGMENT)
           .build(renderer))

# One image for the whole chain. Every effect reads it and writes it back, so
# what orders the passes is what they touch rather than what they are.
image = ctx.create_image(W, H, bz.Format.RGBA8)
frame_buf = ctx.create_buffer(16, bz.BufferUsage.UNIFORM, bz.MemoryUsage.DYNAMIC)

pool = ctx.create_descriptor_pool()
# allocate_frame_set, not allocate_set: a DYNAMIC buffer has one copy per
# frame in flight, and this is the set that follows the ring with it.
pattern_set = pool.allocate_frame_set(pattern)
pattern_set.set_storage_image(0, image)
pattern_set.set_buffer(1, frame_buf)
tint_set = pool.allocate_set(tint)
tint_set.set_storage_image(0, image)
vignette_set = pool.allocate_set(vignette)
vignette_set.set_storage_image(0, image)
present_set = pool.allocate_set(present)
present_set.set_image(0, image)

GROUPS = ((W + 7) // 8, (H + 7) // 8)

# ── The graph, built once ────────────────────────────────────────────────
graph = ctx.graph()

(graph.add_pass(name="pattern")
    .bind_pipeline(pattern)
    .bind_descriptor_set(pattern_set, pattern)
    .dispatch(*GROUPS))

tint_pass = graph.add_pass(name="tint")
tint_pass.bind_pipeline(tint).bind_descriptor_set(tint_set, tint).dispatch(*GROUPS)

vignette_pass = graph.add_pass(name="vignette")
(vignette_pass.bind_pipeline(vignette)
    .bind_descriptor_set(vignette_set, vignette)
    .dispatch(*GROUPS))

with graph.add_pass(renderer, name="present") as p:
    p.bind_pipeline(present).bind_descriptor_set(present_set, present).draw(3)


fps = 0
gpu_ms = None
gpu_timing_ok = True


def title():
    parts = []
    if tint_pass.enabled:
        parts.append("tint")
    if vignette_pass is None:
        parts.append("vignette removed")
    elif vignette_pass.enabled:
        parts.append("vignette")
    effects = ", ".join(parts) if parts else "no effects"
    # Two numbers, because they answer different questions. The FPS is the whole
    # loop, presentation included, and at this size it is mostly that. The GPU
    # figure is the frame's own work, which is what a pass toggle changes.
    speed = f"{fps} FPS" if fps else "measuring"
    if gpu_ms is not None:
        speed += f" | GPU {gpu_ms:.3f} ms"
    return f"Bazalt Demo - Pass Toggles | {effects} | {speed}"


start = time.time()
fps_timer = start
frames = 0
window.set_title(title())
while window.is_open():
    bz.poll_events()
    if window.is_key_pressed(bz.Key.ESCAPE):
        break

    changed = False
    if window.was_key_pressed(bz.Key.D1):
        # One assignment. The graph marks itself dirty and compiles again on
        # the next submit — the pass itself is untouched.
        tint_pass.enabled = not tint_pass.enabled
        changed = True
    if vignette_pass is not None and window.was_key_pressed(bz.Key.D2):
        vignette_pass.enabled = not vignette_pass.enabled
        changed = True
    if vignette_pass is not None and window.was_key_pressed(bz.Key.D):
        graph.remove(vignette_pass)
        try:
            vignette_pass.dispatch(*GROUPS)
        except bz.StateError as e:
            # The point of remove(): the handle stops working, so a removal
            # cannot be mistaken for a pass that quietly does nothing.
            print(f"the removed pass refuses commands: {e}")
        vignette_pass = None
        changed = True
    if changed:
        window.set_title(title())

    ctx.begin_frame()
    if not renderer.acquire():
        continue

    # The only per-frame work: the animation time. No re-recording.
    frame_buf.update([time.time() - start, 0.0, 0.0, 0.0])
    renderer.present(graph)

    # gpu_time_ms is the frame submitted frames_in_flight ago, so it is None
    # until the ring has cycled once — read it every frame and use it when it
    # arrives.
    if gpu_timing_ok:
        try:
            measured = renderer.gpu_time_ms
        except bz.UnsupportedError:
            # Some drivers advertise timestamps and cannot use them. The demo
            # keeps running; it just stops claiming a number it has not got.
            gpu_timing_ok = False
        else:
            if measured is not None:
                gpu_ms = measured
    frames += 1
    now = time.time()
    if now - fps_timer >= 1.0:
        fps = frames
        frames = 0
        fps_timer = now
        window.set_title(title())
