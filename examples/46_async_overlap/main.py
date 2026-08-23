"""Async compute: work the frame does NOT consume, and a switch to try it (0.29).

A million points drift through a swirl. The simulation is a compute pass and the
drawing is a render pass, and the two are INDEPENDENT within a frame: the
dispatch writes the buffer the draw is not reading. Press SPACE to move that
dispatch between `Queue.COMPUTE` and `Queue.GRAPHICS` while it runs, and watch
the numbers in the title.

    graph.add_pass(name="simulate", queue=queue)   # SPACE flips `queue`

## Why this shape and not the obvious one

examples/28_gpu_culling is the obvious one and it is the wrong shape: there the
draw reads what the dispatch wrote IN THE SAME FRAME, so the two cannot overlap
whatever queue they are on — the drawing is waiting for the culling by
definition. Putting that dispatch on `Queue.COMPUTE` makes the frame slower, and
measurably so, because the graph then has to split it into two submits with a
semaphore between them. A second queue is worth trying only when the compute
result is NOT what this frame draws.

So the points are double-buffered. Frame N simulates into buffer A while drawing
buffer B, which frame N-1 wrote; frame N+1 swaps them. The picture is one frame
behind the simulation, which for a million drifting points is invisible, and it
is what makes the two passes independent.

It is also why this uses ONE graph, rebuilt each frame, rather than two. Bazalt
orders a graph against its own previous submit on the other queue — frame N+1's
dispatch waits for frame N's draw to stop reading the buffer it is about to
overwrite. Between two DIFFERENT graphs there is no such order, by design, so
splitting this into a "sim graph" and a "draw graph" would be a race.

## What the numbers said on the machine this was written on

Less than the folklore promises, and the shape of the answer is the lesson.
Measured headless at 1,000,000 points into 1100x1100, in milliseconds per frame:

    octaves   simulate   draw    sum     on GRAPHICS   on COMPUTE
      100       0.49     0.58    1.07       1.07          0.99
      400       1.87     0.55    2.42       2.42          2.39
     1600       7.40     0.56    7.96       7.98          7.94

The frame costs the SUM of the two in both modes, and a perfect overlap would
have cost the larger of them. The second queue wins about 7% where the two
halves are the same size and nothing at all where one of them dominates.

The reason is not the queues, it is the silicon. Both halves here are shader
work, and either one alone already saturates the machine, so running them at
the same time cannot make the total smaller — the work is the work. Overlap
pays when one side leaves units IDLE that the other can fill: drawing bound on
the rasterizer or on memory bandwidth beside compute bound on the ALUs, a depth
prepass, a shadow map. A dispatch and a draw that both want every ALU are not
that.

So this is a switch to MEASURE with, not a switch that makes things faster.
Press +/- to move the balance between the two halves and watch what it does to
your own numbers: the gap opens where they are equal, which is the only place a
second queue has anything to give.

What a second queue reliably does give is independence from the graphics
queue's ORDERING — work that must not be delayed by the frame and must not
delay it. That is a scheduling property, and it is why the queue is the
caller's choice rather than the library's.

One number to carry away, because it decides whether the question is even worth
asking: another queue means another `vkQueueSubmit`, and a submit costs about
0.07-0.11 ms on the driver this was written against — more than an EMPTY submit
has any right to. So a pass is worth moving only when it is worth more than a
submit. The simulation here is, at any of the octave counts above. The culling
in examples/28_gpu_culling is not: that whole frame's GPU work is smaller than
one submit.

Keys: SPACE switches the queue, +/- change the simulation cost, ESC exits.
"""

import struct
import time

import glm
import numpy as np

import bazalt as bz

N = 1_000_000
WORK = 400          # swirl octaves per point — the simulation's whole cost
GROUPS = (N + 255) // 256

logger = bz.Logger()
logger.on_message(lambda msg: print(f"[{msg.severity}] {msg.text}"))

window = bz.Window(1100, 800, "Bazalt Demo - Async Overlap", logger=logger)
# gpu_timing=True adds a timestamp pair around every windowed submit, so the
# title can report the frame's graphics span beside the wall clock.
ctx = bz.Context(logger, gpu_timing=True)
renderer = ctx.create_renderer(window)

if not ctx.supports(bz.Feature.ASYNC_COMPUTE):
    print("This device has no compute-only queue family, so Queue.COMPUTE runs on the")
    print("graphics queue under its own timeline. The switch still works and the program")
    print("is unchanged — there is simply no second queue for the work to move to.")


def seed_points(rng):
    state = np.empty((N, 4), np.float32)
    state[:, :3] = rng.uniform(-1.0, 1.0, (N, 3))
    state[:, 3] = rng.uniform(0.0, 6.28, N)
    return state


rng = np.random.default_rng(3)
# Two buffers, and the ping-pong between them is what makes the dispatch and the
# draw independent inside one frame.
points_state = [ctx.create_buffer(seed_points(rng), bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
                for _ in range(2)]

simulate = (ctx.compute_pipeline()
            .shader(ctx.compile_shader("simulate.comp", bz.ShaderStage.COMPUTE))
            .storage_buffer(0)   # previous state, read
            .storage_buffer(1)   # next state, written
            .push_constant(12)   # time, count, work
            .build())

draw = (ctx.graphics_pipeline()
        .vertex_shader(ctx.compile_shader("points.vert", bz.ShaderStage.VERTEX))
        .fragment_shader(ctx.compile_shader("points.frag", bz.ShaderStage.FRAGMENT))
        .storage_buffer(0, bz.ShaderStage.VERTEX)
        .push_constant(64, bz.ShaderStage.VERTEX)
        .topology(bz.Topology.POINT_LIST)
        .blend(True, bz.BlendMode.ADDITIVE)
        .build(renderer))

pool = ctx.create_descriptor_pool()
# One pair of sets per parity: which buffer is read and which is written swaps
# every frame, and a descriptor set is cheaper to pick than to rewrite.
sim_sets = []
draw_sets = []
for i in range(2):
    s = pool.allocate_set(simulate)
    s.set_buffer(0, points_state[i])
    s.set_buffer(1, points_state[1 - i])
    sim_sets.append(s)
    d = pool.allocate_set(draw)
    d.set_buffer(0, points_state[i])
    draw_sets.append(d)

graph = ctx.graph()

sim_queue = bz.Queue.COMPUTE
work = WORK
start = time.time()
frames = 0
fps = 0
fps_timer = start
frame_ms = 0.0
gpu_ms = None
gpu_timing_ok = True
parity = 0


def title():
    queue_name = "COMPUTE" if sim_queue == bz.Queue.COMPUTE else "GRAPHICS"
    if sim_queue == bz.Queue.COMPUTE and not ctx.supports(bz.Feature.ASYNC_COMPUTE):
        queue_name += " (aliased)"
    speed = f"{fps} FPS, {frame_ms:.2f} ms" if fps else "measuring"
    if gpu_ms is not None:
        speed += f" | graphics {gpu_ms:.2f} ms"
    return (f"Bazalt Demo - Async Overlap | simulate on {queue_name} | "
            f"{work} octaves | {speed} | SPACE switches")


window.set_title(title())
while window.is_open():
    bz.poll_events()
    if window.is_key_pressed(bz.Key.ESCAPE):
        break

    if window.was_key_pressed(bz.Key.SPACE):
        sim_queue = bz.Queue.GRAPHICS if sim_queue == bz.Queue.COMPUTE else bz.Queue.COMPUTE
        window.set_title(title())
    if window.was_key_pressed(bz.Key.EQUAL) or window.was_key_pressed(bz.Key.KP_ADD):
        work = min(work * 2, 6400)
        window.set_title(title())
    if window.was_key_pressed(bz.Key.MINUS) or window.was_key_pressed(bz.Key.KP_SUBTRACT):
        work = max(work // 2, 8)
        window.set_title(title())
    ctx.begin_frame()
    if not renderer.acquire():
        continue

    now = time.time()
    aspect = window.width / max(window.height, 1)
    proj = glm.perspective(glm.radians(60.0), aspect, 0.1, 10.0)
    proj[1][1] *= -1
    angle = (now - start) * 0.15
    eye = glm.vec3(np.sin(angle) * 3.2, 0.9, np.cos(angle) * 3.2)
    view_proj = bytes(glm.transpose(proj * glm.lookAt(eye, glm.vec3(0), glm.vec3(0, 1, 0))))

    # Rebuilt every frame, which costs about a hundredth of a millisecond and
    # keeps both the parity swap and the queue switch to one place. The graph
    # keeps its own GPU objects across reset(), so this allocates nothing.
    graph.reset()
    (graph.add_pass(name="simulate", queue=sim_queue)
        .bind_pipeline(simulate)
        .bind_descriptor_set(sim_sets[parity], simulate)
        .push_constants(simulate, 0, struct.pack("<fII", now - start, N, work))
        .dispatch(GROUPS))
    # Reads the OTHER buffer — the one last frame's dispatch wrote — so nothing
    # in this frame waits for the dispatch above.
    with graph.add_pass(renderer, clear_color=[0.01, 0.01, 0.03, 1.0], name="draw") as p:
        p.bind_pipeline(draw)
        p.bind_descriptor_set(draw_sets[parity], draw)
        p.push_constants(draw, 0, view_proj)
        p.draw(N)
    renderer.present(graph)
    parity = 1 - parity

    if gpu_timing_ok:
        try:
            measured = renderer.gpu_time_ms
        except bz.UnsupportedError:
            gpu_timing_ok = False
        else:
            if measured is not None:
                gpu_ms = measured

    frames += 1
    if now - fps_timer >= 0.5:
        fps = int(frames / (now - fps_timer))
        frame_ms = (now - fps_timer) * 1000.0 / max(frames, 1)
        frames = 0
        fps_timer = now
        window.set_title(title())
