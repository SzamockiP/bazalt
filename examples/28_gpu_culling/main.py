"""Indirect draw — GPU frustum culling, seen from outside the frustum.

Two windows, one Context, one culled scene:

  * **"culled view"** renders from the camera the culling is done FOR. It turns
    slowly, and the cubes it shows are the survivors.
  * **"observer"** is a free camera you fly with WASD and the right mouse button.
    It draws the same survivor buffer, plus the culling camera's frustum as a
    wireframe box.

The observer is the whole point. From inside the culled camera nothing looks
culled — that is what culling means, and it is measurable: the culled view renders
PIXEL-IDENTICAL with culling on and off, while the observer's pixel count drops
sharply — press P with culling on, then C and P again, and read your own two
numbers rather than mine (they depend on where you have flown the observer, and
the ratio is the claim). Fly out to the side and you can see the scene
has been cut to a wedge: cubes exist inside the yellow frustum and nowhere else,
and the wedge swings around as the culling camera turns.

About 16,000 of the 200,000 survive, cross-checked against the same test run on
the CPU — which is how the frustum-plane bug below was found, because a wrong
plane still produces a plausible-looking number.

**Read the GPU milliseconds in the title, not the FPS.** The two say different
things and only one of them is about culling. With 200,000 cubes on the machine
this was written on the frame's own GPU work is about 0.52 ms with culling off
and about 0.07 ms with it on — seven times less, which is the claim. The FPS
counter beside it measures the whole loop: two windows, two acquires, two
presents and the compositor, which together cost more than either number above.
At 20,000 cubes, where this example started, the entire GPU frame was 0.05 ms
and the FPS was the same with culling on and off — or slightly WORSE with it on,
because recording one more pass costs a little CPU and the GPU had nothing left
to save. Culling pays when the drawing is the expensive part, and that takes a
scene big enough to be worth culling.

Each frame:

  1. `p.fill_buffer(args, 0)` zeroes the draw arguments — the prerequisite that
     landed in 0.18 for exactly this, because a counter an atomic increments has to
     start each frame at a known value.
  2. A compute pass tests every cube against the culling camera's frustum planes,
     atomically increments `instanceCount`, and compacts the survivors.
  3. `p.draw_indexed_indirect(args)` draws whatever that came to — in BOTH
     windows, from one argument buffer.

The cull pass stays on `Queue.GRAPHICS`, and that is measured rather than
assumed: moving it to `Queue.COMPUTE` makes this frame SLOWER. The reason is not
the semaphore between the two halves, which is what it looks like — the same
frame split into two submits on ONE queue costs the same, and a frame whose draw
does not wait for the dispatch at all costs nearly as much. What costs is the
second `vkQueueSubmit`: about 0.07-0.11 ms on this driver, against a frame whose
entire GPU work is 0.07 ms. Another queue means another submit, so a pass is
worth moving only when it is worth more than a submit.

Here it never is, and the dependency settles it anyway: the drawing is waiting
for the culling by definition. A second queue pays for work the frame does NOT
consume — a simulation the next frame draws, or post-processing of the previous
one. See examples/46_async_overlap, which is that shape with a switch on it.

The CPU never learns the count. That is what makes it different from culling on the
host: no readback, and no per-instance buffer to size.

One draw command whose instanceCount the GPU accumulates, rather than N commands.
0.21 added the other route — `draw_indexed_indirect(args, count=N,
count_buffer=...)` with Feature.DRAW_INDIRECT_COUNT — and this example keeps the
first one on purpose: a compacted instance buffer plus one command needs no
feature bit at all, and it is the shape to reach for when every survivor draws the
same mesh. Use a count buffer when the survivors need DIFFERENT commands, e.g. one
per mesh or per LOD.

Keys: WASD + QE move the observer, hold RIGHT MOUSE to look, SPACE pauses the
culling camera, C toggles culling off, P prints the observer's drawn-pixel count.
Close either window to exit.

Press P from outside the frustum with culling on, then C and P again: the two
numbers are the claim above, measured on your machine rather than remembered.
Watch the GPU milliseconds in the title while you press C, for the other half.
"""

import struct
import time

import glm
import numpy as np

import bazalt as bz

logger = bz.Logger()
logger.on_message(lambda msg: print(f"[{msg.severity}] {msg.text}"))

# gpu_timing=True puts a timestamp pair around every windowed submit. Without it
# the only number available is the FPS, and at this size the FPS measures the
# presentation rather than the culling — see the note in the docstring.
ctx = bz.Context(logger, gpu_timing=True)

culled_window = bz.Window(760, 560, "Bazalt Demo - culled view (the frustum)", logger=logger)
culled_window.set_position(60, 90)
observer_window = bz.Window(760, 560, "Bazalt Demo - observer (fly with WASD)", logger=logger)
observer_window.set_position(860, 90)

culled_renderer = ctx.create_renderer(culled_window)
observer_renderer = ctx.create_renderer(observer_window)

# Ten times what this example started with, and the reason is the whole point of
# it: at 20,000 the drawing costs 0.05 ms and there is nothing for culling to
# save. Culling is a way to not draw things, so the scene has to be big enough
# that drawing it is the expensive part.
COUNT = 200000
INDEX_COUNT = 36

# ── the candidates ────────────────────────────────────────────────────────
# xyz = centre, w = radius. A slab wide enough that the culling camera never sees
# more than a fraction of it, which is what makes the wedge obvious from outside.
rng = np.random.default_rng(7)
spheres = np.empty((COUNT, 4), dtype=np.float32)
spheres[:, 0] = rng.uniform(-60.0, 60.0, COUNT)
spheres[:, 1] = rng.uniform(-6.0, 6.0, COUNT)
spheres[:, 2] = rng.uniform(-60.0, 60.0, COUNT)
spheres[:, 3] = rng.uniform(0.5, 1.1, COUNT)

candidates = ctx.create_buffer(spheres, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
# Device-local. NOT MemoryUsage.DYNAMIC: that is host-visible memory allocated for
# sequential CPU writes, and a compute shader writing into it does not come back.
visible = ctx.create_buffer(COUNT * 16, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
args = ctx.create_buffer(20, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)

# Culling off: everything is "visible", so the draw runs over all of them.
all_visible = ctx.create_buffer(spheres, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
no_cull_args = ctx.create_buffer(
    np.array([INDEX_COUNT, COUNT, 0, 0, 0], dtype=np.uint32),
    bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)

# ── a unit cube ───────────────────────────────────────────────────────────
vertices = np.array([
    -0.5, -0.5,  0.5,   0.0, 0.0, 1.0,
     0.5, -0.5,  0.5,   0.0, 0.0, 1.0,
     0.5,  0.5,  0.5,   0.0, 0.0, 1.0,
    -0.5,  0.5,  0.5,   0.0, 0.0, 1.0,
    -0.5, -0.5, -0.5,   0.0, 0.0, -1.0,
     0.5, -0.5, -0.5,   0.0, 0.0, -1.0,
     0.5,  0.5, -0.5,   0.0, 0.0, -1.0,
    -0.5,  0.5, -0.5,   0.0, 0.0, -1.0,
    -0.5, -0.5, -0.5,  -1.0, 0.0, 0.0,
    -0.5, -0.5,  0.5,  -1.0, 0.0, 0.0,
    -0.5,  0.5,  0.5,  -1.0, 0.0, 0.0,
    -0.5,  0.5, -0.5,  -1.0, 0.0, 0.0,
     0.5, -0.5, -0.5,   1.0, 0.0, 0.0,
     0.5, -0.5,  0.5,   1.0, 0.0, 0.0,
     0.5,  0.5,  0.5,   1.0, 0.0, 0.0,
     0.5,  0.5, -0.5,   1.0, 0.0, 0.0,
    -0.5, -0.5, -0.5,   0.0, -1.0, 0.0,
     0.5, -0.5, -0.5,   0.0, -1.0, 0.0,
     0.5, -0.5,  0.5,   0.0, -1.0, 0.0,
    -0.5, -0.5,  0.5,   0.0, -1.0, 0.0,
    -0.5,  0.5, -0.5,   0.0, 1.0, 0.0,
     0.5,  0.5, -0.5,   0.0, 1.0, 0.0,
     0.5,  0.5,  0.5,   0.0, 1.0, 0.0,
    -0.5,  0.5,  0.5,   0.0, 1.0, 0.0,
], dtype=np.float32)
indices = np.array([
    0, 1, 2, 2, 3, 0,
    5, 4, 7, 7, 6, 5,
    8, 9, 10, 10, 11, 8,
    13, 12, 15, 15, 14, 13,
    16, 17, 18, 18, 19, 16,
    23, 22, 21, 21, 20, 23,
], dtype=np.uint32)
vbuf = ctx.create_buffer(vertices, bz.BufferType.VERTEX, bz.MemoryUsage.STATIC)
ibuf = ctx.create_buffer(indices, bz.BufferType.INDEX, bz.MemoryUsage.STATIC)

# The 12 edges of a frustum, as pairs of the 8 clip-space corners. Vulkan's depth
# range is 0..1, so the near plane is z=0 and the far plane z=1.
FRUSTUM_CORNERS = np.array([
    [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
], dtype=np.float32)
FRUSTUM_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
frustum_lines = ctx.create_buffer(len(FRUSTUM_EDGES) * 2 * 3 * 4,
                                  bz.BufferType.VERTEX, bz.MemoryUsage.DYNAMIC)

# ── pipelines ─────────────────────────────────────────────────────────────
comp = ctx.compile_shader("cull.comp", bz.ShaderStage.COMPUTE)
cull = (ctx.compute_pipeline()
        .shader(comp)
        .storage_buffer(0)   # draw arguments
        .storage_buffer(1)   # candidates
        .storage_buffer(2)   # compacted survivors
        .push_constant(72)          # mat4 + 2 uints
        .build())

vert = ctx.compile_shader("cube.vert", bz.ShaderStage.VERTEX)
frag = ctx.compile_shader("cube.frag", bz.ShaderStage.FRAGMENT)
line_vert = ctx.compile_shader("line.vert", bz.ShaderStage.VERTEX)
line_frag = ctx.compile_shader("line.frag", bz.ShaderStage.FRAGMENT)


def cube_pipeline(renderer):
    return (ctx.graphics_pipeline()
            .vertex_shader(vert)
            .fragment_shader(frag)
            .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
            .storage_buffer(0, bz.ShaderStage.VERTEX)
            .push_constant(64, bz.ShaderStage.VERTEX)
            .depth_test(True)
            .cull_mode(bz.CullMode.BACK, bz.FrontFace.COUNTER_CLOCKWISE)
            .build(renderer))


def line_pipeline(renderer):
    return (ctx.graphics_pipeline()
            .vertex_shader(line_vert)
            .fragment_shader(line_frag)
            .vertex_format([bz.VertexFormat.FLOAT3])
            .topology(bz.Topology.LINE_LIST)
            .push_constant(64, bz.ShaderStage.VERTEX)
            .depth_test(True)
            .build(renderer))


culled_pipeline = cube_pipeline(culled_renderer)
observer_pipeline = cube_pipeline(observer_renderer)
observer_lines = line_pipeline(observer_renderer)

pool = ctx.create_descriptor_pool()
cull_set = pool.allocate_set(cull)
cull_set.set_buffer(0, args)
cull_set.set_buffer(1, candidates)
cull_set.set_buffer(2, visible)
# The two cube pipelines are built from the same declarators, so they share a
# descriptor set layout and one set per SOURCE buffer serves both windows.
draw_set = pool.allocate_set(culled_pipeline)
draw_set.set_buffer(0, visible)
no_cull_set = pool.allocate_set(culled_pipeline)
no_cull_set.set_buffer(0, all_visible)

# Each window needs its own Graph: one holds a single command buffer per frame
# slot, so two windows replaying one would overwrite work still in flight.
culled_graph = ctx.graph()
observer_graph = ctx.graph()

culling = True
paused = False

# The culling camera: at the origin, turning.
cull_angle = 0.0
CULL_FOV = 55.0
CULL_NEAR, CULL_FAR = 0.5, 45.0

# The observer: starts up and behind, looking down at the origin.
obs_pos = glm.vec3(0.0, 34.0, 42.0)
obs_yaw = -90.0
obs_pitch = -35.0
looking = False

start = time.time()
last = start
frames = 0
fps_timer = start
gpu_ms = None
gpu_timing_ok = True


def culling_view_proj(aspect):
    proj = glm.perspectiveRH_ZO(glm.radians(CULL_FOV), aspect, CULL_NEAR, CULL_FAR)
    proj[1][1] *= -1
    eye = glm.vec3(0.0, 0.0, 0.0)
    look = glm.vec3(glm.cos(cull_angle), 0.0, glm.sin(cull_angle))
    return proj * glm.lookAt(eye, eye + look, glm.vec3(0, 1, 0))


def observer_view_proj(aspect):
    proj = glm.perspectiveRH_ZO(glm.radians(60.0), aspect, 0.1, 400.0)
    proj[1][1] *= -1
    forward = observer_forward()
    return proj * glm.lookAt(obs_pos, obs_pos + forward, glm.vec3(0, 1, 0))


def observer_forward():
    cy, sy = glm.cos(glm.radians(obs_yaw)), glm.sin(glm.radians(obs_yaw))
    cp, sp = glm.cos(glm.radians(obs_pitch)), glm.sin(glm.radians(obs_pitch))
    return glm.normalize(glm.vec3(cy * cp, sp, sy * cp))


def upload_frustum(view_proj):
    """The culling frustum in world space, for the observer to look at.

    Its 8 corners are the clip-space cube corners run back through the inverse of
    the same matrix the compute shader pulled its planes out of — so what is drawn
    cannot drift from what is culled.
    """
    inverse = glm.inverse(view_proj)
    world = []
    for corner in FRUSTUM_CORNERS:
        p = inverse * glm.vec4(float(corner[0]), float(corner[1]), float(corner[2]), 1.0)
        world.append(glm.vec3(p) / p.w)
    data = []
    for a, b in FRUSTUM_EDGES:
        data += [world[a].x, world[a].y, world[a].z]
        data += [world[b].x, world[b].y, world[b].z]
    frustum_lines.update(np.array(data, dtype=np.float32).tobytes())


while culled_window.is_open() and observer_window.is_open():
    bz.poll_events()

    if culled_window.was_key_pressed(bz.Key.SPACE) or observer_window.was_key_pressed(bz.Key.SPACE):
        paused = not paused
    if culled_window.was_key_pressed(bz.Key.C) or observer_window.was_key_pressed(bz.Key.C):
        culling = not culling
    # P measures instead of asking you to look. The docstring's numbers used to be
    # a measurement somebody took once and wrote down; this is the same
    # measurement, on your machine, from the frame you are looking at.
    measure = (culled_window.was_key_pressed(bz.Key.P)
               or observer_window.was_key_pressed(bz.Key.P))

    now = time.time()
    dt = now - last
    last = now
    if not paused:
        cull_angle += dt * 0.35

    # ── the observer's free camera ────────────────────────────────────────
    if observer_window.is_mouse_button_pressed(bz.MouseButton.RIGHT):
        if not looking:
            observer_window.set_cursor_mode(bz.CursorMode.DISABLED)
            looking = True
        # CursorMode.DISABLED already gives unbounded virtual motion and does its own
        # recentring, so there is nothing to warp here. Calling set_cursor_position
        # every frame on top of it is the HIDDEN-cursor pattern, and mixing the two
        # cancels the look entirely: bazalt re-arms its first-event suppression on a
        # warp (so a warp is never mistaken for movement), so warping every frame
        # suppresses every frame's delta.
        mouse = observer_window.get_mouse_state()
        obs_yaw += mouse.dx * 0.15
        obs_pitch = max(min(obs_pitch - mouse.dy * 0.15, 89.0), -89.0)
    elif looking:
        observer_window.set_cursor_mode(bz.CursorMode.NORMAL)
        looking = False

    speed = 30.0 * dt
    forward = observer_forward()
    right = glm.normalize(glm.cross(forward, glm.vec3(0, 1, 0)))
    if observer_window.is_key_pressed(bz.Key.W):
        obs_pos += forward * speed
    if observer_window.is_key_pressed(bz.Key.S):
        obs_pos -= forward * speed
    if observer_window.is_key_pressed(bz.Key.A):
        obs_pos -= right * speed
    if observer_window.is_key_pressed(bz.Key.D):
        obs_pos += right * speed
    if observer_window.is_key_pressed(bz.Key.E):
        obs_pos += glm.vec3(0, 1, 0) * speed
    if observer_window.is_key_pressed(bz.Key.Q):
        obs_pos -= glm.vec3(0, 1, 0) * speed

    ctx.begin_frame()

    cull_vp = culling_view_proj(culled_window.width / max(culled_window.height, 1))
    cull_vp_bytes = bytes(glm.transpose(cull_vp))
    upload_frustum(cull_vp)

    source_set = draw_set if culling else no_cull_set
    source_args = args if culling else no_cull_args

    # ── window 1: the camera the culling is done for ──────────────────────
    if culled_renderer.acquire():
        culled_graph.reset()
        if culling:
            cull_pass = culled_graph.add_pass(name="cull")
            cull_pass.fill_buffer(args, 0)
            cull_pass.bind_pipeline(cull)
            cull_pass.bind_descriptor_set(cull_set, cull)
            cull_pass.push_constants(cull, 0, cull_vp_bytes + struct.pack("II", COUNT, INDEX_COUNT))
            cull_pass.dispatch((COUNT + 63) // 64)
        with culled_graph.add_pass(culled_renderer, clear_color=[0.03, 0.04, 0.07, 1.0],
                                   name="culled view") as p:
            p.bind_pipeline(culled_pipeline)
            p.bind_descriptor_set(source_set, culled_pipeline)
            p.push_constants(culled_pipeline, 0, cull_vp_bytes)
            p.bind_vertex_buffer(vbuf).bind_index_buffer(ibuf)
            p.draw_indexed_indirect(source_args)
        culled_renderer.present(culled_graph)

    # ── window 2: the observer, outside the frustum ───────────────────────
    if observer_renderer.acquire():
        obs_vp = bytes(glm.transpose(
            observer_view_proj(observer_window.width / max(observer_window.height, 1))))
        observer_graph.reset()
        # The compute pass that fills `args` and `visible` runs in the OTHER
        # window's graph, and this one only reads them. Until 0.24 that needed
        # two manual barriers here, because the tracker only knew about one
        # recording and this one writes nothing it can see. It is automatic now:
        # the first READ of a buffer in a graph waits for whatever wrote it last,
        # wherever that was. p.barrier() is still there for the cases the
        # tracker cannot reach.
        with observer_graph.add_pass(observer_renderer, clear_color=[0.05, 0.05, 0.09, 1.0],
                                     name="observer") as p:
            p.bind_pipeline(observer_pipeline)
            p.bind_descriptor_set(source_set, observer_pipeline)
            p.push_constants(observer_pipeline, 0, obs_vp)
            p.bind_vertex_buffer(vbuf).bind_index_buffer(ibuf)
            p.draw_indexed_indirect(source_args)
            # The frustum the culling was done with, so the empty space has a shape.
            p.bind_pipeline(observer_lines)
            p.push_constants(observer_lines, 0, obs_vp)
            p.bind_vertex_buffer(frustum_lines)
            p.draw(len(FRUSTUM_EDGES) * 2)
        observer_renderer.present(observer_graph, capture=measure)
        if measure:
            # The readback stalls the frame, which is why this sits on a key
            # rather than in the loop.
            try:
                pixels = observer_renderer.read_pixels()
            except bz.ResourceError as error:
                print(f"cannot measure: {error}")
            else:
                # Background is whatever colour covers most of the frame, not the
                # clear colour worked out by hand: the swapchain is sRGB, so 0.05
                # does not arrive as 13. Taking the mode asks the picture instead
                # of reproducing the driver's transfer function.
                rgb = pixels[:, :, :3].reshape(-1, 3)
                packed = (rgb[:, 0].astype(np.uint32) << 16
                          | rgb[:, 1].astype(np.uint32) << 8
                          | rgb[:, 2])
                counts = np.unique(packed, return_counts=True)[1]
                drawn = int(packed.size - counts.max())
                state = "on" if culling else "OFF"
                print(f"observer: {drawn} drawn pixels of {packed.size}, culling {state}")

    # The frame's own GPU cost, which is what culling changes. None until the
    # ring has cycled once (the number is the frame submitted frames_in_flight
    # ago), so it is read every frame and kept when it arrives.
    if gpu_timing_ok:
        try:
            measured = culled_renderer.gpu_time_ms
        except bz.UnsupportedError:
            # Advertised and unusable — MoltenVK on a paravirtual device does
            # this. The demo keeps running without the number.
            gpu_timing_ok = False
        else:
            if measured is not None:
                gpu_ms = measured

    frames += 1
    if time.time() - fps_timer >= 1.0:
        state = "on" if culling else "OFF"
        # Both numbers, because they answer different questions: the GPU figure
        # is the frame's work and the FPS is the loop around it, presentation
        # included. Culling moves the first one.
        cost = f" | GPU {gpu_ms:.3f} ms" if gpu_ms is not None else ""
        culled_window.set_title(
            f"Bazalt Demo - culled view | culling {state}{cost} | {frames} FPS")
        observer_window.set_title(
            f"Bazalt Demo - observer | WASD+QE move, RMB look | culling {state}")
        frames = 0
        fps_timer = time.time()
