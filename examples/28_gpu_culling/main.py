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
from about 99,000 to about 13,000. Fly out to the side and you can see the scene
has been cut to a wedge: cubes exist inside the yellow frustum and nowhere else,
and the wedge swings around as the culling camera turns.

About 1,570 of the 20,000 survive, cross-checked against the same test run on the
CPU — which is how the frustum-plane bug below was found, because a wrong plane
still produces a plausible-looking number.

Each frame:

  1. `cmd.fill_buffer(args, 0)` zeroes the draw arguments — the prerequisite that
     landed in 0.18 for exactly this, because a counter an atomic increments has to
     start each frame at a known value.
  2. A compute pass tests every cube against the culling camera's frustum planes,
     atomically increments `instanceCount`, and compacts the survivors.
  3. `cmd.draw_indexed_indirect(args)` draws whatever that came to — in BOTH
     windows, from one argument buffer.

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
culling camera, C toggles culling off. Close either window to exit.
"""

import struct
import time

import glm
import numpy as np

import bazalt as bz

logger = bz.Logger()
logger.on_message(lambda msg: print(f"[{msg.severity}] {msg.text}"))

ctx = bz.Context(logger)

culled_window = bz.Window(760, 560, "Bazalt Demo - culled view (the frustum)")
culled_window.set_position(60, 90)
observer_window = bz.Window(760, 560, "Bazalt Demo - observer (fly with WASD)")
observer_window.set_position(860, 90)

culled_renderer = ctx.create_renderer(culled_window)
observer_renderer = ctx.create_renderer(observer_window)

COUNT = 20000
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
        .storage_buffer(0, set=0)   # draw arguments
        .storage_buffer(1, set=0)   # candidates
        .storage_buffer(2, set=0)   # compacted survivors
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
            .storage_buffer(0, bz.ShaderStage.VERTEX, set=0)
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

pool = ctx.create_descriptor_pool(max_sets=8, storage_buffers=16)
cull_set = pool.allocate_set(cull, set=0)
cull_set.set_buffer(0, args)
cull_set.set_buffer(1, candidates)
cull_set.set_buffer(2, visible)
# The two cube pipelines are built from the same declarators, so they share a
# descriptor set layout and one set per SOURCE buffer serves both windows.
draw_set = pool.allocate_set(culled_pipeline, set=0)
draw_set.set_buffer(0, visible)
no_cull_set = pool.allocate_set(culled_pipeline, set=0)
no_cull_set.set_buffer(0, all_visible)

# Each window needs its own CommandBuffer: one holds a single command buffer per
# frame slot, so two windows recording into one would overwrite each other.
culled_cmd = ctx.create_command_buffer()
observer_cmd = ctx.create_command_buffer()

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

    if culled_window.was_key_pressed(bz.KEY_SPACE) or observer_window.was_key_pressed(bz.KEY_SPACE):
        paused = not paused
    if culled_window.was_key_pressed(bz.KEY_C) or observer_window.was_key_pressed(bz.KEY_C):
        culling = not culling

    now = time.time()
    dt = now - last
    last = now
    if not paused:
        cull_angle += dt * 0.35

    # ── the observer's free camera ────────────────────────────────────────
    if observer_window.is_mouse_button_pressed(bz.MOUSE_BUTTON_RIGHT):
        if not looking:
            observer_window.set_cursor_mode(bz.CURSOR_DISABLED)
            looking = True
        # CURSOR_DISABLED already gives unbounded virtual motion and does its own
        # recentring, so there is nothing to warp here. Calling set_cursor_position
        # every frame on top of it is the HIDDEN-cursor pattern, and mixing the two
        # cancels the look entirely: bazalt re-arms its first-event suppression on a
        # warp (so a warp is never mistaken for movement), so warping every frame
        # suppresses every frame's delta.
        mouse = observer_window.get_mouse_state()
        obs_yaw += mouse.dx * 0.15
        obs_pitch = max(min(obs_pitch + mouse.dy * 0.15, 89.0), -89.0)
    elif looking:
        observer_window.set_cursor_mode(bz.CURSOR_NORMAL)
        looking = False

    speed = 30.0 * dt
    forward = observer_forward()
    right = glm.normalize(glm.cross(forward, glm.vec3(0, 1, 0)))
    if observer_window.is_key_pressed(bz.KEY_W):
        obs_pos += forward * speed
    if observer_window.is_key_pressed(bz.KEY_S):
        obs_pos -= forward * speed
    if observer_window.is_key_pressed(bz.KEY_A):
        obs_pos -= right * speed
    if observer_window.is_key_pressed(bz.KEY_D):
        obs_pos += right * speed
    if observer_window.is_key_pressed(bz.KEY_E):
        obs_pos += glm.vec3(0, 1, 0) * speed
    if observer_window.is_key_pressed(bz.KEY_Q):
        obs_pos -= glm.vec3(0, 1, 0) * speed

    ctx.begin_frame()

    cull_vp = culling_view_proj(culled_window.width / max(culled_window.height, 1))
    cull_vp_bytes = bytes(glm.transpose(cull_vp))
    upload_frustum(cull_vp)

    source_set = draw_set if culling else no_cull_set
    source_args = args if culling else no_cull_args

    # ── window 1: the camera the culling is done for ──────────────────────
    if culled_renderer.acquire():
        culled_cmd.begin()
        if culling:
            culled_cmd.fill_buffer(args, 0)
            culled_cmd.bind_pipeline(cull)
            culled_cmd.bind_descriptor_set(cull_set, cull, set=0)
            culled_cmd.push_constants(cull, 0, cull_vp_bytes + struct.pack("II", COUNT, INDEX_COUNT))
            culled_cmd.dispatch((COUNT + 63) // 64)
        with culled_cmd.rendering(culled_renderer, clear_color=[0.03, 0.04, 0.07, 1.0]) as c:
            c.bind_pipeline(culled_pipeline)
            c.bind_descriptor_set(source_set, culled_pipeline, set=0)
            c.push_constants(culled_pipeline, 0, cull_vp_bytes)
            c.bind_vertex_buffer(vbuf).bind_index_buffer(ibuf)
            c.draw_indexed_indirect(source_args)
        culled_renderer.present(culled_cmd)

    # ── window 2: the observer, outside the frustum ───────────────────────
    if observer_renderer.acquire():
        obs_vp = bytes(glm.transpose(
            observer_view_proj(observer_window.width / max(observer_window.height, 1))))
        observer_cmd.begin()
        # The compute pass that fills `args` and `visible` runs in the OTHER
        # window's recording, and this one only reads them. Until 0.24 that needed
        # two manual barriers here, because the tracker's state is per recording
        # and this recording writes nothing it can see. It is automatic now: the
        # first READ of a buffer in a recording waits for whatever wrote it last,
        # wherever that was. cmd.barrier() is still there for the cases the
        # tracker cannot reach.
        with observer_cmd.rendering(observer_renderer, clear_color=[0.05, 0.05, 0.09, 1.0]) as c:
            c.bind_pipeline(observer_pipeline)
            c.bind_descriptor_set(source_set, observer_pipeline, set=0)
            c.push_constants(observer_pipeline, 0, obs_vp)
            c.bind_vertex_buffer(vbuf).bind_index_buffer(ibuf)
            c.draw_indexed_indirect(source_args)
            # The frustum the culling was done with, so the empty space has a shape.
            c.bind_pipeline(observer_lines)
            c.push_constants(observer_lines, 0, obs_vp)
            c.bind_vertex_buffer(frustum_lines)
            c.draw(len(FRUSTUM_EDGES) * 2)
        observer_renderer.present(observer_cmd)

    frames += 1
    if time.time() - fps_timer >= 1.0:
        state = "on" if culling else "OFF"
        culled_window.set_title(f"Bazalt Demo - culled view | culling {state} | {frames} FPS")
        observer_window.set_title(
            f"Bazalt Demo - observer | WASD+QE move, RMB look | culling {state}")
        frames = 0
        fps_timer = time.time()
