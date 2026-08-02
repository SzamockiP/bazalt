"""Window modes and per-frame input — the 0.16 release feature.

    F11        cycle windowed -> frameless -> fullscreen -> borderless fullscreen
    L          wireframe on/off        (polygon_mode, needs Feature.WIREFRAME)
    V          vsync on/off            (renderer.set_present_mode)
    T          always on top
    O          cycle the window opacity
    scroll     zoom
    mouse      look
    ESC        quit

Three things this demo is built out of, and each one is new in 0.16.

`window.set_mode()` is one verb over four exclusive states, not a set_fullscreen
plus a set_decorated plus the rules for combining them. The swapchain needs
nothing from the caller: every mode change resizes the framebuffer, and the
resize path already recreates the swapchain.

`window.was_key_pressed()` is the edge where `is_key_pressed` is the level. A
toggle needs the edge. Before it existed, every caller kept its own "was it down
last frame" bool. It reads the same twice inside one frame, so asking in two
places is safe.

The bar along the bottom is a SECOND pass into the same swapchain image, which
is what `clear_color=None` buys: the first pass clears and draws the scene, the
second preserves what is there and adds to it. `blend(True,
mode=bz.BlendMode.ADDITIVE)` is what makes it glow instead of replacing, and a
scissor is what keeps it to a strip.
"""

import math
import struct
import time

import glm
import numpy as np

import bazalt as bz

logger = bz.Logger()
logger.on_message(lambda msg: print(f"[{msg.severity}] {msg.text}"))

# WIREFRAME is optional=, not features=: a machine without fillModeNonSolid
# still runs the demo, only without the L key. Asking for it in features= would
# refuse to start over a debug view.
ctx = bz.Context(logger, optional=[bz.Feature.WIREFRAME])

window = bz.Window(900, 600, "Bazalt - Window Modes", logger=logger)
renderer = ctx.create_renderer(window, present_mode=bz.PresentMode.FIFO)

scene_vert = ctx.compile_shader("scene.vert", bz.ShaderStage.VERTEX)
scene_frag = ctx.compile_shader("scene.frag", bz.ShaderStage.FRAGMENT)
overlay_vert = ctx.compile_shader("overlay.vert", bz.ShaderStage.VERTEX)
overlay_frag = ctx.compile_shader("overlay.frag", bz.ShaderStage.FRAGMENT)


def scene_pipeline(polygon_mode):
    return (ctx.graphics_pipeline()
        .vertex_shader(scene_vert)
        .fragment_shader(scene_frag)
        .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
        .depth_test(True)
        .polygon_mode(polygon_mode)
        .cull_mode(bz.CullMode.BACK, bz.FrontFace.COUNTER_CLOCKWISE)
        .uniform_buffer(0, bz.ShaderStage.VERTEX)
        .build(renderer))


filled = scene_pipeline(bz.PolygonMode.FILL)
wireframe = scene_pipeline(bz.PolygonMode.LINE) if ctx.supports(bz.Feature.WIREFRAME) else None

# The overlay adds light to what is already there, so it needs no depth and no
# opaque write. depth_test(False) matters: the bar sits over the scene whatever
# the cube did to the depth buffer.
overlay = (ctx.graphics_pipeline()
    .vertex_shader(overlay_vert)
    .fragment_shader(overlay_frag)
    .depth_test(False)
    .blend(True, mode=bz.BlendMode.ADDITIVE)
    .push_constant(12, bz.ShaderStage.FRAGMENT)
    .build(renderer))

# pos (x,y,z), normal (nx,ny,nz)
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
vbuf = ctx.create_buffer(vertices, bz.BufferType.VERTEX, bz.MemoryUsage.STATIC)

indices = np.array([
    0, 1, 2, 2, 3, 0,
    5, 4, 7, 7, 6, 5,
    8, 9, 10, 10, 11, 8,
    13, 12, 15, 15, 14, 13,
    16, 17, 18, 18, 19, 16,
    23, 22, 21, 21, 20, 23,
], dtype=np.uint32)
ibuf = ctx.create_buffer(indices, bz.BufferType.INDEX, bz.MemoryUsage.STATIC)

ubuf = ctx.create_buffer(np.zeros(16, dtype=np.float32),
                         bz.BufferType.UNIFORM, bz.MemoryUsage.DYNAMIC)
pool = ctx.create_descriptor_pool()
dset = pool.allocate_frame_set(filled)
dset.set_buffer(0, ubuf)

MODES = [bz.WindowMode.WINDOWED, bz.WindowMode.FRAMELESS,
         bz.WindowMode.FULLSCREEN, bz.WindowMode.FULLSCREEN_WINDOWED]
MODE_NAMES = ["windowed", "frameless", "fullscreen", "borderless fullscreen"]
OPACITIES = [1.0, 0.85, 0.65]

BAR_HEIGHT = 46

cmd = ctx.create_command_buffer()


def record(scene):
    """Re-record both passes. Cheap: a recording is lambdas, and it replays every
    frame until something about it changes (here, the L key)."""
    cmd.begin()

    # Pass 1 clears and owns the depth.
    with cmd.rendering(renderer, clear_color=[0.02, 0.02, 0.05, 1.0]) as c:
        (c.bind_pipeline(scene)
          .bind_descriptor_set(dset, scene)
          .bind_vertex_buffer(vbuf)
          .bind_index_buffer(ibuf)
          .draw_indexed(36))

    # Pass 2 preserves it. clear_color=None is the whole difference between a
    # second pass and a second frame.
    with cmd.rendering(renderer, clear_color=None) as c:
        # The scissor is read at replay time, so a resize needs no re-record...
        # except that these numbers are computed now. Recording again on resize
        # would be the fix if the bar had to track the height exactly; a strip
        # anchored to the bottom does not need it.
        c.set_scissor(0, max(renderer.height - BAR_HEIGHT, 0), renderer.width, BAR_HEIGHT)
        c.bind_pipeline(overlay)
        c.push_constants(overlay, 0, bar_bytes())
        c.draw(3)


def bar_bytes():
    """The push constants are read when `record` runs, so the recording carries
    the bar state. Whatever changes it re-records — the same rule as the L key."""
    return struct.pack("3f", float(mode_index), float(len(MODES)), glow)


mode_index = 0
opacity_index = 0
glow = 1.0
use_wireframe = False
vsync = True

record(filled)

camera_yaw = 0.6
camera_pitch = -0.35
distance = 3.2
frames = 0
fps_timer = time.time()
start = time.time()

print(__doc__)
if wireframe is None:
    print("This GPU has no fillModeNonSolid, so the L key does nothing.")

while window.is_open():
    bz.poll_events()

    if window.is_key_pressed(bz.Key.ESCAPE):
        break

    # The edge, not the level: is_key_pressed would cycle the mode every frame
    # the key stays down.
    if window.was_key_pressed(bz.Key.F11):
        mode_index = (mode_index + 1) % len(MODES)
        window.set_mode(MODES[mode_index])
        print(f"mode: {MODE_NAMES[mode_index]}")

    if window.was_key_pressed(bz.Key.L) and wireframe is not None:
        use_wireframe = not use_wireframe

    if window.was_key_pressed(bz.Key.V):
        vsync = not vsync
        renderer.set_present_mode(bz.PresentMode.FIFO if vsync else bz.PresentMode.IMMEDIATE)
        # A request is a preference: the driver may not have the mode.
        print(f"present mode: {renderer.present_mode}")

    if window.was_key_pressed(bz.Key.T):
        window.set_always_on_top(not window.always_on_top)
        print(f"always on top: {window.always_on_top}")

    if window.was_key_pressed(bz.Key.O):
        opacity_index = (opacity_index + 1) % len(OPACITIES)
        window.set_opacity(OPACITIES[opacity_index])

    mouse = window.get_mouse_state()

    # dx/dy are this cycle's delta, so there is nothing to subtract. Left button
    # held to look, so the cursor stays usable everywhere else.
    if window.is_mouse_button_pressed(bz.MouseButton.LEFT):
        camera_yaw += mouse.dx * 0.005
        camera_pitch = max(-1.4, min(1.4, camera_pitch + mouse.dy * 0.005))

    # One notch of the wheel is 1.0, whatever the platform.
    if mouse.scroll_dy:
        distance = max(1.5, min(12.0, distance - mouse.scroll_dy * 0.4))

    # The glow breathes and the recording carries it, so this re-records every
    # frame. That is affordable here — a recording is a handful of lambdas — and
    # it is also why the push constants live in a function: whatever changes
    # them re-records, and there is exactly one rule about when that happens.
    glow = 0.55 + 0.45 * math.sin((time.time() - start) * 2.0)
    record(wireframe if use_wireframe else filled)

    ctx.begin_frame()
    if not renderer.acquire():
        continue

    eye = glm.vec3(
        math.cos(camera_yaw) * math.cos(camera_pitch) * distance,
        math.sin(camera_pitch) * distance,
        math.sin(camera_yaw) * math.cos(camera_pitch) * distance,
    )
    aspect = max(renderer.width, 1) / max(renderer.height, 1)
    proj = glm.perspectiveRH_ZO(glm.radians(50.0), aspect, 0.1, 100.0)
    proj[1][1] *= -1
    view = glm.lookAt(eye, glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
    ubuf.update(bytes(glm.transpose(proj * view)))

    renderer.present(cmd)

    frames += 1
    if time.time() - fps_timer >= 1.0:
        scale_x, _ = window.content_scale
        window.set_title(
            f"Bazalt - {MODE_NAMES[mode_index]} | {renderer.present_mode} | "
            f"{renderer.width}x{renderer.height} @ {scale_x:g}x | {frames} FPS")
        frames = 0
        fps_timer = time.time()

# The renderer holds a surface made from the window, so it goes first.
cmd = None
renderer = None
window = None
