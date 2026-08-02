"""Window extras — drop a texture on the window and see it.

The four small things a prototype keeps needing and bazalt had no way to do:

  1. **Dropped files.** Drag a PNG onto the window and it becomes the texture.
     A drop expires with the poll cycle, like a key edge or a scroll notch, so
     `window.dropped_files()` reads the same twice inside one frame and clears on
     the next `poll_events()`.
  2. **A window icon.** `window.set_icon(rgba)` takes a (height, width, 4) uint8
     array. Press I to build one from the loaded texture, and O to go back to the
     system default. macOS and Wayland take the icon from elsewhere and will
     ignore this — a request, not a guarantee.
  3. **Cursor position.** Hold the right mouse button to look around, and press
     HOME to snap the cursor back to the centre. Holding uses CursorMode.DISABLED,
     which does its own recentring; `set_cursor_position` is the deliberate
     one-shot move, and a warp is never mistaken for the user moving the mouse.
  4. **The clipboard.** C copies the current texture's path, V loads whatever
     path is on the clipboard. Free functions, because the clipboard belongs to
     the process rather than to any one window.

Keys: C copies the path, V pastes one, I sets the icon, O clears it, HOME centres
the cursor, right mouse HELD looks around.
"""

import pathlib

import glm
import numpy as np

import bazalt as bz

logger = bz.Logger()
logger.on_message(lambda msg: print(f"[{msg.severity}] {msg.text}"))

window = bz.Window(1024, 720, "Bazalt Demo - drop a PNG on me", logger=logger)
ctx = bz.Context(logger, hot_reload=True)
renderer = ctx.create_renderer(window)

vert = ctx.compile_shader("quad.vert", bz.ShaderStage.VERTEX)
frag = ctx.compile_shader("quad.frag", bz.ShaderStage.FRAGMENT)

pipeline = (ctx.graphics_pipeline()
            .vertex_shader(vert)
            .fragment_shader(frag)
            .texture(0, bz.ShaderStage.FRAGMENT)
            .push_constant(64, bz.ShaderStage.VERTEX)
            .cull_mode(bz.CullMode.NONE, bz.FrontFace.COUNTER_CLOCKWISE)
            .build(renderer))

sampler = ctx.create_sampler()
pool = ctx.create_descriptor_pool()
desc_set = pool.allocate_set(pipeline)

# A checkerboard until something is dropped, so there is always a texture bound.
checker = np.zeros((64, 64, 4), dtype=np.uint8)
checker[..., 3] = 255
checker[::16, :, :3] = 90
checker[:, ::16, :3] = 90
texture = ctx.create_image(checker, name="placeholder")
texture_path = "(no file: the built-in checkerboard)"
desc_set.set_image(0, texture, sampler)


def load(path):
    """Swap the bound texture for the image at `path`, keeping the old one on any
    failure — the same rule hot reload follows for a bad shader edit."""
    global texture, texture_path
    try:
        loaded = ctx.load_image(str(path))
    except bz.ResourceError as e:
        print(f"[skip] {path}: {e}")
        return

    # Rewriting a descriptor set that a submitted frame is still reading is
    # illegal: Vulkan only allows it for bindings created with
    # UPDATE_AFTER_BIND, which is a descriptor-indexing feature. With frames in
    # flight, the previous frame's command buffer still references this set.
    #
    # So wait first. A drop or a paste happens when a person does something, not
    # every frame, so the stall costs nothing here — and it is the honest fix
    # rather than a second descriptor set to juggle.
    ctx.wait()

    texture = loaded
    texture_path = str(path)
    desc_set.set_image(0, texture, sampler)
    window.set_title(f"Bazalt Demo - {pathlib.Path(path).name}")
    print(f"[load] {path}")


def icon_from(image):
    """An icon out of whatever is on screen. read() gives RGBA8 already, and
    set_icon wants exactly that plus a small size — 32x32 by nearest sampling,
    which is enough for a task bar and keeps this to two lines."""
    pixels = image.read()
    step_y = max(pixels.shape[0] // 32, 1)
    step_x = max(pixels.shape[1] // 32, 1)
    # ascontiguousarray because the slice above is a strided view, and set_icon
    # refuses those rather than silently copying other bytes.
    return np.ascontiguousarray(pixels[::step_y, ::step_x][:32, :32])


yaw = 0.0
pitch = 0.0
looking = False

while window.is_open():
    bz.poll_events()

    # 1. Dropped files. Per-cycle state: empty on almost every frame, and reading
    #    it does not consume, so another reader this frame sees the same list.
    for path in window.dropped_files():
        load(path)

    # 4. The clipboard, as a pair of free functions.
    if window.was_key_pressed(bz.Key.C):
        bz.set_clipboard(texture_path)
        print(f"[copy] {texture_path}")
    if window.was_key_pressed(bz.Key.V):
        pasted = bz.get_clipboard().strip().strip('"')
        if pasted:
            load(pasted)
        else:
            print("[paste] the clipboard has no text")

    # 2. The window icon.
    if window.was_key_pressed(bz.Key.I):
        window.set_icon(icon_from(texture))
        print("[icon] set from the current texture")
    if window.was_key_pressed(bz.Key.O):
        window.set_icon(None)
        print("[icon] back to the system default")

    # 3. Look around while the right button is HELD. CursorMode.DISABLED hides the
    #    cursor and hands out unbounded virtual motion, doing its own recentring —
    #    so there is nothing here to warp, and get_mouse_state().dx/dy is the
    #    movement.
    if window.is_mouse_button_pressed(bz.MouseButton.RIGHT):
        if not looking:
            window.set_cursor_mode(bz.CursorMode.DISABLED)
            looking = True
        mouse = window.get_mouse_state()
        yaw += mouse.dx * 0.25
        pitch = max(min(pitch + mouse.dy * 0.25, 85.0), -85.0)
    elif looking:
        window.set_cursor_mode(bz.CursorMode.NORMAL)
        looking = False

    #    set_cursor_position is for putting the cursor somewhere on purpose, which
    #    is what HOME does here. It must NOT be combined with CursorMode.DISABLED and a
    #    per-frame warp: bazalt re-arms its first-event suppression on a warp, so
    #    that a warp is never mistaken for the user moving the mouse — and warping
    #    every frame therefore cancels every frame's delta. That suppression is what
    #    makes the hidden-cursor recentring pattern work, and what makes this
    #    one-shot snap leave the camera alone.
    if window.was_key_pressed(bz.Key.HOME):
        window.set_cursor_position(window.width / 2, window.height / 2)
        print("[cursor] snapped to the centre")

    ctx.begin_frame()
    if not renderer.acquire():
        continue

    # Built per frame from the renderer, so a resize does not stretch the quad.
    proj = glm.perspectiveRH_ZO(glm.radians(50.0), renderer.width / renderer.height,
                                0.1, 100.0)
    proj[1][1] *= -1
    view = glm.lookAt(glm.vec3(0, 0, 2.6), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
    model = glm.rotate(glm.mat4(1.0), glm.radians(yaw), glm.vec3(0, 1, 0))
    model = glm.rotate(model, glm.radians(pitch), glm.vec3(1, 0, 0))

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(renderer, clear_color=[0.05, 0.06, 0.09, 1.0]) as c:
        c.bind_pipeline(pipeline)
        c.bind_descriptor_set(desc_set, pipeline)
        c.push_constants(pipeline, 0, bytes(glm.transpose(proj * view * model)))
        c.draw(6)
    renderer.present(cmd)
