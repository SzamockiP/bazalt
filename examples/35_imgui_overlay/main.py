"""ImGui overlay — sliders that tune the picture while it runs.

This is the case a prototyping library exists for: change a number, see the
result, without editing a file and waiting for a reload. The scene is a shader
with five knobs, and every knob is a widget.

**Nothing in bazalt knows about ImGui.** The backend below is about 90 lines of
ordinary calls — a pipeline, a dynamic vertex buffer, push constants, a scissor
and draw_indexed — and any immediate-mode UI would need the same ones. That is
why the integration is an example and not a module in the library: there is
nothing here to share.

Two things did have to arrive in 0.25, because no UI can supply them for itself:

  * `window.text_input()` — the characters typed, which is what feeds
    `io.add_input_characters_utf8`. A physical key is not a character: the
    layout, shift, AltGr, the dead keys and an IME all sit in between. Without
    it an InputText field is deaf, so type in the title box to see it work.
  * `window.set_cursor(shape)` — ImGui asks for an I-beam over a text field and a
    resize arrow over a panel edge, and the ten standard shapes map onto it
    one for one.

Needs pyimgui: `pip install imgui`. Everything else is bazalt.

Keys: ESC quits. Drag the panel, type in the title field, turn the knobs.
"""

import ctypes
import struct
import time

import imgui
import numpy as np

import bazalt as bz

logger = bz.Logger()
logger.on_message(lambda msg: print(f"[{msg.severity}] {msg.text}"))

window = bz.Window(1100, 720, "Bazalt Demo - ImGui overlay", logger=logger)
ctx = bz.Context(logger)
renderer = ctx.create_renderer(window)


# ── the backend ───────────────────────────────────────────────────────────
# One class so it can be copied into a project whole. It owns a pipeline, one
# descriptor set for the font atlas and a pair of dynamic buffers that grow.

# ImGui reports which pointer it wants; bazalt draws it. The two enums line up
# because both are the same ten shapes every toolkit has.
CURSORS = {
    imgui.MOUSE_CURSOR_ARROW: bz.Cursor.ARROW,
    imgui.MOUSE_CURSOR_TEXT_INPUT: bz.Cursor.IBEAM,
    imgui.MOUSE_CURSOR_RESIZE_ALL: bz.Cursor.RESIZE_ALL,
    imgui.MOUSE_CURSOR_RESIZE_NS: bz.Cursor.RESIZE_NS,
    imgui.MOUSE_CURSOR_RESIZE_EW: bz.Cursor.RESIZE_EW,
    imgui.MOUSE_CURSOR_RESIZE_NESW: bz.Cursor.RESIZE_NESW,
    imgui.MOUSE_CURSOR_RESIZE_NWSE: bz.Cursor.RESIZE_NWSE,
    imgui.MOUSE_CURSOR_HAND: bz.Cursor.POINTING_HAND,
    imgui.MOUSE_CURSOR_NOT_ALLOWED: bz.Cursor.NOT_ALLOWED,
}

# The keys a text field needs. ImGui indexes io.keys_down by whatever numbers we
# put in io.key_map, so the GLFW values bazalt already uses go straight in.
KEY_MAP = {
    imgui.KEY_TAB: bz.Key.TAB,
    imgui.KEY_LEFT_ARROW: bz.Key.LEFT,
    imgui.KEY_RIGHT_ARROW: bz.Key.RIGHT,
    imgui.KEY_UP_ARROW: bz.Key.UP,
    imgui.KEY_DOWN_ARROW: bz.Key.DOWN,
    imgui.KEY_PAGE_UP: bz.Key.PAGE_UP,
    imgui.KEY_PAGE_DOWN: bz.Key.PAGE_DOWN,
    imgui.KEY_HOME: bz.Key.HOME,
    imgui.KEY_END: bz.Key.END,
    imgui.KEY_INSERT: bz.Key.INSERT,
    imgui.KEY_DELETE: bz.Key.DELETE,
    imgui.KEY_BACKSPACE: bz.Key.BACKSPACE,
    imgui.KEY_SPACE: bz.Key.SPACE,
    imgui.KEY_ENTER: bz.Key.ENTER,
    imgui.KEY_ESCAPE: bz.Key.ESCAPE,
    imgui.KEY_PAD_ENTER: bz.Key.KP_ENTER,
    imgui.KEY_A: bz.Key.A,
    imgui.KEY_C: bz.Key.C,
    imgui.KEY_V: bz.Key.V,
    imgui.KEY_X: bz.Key.X,
}


class ImGuiOverlay:
    def __init__(self, ctx, window, target):
        self.ctx = ctx
        self.window = window
        imgui.create_context()
        io = imgui.get_io()
        for imgui_key, bazalt_key in KEY_MAP.items():
            io.key_map[imgui_key] = int(bazalt_key)

        # The atlas is one ordinary texture: glyphs in the alpha channel plus a
        # white block every solid rectangle samples.
        width, height, pixels = io.fonts.get_tex_data_as_rgba32()
        atlas = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 4)
        self.font = ctx.create_image(atlas, name="imgui atlas")
        # texture_id is ImGui's opaque handle. One atlas here, so it is a
        # constant and every draw command reports it.
        io.fonts.texture_id = 1
        io.fonts.clear_tex_data()

        vert = ctx.compile_shader("ui.vert", bz.ShaderStage.VERTEX)
        frag = ctx.compile_shader("ui.frag", bz.ShaderStage.FRAGMENT)
        self.pipeline = (ctx.graphics_pipeline()
                         .vertex_shader(vert)
                         .fragment_shader(frag)
                         # 20 bytes per vertex, which is what imgui.VERTEX_SIZE
                         # reports: two floats, two floats, four bytes.
                         .vertex_format([bz.VertexFormat.FLOAT2,
                                         bz.VertexFormat.FLOAT2,
                                         bz.VertexFormat.UBYTE4_NORM])
                         .push_constant(16, bz.ShaderStage.VERTEX)
                         .texture(0, bz.ShaderStage.FRAGMENT)
                         .blend(True, mode=bz.BlendMode.ALPHA)
                         # ImGui does not wind its triangles consistently, and it
                         # has no depth to test against.
                         .cull_mode(bz.CullMode.NONE)
                         .depth_test(False)
                         .build(target))

        pool = ctx.create_descriptor_pool()
        self.dset = pool.allocate_set(self.pipeline)
        self.dset.set_image(0, self.font, sampler=ctx.create_sampler(filter=bz.Filter.LINEAR))

        # Grown on demand rather than sized by a guess: a UI's vertex count
        # depends on what the user opened.
        self.vertex_bytes = 0
        self.index_bytes = 0
        self.vbuf = None
        self.ibuf = None
        self.mouse_was_down = [False, False, False]

    def new_frame(self, dt):
        """Feed ImGui what the last poll cycle saw, then open a frame."""
        io = imgui.get_io()
        io.display_size = renderer.width, renderer.height
        io.delta_time = max(dt, 1e-4)

        # The cursor is reported in WINDOW coordinates and the UI is laid out in
        # FRAMEBUFFER pixels, which are the same number until somebody runs this
        # on a HiDPI display. window.width and renderer.width are the two sides
        # of that ratio.
        mouse = self.window.get_mouse_state()
        scale_x = renderer.width / max(self.window.width, 1)
        scale_y = renderer.height / max(self.window.height, 1)
        io.mouse_pos = mouse.x * scale_x, mouse.y * scale_y
        for button in range(3):
            io.mouse_down[button] = self.window.is_mouse_button_pressed(button)
        io.mouse_wheel = mouse.scroll_dy
        io.mouse_wheel_horizontal = mouse.scroll_dx

        for key in KEY_MAP.values():
            io.keys_down[int(key)] = self.window.is_key_pressed(key)
        io.key_ctrl = (self.window.is_key_pressed(bz.Key.LEFT_CONTROL)
                       or self.window.is_key_pressed(bz.Key.RIGHT_CONTROL))
        io.key_shift = (self.window.is_key_pressed(bz.Key.LEFT_SHIFT)
                        or self.window.is_key_pressed(bz.Key.RIGHT_SHIFT))
        io.key_alt = (self.window.is_key_pressed(bz.Key.LEFT_ALT)
                      or self.window.is_key_pressed(bz.Key.RIGHT_ALT))

        # The one call nothing else can make. text_input() is the characters the
        # user typed, after every layout rule the OS applies; is_key_pressed
        # cannot produce them.
        typed = self.window.text_input()
        if typed:
            io.add_input_characters_utf8(typed)

        imgui.new_frame()

    def draw(self, cmd):
        """Render whatever the frame built. Call inside a rendering scope."""
        imgui.render()
        data = imgui.get_draw_data()
        if data is None or not data.commands_lists:
            return

        # ImGui hands out one buffer per command list; they are concatenated so
        # the whole UI is one pair of buffers and one bind.
        vertex_blocks = []
        index_blocks = []
        draws = []
        first_index = 0
        base_vertex = 0
        for commands in data.commands_lists:
            vertex_blocks.append(self._read(commands.vtx_buffer_data,
                                            commands.vtx_buffer_size * imgui.VERTEX_SIZE, np.uint8))
            indices = self._read(commands.idx_buffer_data,
                                 commands.idx_buffer_size * imgui.INDEX_SIZE, np.uint32)
            index_blocks.append(indices + base_vertex)
            for command in commands.commands:
                draws.append((command.elem_count, first_index, command.clip_rect))
                first_index += command.elem_count
            base_vertex += commands.vtx_buffer_size

        vertices = np.concatenate(vertex_blocks)
        indices = np.concatenate(index_blocks)
        self._upload(vertices, indices)

        # Pixels to clip space. Y is not flipped: ImGui's Y points down and so
        # does Vulkan's.
        width, height = imgui.get_io().display_size
        push = struct.pack("4f", 2.0 / width, 2.0 / height, -1.0, -1.0)

        cmd.bind_pipeline(self.pipeline)
        cmd.bind_descriptor_set(self.dset)
        cmd.push_constants(0, push)
        cmd.bind_vertex_buffer(self.vbuf)
        cmd.bind_index_buffer(self.ibuf)
        for elem_count, index_offset, clip in draws:
            x = max(int(clip.x), 0)
            y = max(int(clip.y), 0)
            cmd.set_scissor(x, y, max(int(clip.z) - x, 0), max(int(clip.w) - y, 0))
            cmd.draw_indexed(elem_count, first_index=index_offset)
        # The scissor is pipeline state the next recording inherits nothing of,
        # but the same recording would, so put it back.
        cmd.set_scissor(0, 0, int(width), int(height))

    def apply_cursor(self):
        """What ImGui wants the pointer to look like right now."""
        wanted = imgui.get_mouse_cursor()
        if wanted == imgui.MOUSE_CURSOR_NONE:
            self.window.set_cursor_mode(bz.CursorMode.HIDDEN)
            return
        self.window.set_cursor_mode(bz.CursorMode.NORMAL)
        self.window.set_cursor(CURSORS.get(wanted, bz.Cursor.ARROW))

    @staticmethod
    def _read(address, size_in_bytes, dtype):
        """A view of ImGui's own memory, copied once. No per-vertex Python."""
        raw = (ctypes.c_ubyte * size_in_bytes).from_address(address)
        return np.frombuffer(raw, dtype=dtype).copy()

    def _upload(self, vertices, indices):
        if self.vbuf is None or vertices.nbytes > self.vertex_bytes:
            self.vertex_bytes = max(vertices.nbytes * 2, 64 * 1024)
            self.vbuf = self.ctx.create_buffer(self.vertex_bytes, bz.BufferType.VERTEX,
                                               bz.MemoryUsage.DYNAMIC, name="imgui vertices")
        if self.ibuf is None or indices.nbytes > self.index_bytes:
            self.index_bytes = max(indices.nbytes * 2, 32 * 1024)
            self.ibuf = self.ctx.create_buffer(self.index_bytes, bz.BufferType.INDEX,
                                               bz.MemoryUsage.DYNAMIC, name="imgui indices")
        self.vbuf.update(vertices)
        self.ibuf.update(indices)


# ── the scene the overlay tunes ───────────────────────────────────────────

scene_vert = ctx.compile_shader("scene.vert", bz.ShaderStage.VERTEX)
scene_frag = ctx.compile_shader("scene.frag", bz.ShaderStage.FRAGMENT)
scene = (ctx.graphics_pipeline()
         .vertex_shader(scene_vert)
         .fragment_shader(scene_frag)
         .push_constant(32, bz.ShaderStage.FRAGMENT)
         .build(renderer))

overlay = ImGuiOverlay(ctx, window, renderer)

tint = [0.35, 0.72, 0.95]
scale = 6.0
speed = 1.0
warp = 1.6
animate = True
title = "Bazalt Demo - ImGui overlay"

print("drag the panel, turn the knobs, and type in the title box. ESC quits.")
start = time.perf_counter()
last = start
elapsed = 0.0

while window.is_open():
    bz.poll_events()
    if window.is_key_pressed(bz.Key.ESCAPE):
        break

    now = time.perf_counter()
    dt = now - last
    last = now
    if animate:
        elapsed += dt

    overlay.new_frame(dt)

    imgui.begin("Scene", closable=False)
    imgui.text(f"{1.0 / max(dt, 1e-4):5.1f} fps")
    imgui.separator()
    changed, tint = imgui.color_edit3("tint", *tint)
    _, scale = imgui.slider_float("scale", scale, 1.0, 20.0)
    _, speed = imgui.slider_float("speed", speed, 0.0, 4.0)
    _, warp = imgui.slider_float("warp", warp, 0.0, 6.0)
    _, animate = imgui.checkbox("animate", animate)
    imgui.separator()
    # The reason text_input() had to exist. Typing here reaches ImGui only
    # through window.text_input().
    retyped, title = imgui.input_text("title", title, 128)
    if retyped:
        window.set_title(title)
    imgui.end()

    overlay.apply_cursor()

    scene_push = struct.pack("4f4f", *tint, 1.0, elapsed, scale, speed, warp)

    with ctx.record() as cmd:
        with cmd.rendering(renderer, clear_color=[0.05, 0.05, 0.07, 1.0]):
            cmd.bind_pipeline(scene)
            cmd.push_constants(0, scene_push)
            cmd.draw(3)
            overlay.draw(cmd)

    ctx.begin_frame()
    if renderer.acquire():
        renderer.present(cmd)
