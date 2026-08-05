"""Multi-window — two windows, one Context.

The 0.14 release feature. Two `SwapchainRenderer`s share one Context, one
pipeline and one mesh; each window has its own swapchain, its own camera and
its own present mode (window B runs FIFO_RELAXED, the mode 0.14 added).

The shape of the loop is the point:

    ctx.begin_frame()          # the FRAME — once, on the Context
    if renderer.acquire():     # the WINDOW — once per window
        renderer.present(cmd)

`ctx.begin_frame()` advances the ring slot that CommandBuffer, DynamicBuffer and
the per-frame descriptor sets index — all Context-owned, so a window has no
business advancing it. A window contributes one acquired image and one present.
With a single window this reads exactly as it does in every other example.

Two things each window needs of its own, because both are per (window, slot):
its own CommandBuffer (one holds a single command buffer per frame slot) and
its own DynamicBuffer for the camera.

Close either window and it disappears while the other keeps rendering — which
is why `bz.poll_events()` is a free function and not a Window method: GLFW's
event queue is process-wide, so a loop whose windows come and go has nothing
sensible to call a method on. Close both to exit.
"""

import math
import time

import glm
import numpy as np

import bazalt as bz

logger = bz.Logger()
logger.on_message(lambda msg: print(f"[{msg.severity}] {msg.text}"))

# One Context, and it says which GPU it picked. `bz.list_devices()` (also new in
# 0.14) shows what else was on offer — pass one as Context(device=...).
ctx = bz.Context(logger)
for d in bz.list_devices():
    mark = "*" if d.name == ctx.device_name else " "
    print(f" {mark} {d.name} ({d.type}, {d.limits.device_memory // 2**20} MB, Vulkan {d.api_version})")

window_a = bz.Window(800, 600, "Bazalt - Window A (MAILBOX)", logger=logger)
window_b = bz.Window(800, 600, "Bazalt - Window B (FIFO_RELAXED)", logger=logger)

renderer_a = ctx.create_renderer(window_a, present_mode=bz.PresentMode.MAILBOX)
renderer_b = ctx.create_renderer(window_b, present_mode=bz.PresentMode.FIFO_RELAXED)
print(f"A: {renderer_a.present_mode}   B: {renderer_b.present_mode}")

vert = ctx.compile_shader("scene.vert", bz.ShaderStage.VERTEX)
frag = ctx.compile_shader("scene.frag", bz.ShaderStage.FRAGMENT)

# One pipeline for both windows: it reads formats and sample count off the
# target it builds against, and both swapchains negotiate the same ones.
pipeline = (ctx.graphics_pipeline()
    .vertex_shader(vert)
    .fragment_shader(frag)
    .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
    .depth_test(True)
    .cull_mode(bz.CullMode.BACK, bz.FrontFace.COUNTER_CLOCKWISE)
    .uniform_buffer(0, bz.ShaderStage.VERTEX)
    .push_constant(12, bz.ShaderStage.FRAGMENT)
    .build(renderer_a))

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

pool = ctx.create_descriptor_pool()


class View:
    """Everything one window owns. The per-window CommandBuffer and uniform
    buffer are not a style choice: both hold one copy per ring slot, and both
    windows now render on the SAME slot."""

    def __init__(self, window, renderer, title, tint, clear, orbit_speed):
        self.window = window
        self.title = title
        self.renderer = renderer
        self.tint = np.array(tint, dtype=np.float32).tobytes()
        self.orbit_speed = orbit_speed
        self.open = True
        self.frames = 0
        self.fps_timer = time.time()

        self.ubuf = ctx.create_buffer(np.zeros(16, dtype=np.float32),
                                      bz.BufferType.UNIFORM, bz.MemoryUsage.DYNAMIC)
        self.dset = pool.allocate_frame_set(pipeline)
        self.dset.set_buffer(0, self.ubuf)

        self.cmd = ctx.create_command_buffer()
        self.cmd.begin()
        with self.cmd.rendering(renderer, clear_color=clear) as c:
            (c.bind_pipeline(pipeline)
              .bind_descriptor_set(self.dset, pipeline)
              .push_constants(pipeline, 0, self.tint)
              .bind_vertex_buffer(vbuf)
              .bind_index_buffer(ibuf)
              .draw_indexed(36))

    def draw(self, t):
        """One window's contribution to the frame the Context already opened."""
        if not self.open:
            return
        if not self.window.is_open():
            self.open = False
            return
        if not self.renderer.acquire():
            return

        angle = t * self.orbit_speed
        eye = glm.vec3(math.cos(angle) * 3.0, 1.2, math.sin(angle) * 3.0)
        aspect = max(self.renderer.width, 1) / max(self.renderer.height, 1)
        proj = glm.perspectiveRH_ZO(glm.radians(45.0), aspect, 0.1, 100.0)
        proj[1][1] *= -1
        view = glm.lookAt(eye, glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
        self.ubuf.update(bytes(glm.transpose(proj * view)))

        self.renderer.present(self.cmd)

        self.frames += 1
        if time.time() - self.fps_timer >= 1.0:
            self.window.set_title(f"{self.title} | {self.frames} FPS")
            self.frames = 0
            self.fps_timer = time.time()

    def close(self):
        """Drop the swapchain BEFORE the window, in that order: the renderer
        owns a VkSurfaceKHR created from this window's handle. Dropping the
        last reference is what makes the window disappear, so nothing outside
        this object may keep one."""
        self.cmd = None
        self.dset = None
        self.ubuf = None
        self.renderer = None
        self.window = None


views = [
    View(window_a, renderer_a, "Bazalt - Window A (MAILBOX)",
         tint=(1.0, 0.45, 0.2), clear=[0.02, 0.02, 0.05, 1.0], orbit_speed=0.6),
    View(window_b, renderer_b, "Bazalt - Window B (FIFO_RELAXED)",
         tint=(0.25, 0.7, 1.0), clear=[0.05, 0.02, 0.03, 1.0], orbit_speed=-0.9),
]

# The Views own the windows and renderers now. These names have to go, or a
# closed window stays on screen as a frozen husk nobody can free.
window_a = window_b = None
renderer_a = renderer_b = None

start = time.time()
while any(v.open for v in views):
    # Process-wide, so one call services every window — including none.
    bz.poll_events()

    # ONE frame, however many windows draw into it.
    ctx.begin_frame()

    t = time.time() - start
    for v in views:
        v.draw(t)

    # Release a closed window's swapchain and handle right away; the survivors
    # keep rendering into the same Context.
    for v in views:
        if not v.open:
            v.close()

for v in views:
    v.close()
