"""CPU image streaming: a texture whose pixels change every frame.

The 0.18 headline. Before it, changing the pixels of an existing image had no
spelling at all, so a video frame, a camera feed, a matplotlib figure or a
painted texture each meant creating a new bz.Image per frame.

Three things are on show:

  * `img.update(frame)` rewrites the whole texture from a numpy array. It is
    asynchronous, on the same worker as load_image, so the call returns while
    the copy is still queued and the frame that samples the image waits for it
    GPU-side. Two updates of one image land in the order they were called.
  * `img.update(patch, region=(x, y, w, h))` rewrites a rectangle and leaves
    the rest alone — click and drag to paint a white square into the stream.
  * `with cmd.label(...)` names the pass, so a RenderDoc capture reads as a
    frame rather than a list of draws, and `ctx.memory_stats()` says whether
    streaming is leaking.

Press S to save a screenshot of the window with
`renderer.present(cmd, capture=True)` + `renderer.read_pixels()`. It takes two
calls because a presentable image may only be touched between acquire and
present, so the copy has to ride the frame's own submit.

There is no video decoder here on purpose — bazalt does not decode video, and
the ecosystem does. `frame_at()` stands in for whatever produces your pixels:
swap it for cv2.VideoCapture.read(), a PIL frame, or a matplotlib canvas.
"""

import numpy as np

import bazalt as bz

W, H = 512, 512
TEX = 256  # the streamed texture is smaller than the window, on purpose

logger = bz.Logger()


@logger.on_message
def on_message(msg):
    print(f"[{msg.severity}] {msg.text}")


window = bz.Window(W, H, "Bazalt Demo - CPU Image Streaming", logger=logger)
ctx = bz.Context(logger)
renderer = ctx.create_renderer(window)


def frame_at(t):
    """Whatever produces your pixels. Here: a moving plasma, in numpy.

    RGBA8 to match the image's format — update() checks the dtype and the shape
    and refuses a mismatch rather than uploading bytes that mean something else.
    """
    y, x = np.mgrid[0:TEX, 0:TEX].astype(np.float32) / TEX
    r = np.sin(6.0 * x + t) * 0.5 + 0.5
    g = np.sin(6.0 * y + t * 1.3) * 0.5 + 0.5
    b = np.sin(6.0 * (x + y) + t * 0.7) * 0.5 + 0.5

    out = np.empty((TEX, TEX, 4), dtype=np.uint8)
    out[:, :, 0] = r * 255
    out[:, :, 1] = g * 255
    out[:, :, 2] = b * 255
    out[:, :, 3] = 255
    # C-contiguous already. Had it come from a slice or a transpose, update()
    # would refuse it: memcpy ignores strides, so a strided array would upload
    # garbage instead of raising.
    return out


# The streamed texture. An ordinary image — what makes it a video texture is
# that something writes into it every frame.
stream = ctx.create_image(frame_at(0.0), name="stream")

vert = ctx.compile_shader("fullscreen.vert", bz.ShaderStage.VERTEX)
frag = ctx.compile_shader("present.frag", bz.ShaderStage.FRAGMENT)
present = (ctx.graphics_pipeline()
           .vertex_shader(vert)
           .fragment_shader(frag)
           .texture(0, bz.ShaderStage.FRAGMENT, set=0)
           .name("present")
           .build(renderer))

pool = ctx.create_descriptor_pool(max_sets=1, textures=1)
dset = pool.allocate_set(present, set=0)
# Bound ONCE. update() writes into the same VkImage, so no descriptor set is
# ever rewritten — that is the difference from creating a new image per frame.
dset.set_image(0, stream, sampler=ctx.create_sampler(filter=bz.Filter.LINEAR))

cmd = ctx.create_command_buffer()
cmd.begin()
with cmd.label("present streamed texture"):
    with cmd.rendering(renderer, clear_color=[0, 0, 0, 1]):
        cmd.bind_pipeline(present)
        cmd.bind_descriptor_set(dset, present, 0)
        cmd.draw(3)

print("drag with the left mouse button to paint; S saves a screenshot; Esc quits")

WHITE_PATCH = np.full((16, 16, 4), 255, dtype=np.uint8)

t = 0.0
frames = 0
capture_next = False
saved = False

while window.is_open():
    bz.poll_events()
    if window.is_key_pressed(bz.KEY_ESCAPE):
        break

    t += 1.0 / 60.0

    # The whole texture, every frame. Asynchronous: this returns while the copy
    # is still queued, and the submit below waits for it on the GPU.
    stream.update(frame_at(t))

    # A rectangle, only where the mouse is down. Same verb, one more argument —
    # painting and a sprite atlas are the same operation as a video frame.
    mouse = window.get_mouse_state()
    if window.is_mouse_button_pressed(bz.MOUSE_BUTTON_LEFT):
        x = int(mouse.x / W * TEX) - 8
        y = int(mouse.y / H * TEX) - 8
        x = max(0, min(TEX - 16, x))
        y = max(0, min(TEX - 16, y))
        stream.update(WHITE_PATCH, region=(x, y, 16, 16))

    if window.was_key_pressed(bz.KEY_S):
        capture_next = True

    ctx.begin_frame()
    if not renderer.acquire():
        continue
    renderer.present(cmd, capture=capture_next)

    if capture_next:
        shot = renderer.read_pixels()
        # RGBA8 whatever channel order the compositor picked, so shot[y, x, 0]
        # is red on every machine. Saving it is the ecosystem's job:
        #   from PIL import Image; Image.fromarray(shot).save("frame.png")
        print(f"captured {shot.shape[1]}x{shot.shape[0]}, "
              f"centre pixel {shot[shot.shape[0] // 2, shot.shape[1] // 2].tolist()}")
        capture_next = False
        saved = True

    frames += 1
    if frames % 120 == 0:
        stats = ctx.memory_stats()
        # Streaming into ONE image should hold this flat. A version that created
        # a new Image per frame would climb here until the deletion queue drained.
        print(f"{stats} after {frames} streamed frames")

print(f"streamed {frames} frames" + (", saved a screenshot" if saved else ""))
