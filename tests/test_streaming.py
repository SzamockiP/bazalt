"""Getting pixels and bytes in and out of a running program.

Before 0.18 the CPU/GPU boundary was one-way and one-shot: pixels went in when
an image was created and came out only from mip 0 of layer 0 of an offscreen
target. A video frame, a camera feed, a matplotlib figure, a procedural texture
or painting all meant a new Image every frame.
"""

import io
import struct

import numpy as np
import pytest

import bazalt as bz


def rgba(w, h, colour):
    a = np.zeros((h, w, 4), dtype=np.uint8)
    a[:, :, 0], a[:, :, 1], a[:, :, 2], a[:, :, 3] = colour
    return a


RED = (255, 0, 0, 255)
GREEN = (0, 255, 0, 255)
BLUE = (0, 0, 255, 255)


# ── image.update ────────────────────────────────────────────────────────────


def test_update_replaces_the_pixels(ctx):
    img = ctx.create_image(rgba(8, 8, BLUE))
    img.update(rgba(8, 8, RED))
    img.wait()

    assert img.read()[0, 0].tolist() == list(RED)


def test_update_is_visible_only_after_the_upload_lands(ctx):
    """The update goes through the same worker as load_image, so the image is
    not ready the instant update() returns. Without the pending mark a read
    right after an update silently returns the PREVIOUS contents, one update
    behind, every time — which is a wrong answer, not a slow one."""
    img = ctx.create_image(rgba(8, 8, BLUE))
    img.update(rgba(8, 8, RED))
    assert img.ready is False
    img.wait()
    assert img.ready is True
    assert img.read()[0, 0].tolist() == list(RED)


def test_update_region_leaves_the_rest_alone(ctx):
    """What painting and a sprite atlas need."""
    img = ctx.create_image(rgba(8, 8, RED))
    img.update(rgba(2, 2, GREEN), region=(0, 0, 2, 2))
    img.wait()

    out = img.read()
    assert out[0, 0].tolist() == list(GREEN)
    assert out[1, 1].tolist() == list(GREEN)
    assert out[7, 7].tolist() == list(RED), "the update escaped its region"
    assert out[0, 3].tolist() == list(RED)


def test_update_region_at_an_offset(ctx):
    img = ctx.create_image(rgba(8, 8, RED))
    img.update(rgba(2, 2, GREEN), region=(4, 4, 2, 2))
    img.wait()

    out = img.read()
    assert out[4, 4].tolist() == list(GREEN)
    assert out[0, 0].tolist() == list(RED)


def test_updates_of_one_image_land_in_call_order(ctx):
    """One FIFO worker, so the order the calls were made in is the order the GPU
    sees. That is a guarantee worth pinning: a video decoder that queued two
    frames must not show them backwards.

    Two things have to be true for it, and both were false. The submits have to
    be ORDERED against each other, because two on one queue may overlap unless
    one waits for the other. And wait() has to wait for every QUEUED update, not
    for the first one the worker happens to submit.

    Six updates is enough to state the promise and not enough to break it — this
    passed on a desktop driver for three releases. The deep-queue test below is
    the one that fails deterministically."""
    colors = [RED, GREEN, BLUE, RED, GREEN, BLUE]
    img = ctx.create_image(rgba(8, 8, BLUE))
    for color in colors:
        img.update(rgba(8, 8, color))
    img.wait()

    assert img.read()[0, 0].tolist() == list(colors[-1])


def test_wait_covers_updates_the_worker_has_not_reached(ctx):
    """The queue deeper than the worker can drain, which is what makes this fail
    on every machine rather than only on a driver that reorders.

    `upload_state_` used to be one flag, so the worker submitting the FIRST of
    these flipped it to Submitted and wait() returned with most of the queue
    untouched — read() then gave whichever frame had landed. Measured before the
    fix: update 71 of 200. It is a wrong answer and it scales with how far behind
    the worker is, which is the definition of a test that passes on a fast
    machine."""
    colors = [(i % 256, (i * 7) % 256, 200, 255) for i in range(200)]
    img = ctx.create_image(rgba(8, 8, BLUE))
    for color in colors:
        img.update(rgba(8, 8, color))
    img.wait()

    assert img.read()[0, 0].tolist() == list(colors[-1])


def test_update_one_layer_of_an_array(ctx):
    img = ctx.create_image([rgba(4, 4, RED), rgba(4, 4, RED), rgba(4, 4, RED)])
    img.update(rgba(4, 4, GREEN), layer=1)
    img.wait()

    assert img.read(layer=1)[0, 0].tolist() == list(GREEN)
    assert img.read(layer=0)[0, 0].tolist() == list(RED)
    assert img.read(layer=2)[0, 0].tolist() == list(RED)


def test_update_one_mip_level(ctx):
    img = ctx.create_image(rgba(16, 16, RED), mipmaps=True)
    img.update(rgba(8, 8, GREEN), mip=1)
    img.wait()

    assert img.read(mip=1)[0, 0].tolist() == list(GREEN)
    assert img.read(mip=0)[0, 0].tolist() == list(RED)


def test_update_rejects_a_non_contiguous_array(ctx):
    """memcpy ignores strides, so a transposed or sliced array would upload
    garbage. The rule create_image has followed since 0.4."""
    img = ctx.create_image(rgba(8, 8, RED))
    strided = rgba(8, 16, GREEN)[:, ::2]

    with pytest.raises(bz.ResourceError, match="ascontiguousarray"):
        img.update(strided)


def test_update_rejects_the_wrong_dtype(ctx):
    img = ctx.create_image(rgba(8, 8, RED))
    with pytest.raises(bz.ResourceError, match="dtype"):
        img.update(np.zeros((8, 8, 4), dtype=np.float32))


def test_update_rejects_the_wrong_shape(ctx):
    img = ctx.create_image(rgba(8, 8, RED))
    with pytest.raises(bz.ResourceError, match="shape"):
        img.update(rgba(4, 4, GREEN))


def test_update_rejects_a_region_outside_the_image(ctx):
    img = ctx.create_image(rgba(8, 8, RED))
    with pytest.raises(bz.ResourceError, match="does not fit"):
        img.update(rgba(4, 4, GREEN), region=(6, 6, 4, 4))


def test_update_rejects_a_layer_that_does_not_exist(ctx):
    img = ctx.create_image(rgba(8, 8, RED))
    with pytest.raises(bz.ResourceError, match="layer"):
        img.update(rgba(8, 8, GREEN), layer=3)


def test_an_updated_image_is_still_sampleable(ctx):
    """The subresource has to come back to SHADER_READ_ONLY, or the next draw
    that samples it is a validation error. The referee is the ctx fixture."""
    from conftest import SHADER_DIR

    img = ctx.create_image(rgba(8, 8, BLUE))
    img.update(rgba(8, 8, RED))
    img.wait()

    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "textured.frag"), bz.ShaderStage.FRAGMENT)
    screen = ctx.create_render_target(8, 8)
    pipe = (ctx.graphics_pipeline()
            .vertex_shader(vert)
            .fragment_shader(frag)
            .texture(0, bz.ShaderStage.FRAGMENT, set=0)
            .build(screen))
    pool = ctx.create_descriptor_pool(max_sets=1, textures=1)
    dset = pool.allocate_set(pipe, set=0)
    dset.set_image(0, img, sampler=ctx.create_sampler(filter=bz.Filter.NEAREST))

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(screen, clear_color=[0, 0, 0, 1]):
        cmd.bind_pipeline(pipe).bind_descriptor_set(dset, pipe, 0).draw(3)
    ctx.submit(cmd)

    assert screen.color[0].read()[4, 4, 0] > 200


# ── image.read(layer=, mip=) ────────────────────────────────────────────────


def test_read_a_specific_mip_has_that_mips_shape(ctx):
    img = ctx.create_image(rgba(16, 16, RED), mipmaps=True)
    assert img.read(mip=0).shape == (16, 16, 4)
    assert img.read(mip=1).shape == (8, 8, 4)
    assert img.read(mip=2).shape == (4, 4, 4)


def test_read_a_generated_mip_actually_contains_something(ctx):
    """The test generate_mipmaps never had. Reading only mip 0 meant a mip
    generator that did nothing at all would pass every test."""
    img = ctx.create_image(rgba(16, 16, RED), mipmaps=True)
    smallest = img.mip_levels - 1
    assert img.read(mip=smallest)[0, 0].tolist() == list(RED)


def test_read_a_cube_face(ctx):
    faces = [rgba(4, 4, c) for c in (RED, GREEN, BLUE, RED, GREEN, BLUE)]
    cube = ctx.create_image(faces, cube=True)

    assert cube.read(layer=0)[0, 0].tolist() == list(RED)
    assert cube.read(layer=1)[0, 0].tolist() == list(GREEN)
    assert cube.read(layer=2)[0, 0].tolist() == list(BLUE)


def test_read_rejects_a_mip_that_does_not_exist(ctx):
    img = ctx.create_image(rgba(8, 8, RED))
    with pytest.raises(bz.ResourceError, match="mip"):
        img.read(mip=4)


# ── buffer.update(offset=) ──────────────────────────────────────────────────


def test_buffer_update_at_an_offset(ctx):
    buf = ctx.create_buffer([1.0, 2.0, 3.0, 4.0], bz.BufferType.STORAGE,
                            bz.MemoryUsage.DYNAMIC, bz.DataType.FLOAT)
    buf.update(struct.pack("ff", 9.0, 9.0), offset=8)

    assert list(buf.read("float32")) == [1.0, 2.0, 9.0, 9.0]


def test_buffer_update_offset_rejects_an_overrun(ctx):
    buf = ctx.create_buffer([1.0, 2.0], bz.BufferType.STORAGE,
                            bz.MemoryUsage.DYNAMIC, bz.DataType.FLOAT)
    with pytest.raises(bz.ResourceError):
        buf.update(struct.pack("ff", 9.0, 9.0), offset=4)


# ── load_image(bytes) ───────────────────────────────────────────────────────


def png_bytes(w, h, colour):
    """A PNG with no file behind it, which is the whole point."""
    Image = pytest.importorskip("PIL.Image", reason="Pillow is needed to encode a PNG in memory")
    buf = io.BytesIO()
    Image.fromarray(rgba(w, h, colour)).save(buf, format="PNG")
    return buf.getvalue()


def test_load_image_from_bytes(ctx):
    img = ctx.load_image(png_bytes(8, 8, RED))
    img.wait()

    assert img.width == 8 and img.height == 8
    # sRGB on the file path, so the stored value is not the literal 255 — the
    # assertion is that the red channel dominates, which is what survives the
    # colour-space round trip.
    out = img.read()
    assert out[0, 0, 0] > out[0, 0, 1]


def test_load_image_from_bytes_rejects_garbage(ctx):
    with pytest.raises(bz.ResourceError, match="decodable"):
        ctx.load_image(b"this is not a png")


def test_bytes_are_not_mistaken_for_a_path(ctx):
    """pybind converts str AND bytes to std::string, so without the overload
    order this arrives at the path version and is reported as a missing file.
    The same trap compile_shader(source=) hit in 0.16."""
    with pytest.raises(bz.ResourceError) as excinfo:
        ctx.load_image(b"\x89PNG\r\n\x1a\n truncated")
    assert "Failed to load image" not in str(excinfo.value)
