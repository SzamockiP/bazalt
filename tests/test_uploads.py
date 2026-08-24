"""Async image uploads: load_image returns immediately, the GPU waits for it.

The image IS the future — there is no Future[Image] wrapper. A submit that
samples a pending image waits for exactly that upload (GPU-side, via the
submission timeline); everything else never blocks. `ready`/`wait()`/
`ctx.wait()` exist for explicit control, not correctness.
"""

import pathlib
import struct
import zlib

import numpy as np
import pytest

import bazalt as bz

from test_bindings import write_png

SHADER_DIR = pathlib.Path(__file__).parent / "shaders"


def write_corrupt_png(path):
    """A PNG whose header parses (stbi_info succeeds — so load_image returns
    an Image) but whose pixel data is missing (the decode fails on the worker).
    """
    def chunk(kind, payload):
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload)))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", b"\xff\xff\xff\xff\xff\xff\xff\xff")  # invalid zlib stream
           + chunk(b"IEND", b""))
    path.write_bytes(png)


def test_load_image_returns_immediately_with_correct_dimensions(ctx, tmp_path):
    png_path = tmp_path / "img.png"
    write_png(png_path, [[(255, 0, 0, 255)] * 64] * 32)

    img = ctx.load_image(str(png_path))
    # Width/height come from the synchronous header probe, before the decode.
    assert (img.width, img.height) == (64, 32)

    img.wait()
    assert img.ready
    np.testing.assert_array_equal(img.read()[0, 0], [255, 0, 0, 255])


def test_missing_file_raises_at_the_call_site(ctx):
    with pytest.raises(bz.ResourceError) as info:
        ctx.load_image("no_such_file.png")
    assert "no_such_file.png" in str(info.value)


def test_corrupt_file_fails_at_wait_and_logs(ctx, tmp_path, messages):
    png_path = tmp_path / "broken.png"
    write_corrupt_png(png_path)

    img = ctx.load_image(str(png_path))  # header is fine, so this succeeds
    with pytest.raises(bz.ResourceError) as info:
        img.wait()
    assert "broken.png" in str(info.value)
    assert not img.ready

    upload_errors = [m for m in messages()
                     if m.source == bz.Source.UPLOAD and m.severity >= bz.Severity.ERROR]
    assert upload_errors, "the failed decode should be logged with Source.UPLOAD"


def test_sampling_without_wait_renders_correctly(ctx, fullscreen_and_textured, tmp_path):
    """The residency contract: no explicit wait anywhere, correct pixels out."""
    red, green = (255, 0, 0, 255), (0, 255, 0, 255)
    blue, white = (0, 0, 255, 255), (255, 255, 255, 255)
    png_path = tmp_path / "quad.png"
    write_png(png_path, [[red, green], [blue, white]])

    tex = ctx.load_image(str(png_path))  # NOT waited on

    vert, frag = fullscreen_and_textured
    target = ctx.create_render_target(62, 62)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .texture(0, bz.ShaderStage.FRAGMENT, set=0)
                .build(target))

    pool = ctx.create_descriptor_pool(max_sets=4, textures=4)
    dset = pool.allocate_set(pipeline, set=0)
    dset.set_image(0, tex)

    g = ctx.graph()
    with g.add_pass(target, clear_color=[0, 0, 0, 1]) as p:
        p.bind_pipeline(pipeline).bind_descriptor_set(dset, pipeline, set=0).draw(3)
    ctx.submit(g)

    pixels = target.color[0].read()
    assert np.allclose(pixels[15, 15, :3], red[:3], atol=2), pixels[15, 15]
    assert np.allclose(pixels[46, 46, :3], white[:3], atol=2), pixels[46, 46]


def test_unrelated_submits_do_not_wait_for_uploads(ctx, triangle_shaders, triangle_buffers, tmp_path):
    """The loading-screen shape: draw something that does not reference the
    pending images while they stream in. Nothing here may deadlock or fail."""
    png_path = tmp_path / "big.png"
    write_png(png_path, [[(128, 128, 128, 255)] * 256] * 256)
    pending = [ctx.load_image(str(png_path)) for _ in range(4)]

    vert, frag = triangle_shaders
    vbuf, ibuf = triangle_buffers
    target = ctx.create_render_target(64, 64)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                .build(target))

    g = ctx.graph()
    with g.add_pass(target, clear_color=[0.1, 0.2, 0.3, 1.0]) as p:
        p.bind_pipeline(pipeline).bind_vertex_buffer(vbuf).bind_index_buffer(ibuf).draw_indexed(3)
    ctx.submit(g)
    assert target.color[0].read() is not None

    ctx.wait()
    assert all(img.ready for img in pending)


def test_wait_and_progress_endpoints(ctx, tmp_path):
    assert ctx.upload_progress == 1.0  # idle

    png_path = tmp_path / "img.png"
    write_png(png_path, [[(1, 2, 3, 255)] * 32] * 32)
    imgs = [ctx.load_image(str(png_path)) for _ in range(8)]

    ctx.wait()
    assert ctx.upload_progress == 1.0
    assert all(img.ready for img in imgs)
    np.testing.assert_array_equal(imgs[-1].read()[0, 0], [1, 2, 3, 255])


# ── one-shot uploads: no decode, no worker, no wait (0.18.0) ──────────────
#
# create_buffer and create_image(array) hand over bytes that are already
# decoded, so there is nothing to move onto the upload worker. They submit the
# staging copy on the calling thread and skip only the wait — which makes the
# resource its own future exactly like a load_image one.


def test_static_buffer_upload_is_async_and_wait_settles_it(ctx):
    data = np.arange(1024, dtype=np.float32)
    buf = ctx.create_buffer(data, bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    buf.wait()
    assert buf.ready


def test_dynamic_buffers_are_never_pending(ctx):
    """Host-visible memory is written by mapping, so there is no copy to wait
    for and `ready` is the honest constant it looks like."""
    buf = ctx.create_buffer(np.zeros(16, dtype=np.float32), bz.BufferUsage.UNIFORM,
                            bz.MemoryUsage.DYNAMIC)
    assert buf.ready


def test_reading_a_fresh_static_buffer_needs_no_wait(ctx):
    """read() is a blocking round trip either way, so it waits for the fill
    itself. Without that wait this is a race that returns uninitialized memory
    on any driver that overlaps two submits."""
    data = np.arange(256, dtype=np.float32)
    buf = ctx.create_buffer(data, bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    np.testing.assert_array_equal(buf.read(np.float32), data)


def test_drawing_from_a_fresh_buffer_needs_no_wait(ctx, triangle_shaders):
    """The residency contract for buffers: created and bound with no wait
    anywhere, and the draw still sees the vertices."""
    vertices = np.array([
        +0.0, -0.5, 0.0, 1.0, 0.0, 0.0,
        -0.5, +0.5, 0.0, 1.0, 0.0, 0.0,
        +0.5, +0.5, 0.0, 1.0, 0.0, 0.0,
    ], dtype=np.float32)
    vbuf = ctx.create_buffer(vertices, bz.BufferUsage.VERTEX, bz.MemoryUsage.STATIC)
    ibuf = ctx.create_buffer(np.array([0, 1, 2], dtype=np.uint32), bz.BufferUsage.INDEX,
                             bz.MemoryUsage.STATIC)

    vert, frag = triangle_shaders
    target = ctx.create_render_target(64, 64)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                .build(target))

    g = ctx.graph()
    with g.add_pass(target, clear_color=[0, 0, 0, 1]) as p:
        p.bind_pipeline(pipeline).bind_vertex_buffer(vbuf).bind_index_buffer(ibuf).draw_indexed(3)
    ctx.submit(g)

    pixels = target.color[0].read()
    assert pixels[32, 32, 0] > 200, pixels[32, 32]


def test_wait_covers_one_shot_uploads(ctx):
    """They have no decode stage, so they join the batch already submitted
    rather than being tracked beside it. ctx.wait() and upload_progress would
    both be a lie otherwise."""
    ctx.wait()

    buf = ctx.create_buffer(np.zeros(4096, dtype=np.float32), bz.BufferUsage.STORAGE,
                            bz.MemoryUsage.STATIC)
    img = ctx.create_image(np.zeros((256, 256, 4), dtype=np.uint8))

    ctx.wait()
    assert ctx.upload_progress == 1.0
    assert buf.ready
    assert img.ready


@pytest.fixture
def fullscreen_and_textured(ctx):
    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "textured.frag"), bz.ShaderStage.FRAGMENT)
    return vert, frag


def test_a_mipped_upload_is_sampled_after_the_split(ctx, tmp_path):
    """A mipped upload is TWO submits since 0.30 — the copy on the transfer
    queue, the blit cascade on graphics waiting for it — and the image carries
    a serial on each timeline. Reading a generated level proves the cascade
    ran and the split stayed ordered; the validation fixture is the referee
    for the layouts."""
    png_path = tmp_path / "mipped.png"
    write_png(png_path, [[(64, 128, 192, 255)] * 64] * 64)
    img = ctx.load_image(str(png_path), mipmaps=True)
    assert img.mip_levels >= 3
    img.wait()
    assert img.ready
    top = img.read(mip=0)
    low = img.read(mip=2)
    # A constant image box-filters to itself, so every level holds the colour.
    assert tuple(top[0, 0]) == (64, 128, 192, 255)
    assert tuple(low[0, 0]) == (64, 128, 192, 255)


def test_uploads_ride_the_transfer_timeline(ctx):
    """create_buffer's staging copy submits on the transfer runtime since
    0.30, so a graph that binds the buffer waits that timeline through
    require_uploads_resident. Validation-clean is the whole assertion; a
    missing wait is a hazard the fixture reports."""
    buf = ctx.create_buffer(np.arange(4096, dtype=np.float32),
                            bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    out = ctx.create_buffer(4096 * 4, bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    g = ctx.graph()
    g.add_pass(name="copy").copy_buffer(buf, out)
    ctx.submit(g)
    assert out.read(np.float32)[-1] == 4095.0
    assert buf.ready


def test_a_copy_verb_waits_for_an_upload_it_never_bound():
    """An image a copy or a blit names directly is in no descriptor set, so
    `used_sets` never sees it.

    Until 0.30 that cost nothing: the upload submitted on the graphics queue,
    the graph replayed on the graphics queue, and a pipeline barrier's first
    scope covers everything submitted earlier there. The upload runs on the
    TRANSFER queue now and a barrier cannot reach it, so the image has to be
    named for the submit's timeline wait — the job used_buffers has done for a
    staged buffer since 0.18.

    The referee is sync validation plus the pixels. The image is large enough
    that the copy really can start before the upload lands: the bug this pins
    passed the whole suite once and was found by a run that shifted the
    timing."""
    hazards = []
    log = bz.Logger(min_severity=bz.Severity.INFO)

    @log.on_message
    def _(msg):
        if msg.source == bz.Source.VALIDATION and "hazard" in msg.text.lower():
            hazards.append(msg.text)

    context = bz.Context(log, validation="sync")
    pixels = np.zeros((512, 512, 4), np.uint8)
    pixels[:, :, 1] = 200
    pixels[:, :, 3] = 255

    src = context.create_image(pixels, name="uploaded")
    dst = context.create_image(512, 512, bz.Format.RGBA8, name="copy of it")
    g = context.graph()
    g.add_pass(name="copy").copy_image(src, dst)
    context.submit(g)

    out = dst.read()
    log.flush()
    assert hazards == []
    assert out[256, 256, 1] == 200, "the copy read the image before its upload landed"
