"""Async image uploads: load_image returns immediately, the GPU waits for it.

The image IS the future — there is no Future[Image] wrapper. A submit that
samples a pending image waits for exactly that upload (GPU-side, via the
submission timeline); everything else never blocks. `ready`/`wait()`/
`wait_for_uploads()` exist for explicit control, not correctness.
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
    target = bz.RenderTarget(ctx, 62, 62)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .texture(0, bz.ShaderStage.FRAGMENT, set=0)
                .build(target))

    pool = ctx.create_descriptor_pool(max_sets=4, samplers=4)
    dset = pool.allocate_set(pipeline, set=0)
    dset.set_image(0, tex)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(target, clear_color=[0, 0, 0, 1]) as c:
        c.bind_pipeline(pipeline).bind_descriptor_set(dset, pipeline, set=0).draw(3)
    ctx.submit(cmd)

    pixels = target.read_pixels()
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
    target = bz.RenderTarget(ctx, 64, 64)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                .build(target))

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(target, clear_color=[0.1, 0.2, 0.3, 1.0]) as c:
        c.bind_pipeline(pipeline).bind_vertex_buffer(vbuf).bind_index_buffer(ibuf).draw_indexed(3)
    ctx.submit(cmd)
    assert target.read_pixels() is not None

    ctx.wait_for_uploads()
    assert all(img.ready for img in pending)


def test_wait_for_uploads_and_progress_endpoints(ctx, tmp_path):
    assert ctx.uploads_done  # idle
    assert ctx.upload_progress == 1.0

    png_path = tmp_path / "img.png"
    write_png(png_path, [[(1, 2, 3, 255)] * 32] * 32)
    imgs = [ctx.load_image(str(png_path)) for _ in range(8)]

    ctx.wait_for_uploads()
    assert ctx.uploads_done
    assert ctx.upload_progress == 1.0
    assert all(img.ready for img in imgs)
    np.testing.assert_array_equal(imgs[-1].read()[0, 0], [1, 2, 3, 255])


@pytest.fixture
def fullscreen_and_textured(ctx):
    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "textured.frag"), bz.ShaderStage.FRAGMENT)
    return vert, frag
