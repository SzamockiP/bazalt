"""Behaviour-pinning tests for bindings no example or test ever exercised.

An unexercised binding is an unimplemented binding: push_constants, STORAGE
buffers, blend, plain draw(), DescriptorSet and load_texture all shipped in
0.4.0 with zero coverage. These tests pin their behaviour down *before* the
0.4.1 refactoring touches the code underneath them.

Everything runs headless against the session Context; the `ctx` fixture fails
any test that provokes a validation error.
"""

import pathlib
import struct
import zlib

import numpy as np
import pytest

import bazalt as bz

SHADER_DIR = pathlib.Path(__file__).parent / "shaders"

CLEAR = [0.1, 0.2, 0.3, 1.0]
CLEAR_RGB = np.array([26, 51, 77])


@pytest.fixture
def fullscreen_vert(ctx):
    return ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)


def submit_and_read(ctx, target, record):
    """Record commands via `record(cmd)`, submit, and read the target back."""
    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(target, clear_color=CLEAR)
    record(cmd)
    cmd.end_rendering(target)
    ctx.submit(cmd)
    return target.read_pixels()


# ── draw() — non-indexed ──────────────────────────────────────────────────


def test_plain_draw_renders(ctx, triangle_shaders, triangle_buffers):
    """draw(3) straight from the vertex buffer, no index buffer involved."""
    vert, frag = triangle_shaders
    vbuf, _ = triangle_buffers
    target = bz.RenderTarget(ctx, 64, 64)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                .build(target))

    def record(cmd):
        cmd.bind_pipeline(pipeline)
        cmd.bind_vertex_buffer(vbuf)
        cmd.draw(3)

    pixels = submit_and_read(ctx, target, record)
    assert not np.allclose(pixels[32, 32, :3], CLEAR_RGB, atol=2), \
        "centre still shows the clear colour"


# ── push constants ────────────────────────────────────────────────────────


def test_push_constants_reach_the_shader(ctx, fullscreen_vert):
    frag = ctx.compile_shader(str(SHADER_DIR / "push.frag"), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, 64, 64)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(fullscreen_vert)
                .fragment_shader(frag)
                .push_constant(16, bz.ShaderStage.FRAGMENT)
                .build(target))

    def record(cmd):
        cmd.bind_pipeline(pipeline)
        cmd.push_constants(pipeline, 0, struct.pack("4f", 1.0, 0.5, 0.0, 1.0))
        cmd.draw(3)

    pixels = submit_and_read(ctx, target, record)
    assert np.allclose(pixels[32, 32, :3], [255, 128, 0], atol=2), pixels[32, 32]


# ── blending ──────────────────────────────────────────────────────────────


def test_blend_composites_two_draws(ctx, fullscreen_vert):
    """Two translucent fullscreen passes over the clear colour.

    UNORM target, so the arithmetic is exact up to rounding:
    black cleared to (0,0,0), then 0.5 red -> (0.5,0,0),
    then 0.5 blue over that -> (0.25, 0, 0.5).
    """
    frag = ctx.compile_shader(str(SHADER_DIR / "push.frag"), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, 64, 64)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(fullscreen_vert)
                .fragment_shader(frag)
                .push_constant(16, bz.ShaderStage.FRAGMENT)
                .blend(True)
                .build(target))

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0])
    cmd.bind_pipeline(pipeline)
    cmd.push_constants(pipeline, 0, struct.pack("4f", 1.0, 0.0, 0.0, 0.5))
    cmd.draw(3)
    cmd.push_constants(pipeline, 0, struct.pack("4f", 0.0, 0.0, 1.0, 0.5))
    cmd.draw(3)
    cmd.end_rendering(target)
    ctx.submit(cmd)

    pixels = target.read_pixels()
    assert np.allclose(pixels[32, 32, :3], [64, 0, 128], atol=3), pixels[32, 32]


# ── descriptor sets: uniform buffer end-to-end ────────────────────────────


def test_uniform_buffer_via_frame_descriptor_set(ctx, fullscreen_vert):
    """The full descriptor path: pool -> frame set -> DYNAMIC uniform -> shader.

    Also pins DynamicBuffer.update: the second submit must see the new colour
    through the same descriptor set.
    """
    frag = ctx.compile_shader(str(SHADER_DIR / "ubo.frag"), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, 64, 64)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(fullscreen_vert)
                .fragment_shader(frag)
                .uniform_buffer(0, bz.ShaderStage.FRAGMENT, set=0)
                .build(target))

    ubuf = ctx.create_buffer([0.0, 1.0, 0.0, 1.0], bz.BufferType.UNIFORM,
                             bz.MemoryUsage.DYNAMIC)
    pool = ctx.create_descriptor_pool(max_sets=8, uniform_buffers=8)
    dset = pool.allocate_frame_set(pipeline, set=0)
    dset.set_buffer(0, ubuf)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(target, clear_color=CLEAR)
    cmd.bind_pipeline(pipeline)
    cmd.bind_descriptor_set(dset, pipeline, set=0)
    cmd.draw(3)
    cmd.end_rendering(target)

    ctx.submit(cmd)
    assert np.allclose(target.read_pixels()[32, 32, :3], [0, 255, 0], atol=2)

    ubuf.update([1.0, 0.0, 1.0, 1.0])
    ctx.submit(cmd)
    assert np.allclose(target.read_pixels()[32, 32, :3], [255, 0, 255], atol=2)


# ── storage buffers ───────────────────────────────────────────────────────


@pytest.mark.parametrize("usage", [bz.MemoryUsage.STATIC, bz.MemoryUsage.DYNAMIC])
def test_storage_buffer_read_in_fragment_shader(ctx, fullscreen_vert, usage):
    frag = ctx.compile_shader(str(SHADER_DIR / "ssbo.frag"), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, 64, 64)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(fullscreen_vert)
                .fragment_shader(frag)
                .storage_buffer(0, bz.ShaderStage.FRAGMENT, set=0)
                .build(target))

    sbuf = ctx.create_buffer([0.0, 0.0, 1.0, 1.0], bz.BufferType.STORAGE, usage)
    pool = ctx.create_descriptor_pool(max_sets=8, storage_buffers=8)
    if usage == bz.MemoryUsage.DYNAMIC:
        dset = pool.allocate_frame_set(pipeline, set=0)
    else:
        dset = pool.allocate_set(pipeline, set=0)
    dset.set_buffer(0, sbuf)

    def record(cmd):
        cmd.bind_pipeline(pipeline)
        cmd.bind_descriptor_set(dset, pipeline, set=0)
        cmd.draw(3)

    pixels = submit_and_read(ctx, target, record)
    assert np.allclose(pixels[32, 32, :3], [0, 0, 255], atol=2), pixels[32, 32]


# ── textures ──────────────────────────────────────────────────────────────


def write_png(path, rows):
    """Minimal RGBA PNG writer (stdlib only — no Pillow dependency).

    `rows` is a list of rows, each a list of (r, g, b, a) tuples.
    """
    height = len(rows)
    width = len(rows[0])

    def chunk(kind, payload):
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload)))

    raw = b"".join(b"\x00" + bytes(v for px in row for v in px) for row in rows)
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw))
           + chunk(b"IEND", b""))
    path.write_bytes(png)


def test_texture_is_sampled(ctx, fullscreen_vert, tmp_path):
    """A 2x2 texture stretched over the target: each quadrant shows its texel.

    Pure 0/255 channels survive the sRGB texture -> UNORM target round trip
    exactly, so no colour-space tolerance is needed.
    """
    red, green, blue, white = (255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255), (255, 255, 255, 255)
    png_path = tmp_path / "quad.png"
    write_png(png_path, [[red, green],
                         [blue, white]])

    tex = ctx.load_image(str(png_path))
    assert (tex.width, tex.height) == (2, 2)
    # 2x2 -> a 2-level mip chain, generated automatically by load_image.
    assert tex.mip_levels == 2
    assert tex.format == bz.Format.RGBA8_SRGB

    frag = ctx.compile_shader(str(SHADER_DIR / "textured.frag"), bz.ShaderStage.FRAGMENT)
    # 62, not 64: pixel centres (15.5/62, 46.5/62) land exactly on uv 0.25/0.75,
    # i.e. on texel centres, where linear filtering contributes nothing.
    target = bz.RenderTarget(ctx, 62, 62)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(fullscreen_vert)
                .fragment_shader(frag)
                .texture(0, bz.ShaderStage.FRAGMENT, set=0)
                .build(target))

    pool = ctx.create_descriptor_pool(max_sets=8, samplers=8)
    dset = pool.allocate_set(pipeline, set=0)
    dset.set_image(0, tex)

    def record(cmd):
        cmd.bind_pipeline(pipeline)
        cmd.bind_descriptor_set(dset, pipeline, set=0)
        cmd.draw(3)

    pixels = submit_and_read(ctx, target, record)

    # PNG row 0 is the top of the image and uv (0,0) is the top-left of the
    # target, so the quadrants map straight through.
    assert np.allclose(pixels[15, 15, :3], red[:3], atol=2), pixels[15, 15]
    assert np.allclose(pixels[15, 46, :3], green[:3], atol=2), pixels[15, 46]
    assert np.allclose(pixels[46, 15, :3], blue[:3], atol=2), pixels[46, 15]
    assert np.allclose(pixels[46, 46, :3], white[:3], atol=2), pixels[46, 46]


def test_set_image_on_buffer_binding_points_to_set_buffer(ctx, fullscreen_vert, tmp_path):
    """Descriptor writes are validated against the layout at the call site,
    not left for the validation layers at submit time."""
    png_path = tmp_path / "px.png"
    write_png(png_path, [[(255, 0, 0, 255)]])
    tex = ctx.load_image(str(png_path))

    frag = ctx.compile_shader(str(SHADER_DIR / "ubo.frag"), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, 16, 16)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(fullscreen_vert)
                .fragment_shader(frag)
                .uniform_buffer(0, bz.ShaderStage.FRAGMENT, set=0)
                .build(target))
    pool = ctx.create_descriptor_pool(max_sets=4, uniform_buffers=4)
    dset = pool.allocate_set(pipeline, set=0)

    with pytest.raises(bz.ResourceError) as info:
        dset.set_image(0, tex)
    assert "set_buffer" in str(info.value)

    with pytest.raises(bz.ResourceError) as info:
        dset.set_image(7, tex)
    assert "7" in str(info.value)
