"""Configurable render targets: formats, MRT, depth-only, render-to-texture.

The attachments are ordinary bz.Image objects — `target.color[0]` and
`target.depth` go straight into set_image() with no extra API. Depth on an
offscreen target ends every pass in SHADER_READ_ONLY, which is the entire
shadow-map story.
"""

import pathlib

import numpy as np
import pytest

import bazalt as bz

SHADER_DIR = pathlib.Path(__file__).parent / "shaders"

CLEAR = [0.1, 0.2, 0.3, 1.0]


# ── constructor contract ──────────────────────────────────────────────────


def test_no_attachments_is_refused(ctx):
    with pytest.raises(bz.ResourceError):
        ctx.create_render_target(16, 16, color=None, depth=None)


def test_bool_depth_gets_a_migration_hint(ctx):
    """depth was a bool in 0.4; the error must say what to write instead."""
    with pytest.raises(bz.ResourceError) as info:
        ctx.create_render_target(16, 16, depth=True)
    assert "D32F" in str(info.value)


def test_depth_format_in_color_slot_is_refused(ctx):
    with pytest.raises(bz.ResourceError) as info:
        ctx.create_render_target(16, 16, color=bz.Format.D32F)
    assert "depth" in str(info.value)


def test_color_format_in_depth_slot_is_refused(ctx):
    with pytest.raises(bz.ResourceError) as info:
        ctx.create_render_target(16, 16, depth=bz.Format.RGBA8)
    assert "D32F" in str(info.value)


def test_attachments_are_images(ctx):
    target = ctx.create_render_target(16, 16, color=bz.Format.RGBA16F, depth=bz.Format.D32F)
    assert len(target.color) == 1
    assert target.color[0].format == bz.Format.RGBA16F
    assert target.depth.format == bz.Format.D32F
    assert ctx.create_render_target(16, 16).depth is None


# ── depth-only target: the shadow-map shape ───────────────────────────────


def test_depth_only_pass_writes_sampleable_depth(ctx, triangle_shaders, triangle_buffers):
    """Rasterize into a depth-only target (no fragment shader), then consume
    the result twice: read back directly, and sample it in a second pass."""
    vert, _ = triangle_shaders
    vbuf, ibuf = triangle_buffers

    shadow = ctx.create_render_target(64, 64, color=None, depth=bz.Format.D32F)

    # No fragment shader: legal exactly because the target has no colour.
    depth_pipe = (ctx.graphics_pipeline()
                  .vertex_shader(vert)
                  .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                  .depth_test(True)
                  .build(shadow))

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(shadow)
    cmd.bind_pipeline(depth_pipe)
    cmd.bind_vertex_buffer(vbuf)
    cmd.bind_index_buffer(ibuf)
    cmd.draw_indexed(3)
    cmd.end_rendering(shadow)
    ctx.submit(cmd)

    # Direct readback: (h, w) float32, cleared to 1.0, triangle at z=0.
    depth = shadow.depth.read()
    assert depth.shape == (64, 64)
    assert depth.dtype == np.float32
    assert depth[2, 2] == pytest.approx(1.0), "corner should be the clear depth"
    assert depth[40, 32] == pytest.approx(0.0, abs=1e-5), "triangle interior should be at z=0"

    # Sampled in a second pass: shadow.depth is just an Image.
    fullscreen = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    view_frag = ctx.compile_shader(str(SHADER_DIR / "depth_view.frag"), bz.ShaderStage.FRAGMENT)
    screen = ctx.create_render_target(64, 64)
    view_pipe = (ctx.graphics_pipeline()
                 .vertex_shader(fullscreen)
                 .fragment_shader(view_frag)
                 .texture(0, bz.ShaderStage.FRAGMENT, set=0)
                 .build(screen))

    pool = ctx.create_descriptor_pool(max_sets=4, textures=4)
    dset = pool.allocate_set(view_pipe, set=0)
    # NEAREST: linear filtering of depth formats is not universally supported.
    dset.set_image(0, shadow.depth, sampler=ctx.create_sampler(filter=bz.Filter.NEAREST))

    cmd2 = ctx.create_command_buffer()
    cmd2.begin()
    cmd2.begin_rendering(screen, clear_color=[0, 0, 0, 1])
    cmd2.bind_pipeline(view_pipe)
    cmd2.bind_descriptor_set(dset, view_pipe, set=0)
    cmd2.draw(3)
    cmd2.end_rendering(screen)
    ctx.submit(cmd2)

    pixels = screen.color[0].read()
    assert pixels[2, 2, 0] == 255, "far depth should view as white"
    assert pixels[40, 32, 0] == 0, "triangle depth should view as black"


def test_depth_only_target_has_no_colour_to_read(ctx):
    """A depth-only target exposes an empty color tuple, so the mistake is an
    IndexError at the subscript rather than a read of the wrong attachment.
    target.depth is the one to read."""
    target = ctx.create_render_target(16, 16, color=None, depth=bz.Format.D32F)
    assert target.color == ()
    with pytest.raises(IndexError):
        target.color[0].read()
    assert target.depth is not None


# ── MRT ───────────────────────────────────────────────────────────────────


def test_mrt_renders_into_both_attachments(ctx):
    """One pass, two colour attachments with different formats; each read()
    comes back in its own dtype with its own contents."""
    gbuf = ctx.create_render_target(32, 32, color=[bz.Format.RGBA16F, bz.Format.RGBA8])

    fullscreen = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    mrt_frag = ctx.compile_shader(str(SHADER_DIR / "mrt.frag"), bz.ShaderStage.FRAGMENT)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(fullscreen)
                .fragment_shader(mrt_frag)
                .build(gbuf))

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(gbuf, clear_color=[0, 0, 0, 0])
    cmd.bind_pipeline(pipeline)
    cmd.draw(3)
    cmd.end_rendering(gbuf)
    ctx.submit(cmd)

    a = gbuf.color[0].read()
    assert a.dtype == np.float16
    assert a.shape == (32, 32, 4)
    assert np.allclose(a[16, 16], [0.25, 0.5, 0.75, 1.0], atol=1e-3)

    b = gbuf.color[1].read()
    assert b.dtype == np.uint8
    assert b.shape == (32, 32, 4)
    assert np.array_equal(b[16, 16], [255, 0, 255, 255])


def test_missing_fragment_shader_with_color_attachments_is_refused(ctx, triangle_shaders):
    vert, _ = triangle_shaders
    target = ctx.create_render_target(16, 16)
    with pytest.raises(bz.ShaderError):
        (ctx.graphics_pipeline()
         .vertex_shader(vert)
         .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
         .build(target))


# ── render-to-texture ─────────────────────────────────────────────────────


def test_color_attachment_samples_as_a_texture(ctx, triangle_shaders, triangle_buffers):
    """First pass renders the triangle offscreen; second pass samples that
    attachment fullscreen. RTT is not an API — it is target.color[0]."""
    vert, frag = triangle_shaders
    vbuf, ibuf = triangle_buffers

    first = ctx.create_render_target(64, 64)
    pipe1 = (ctx.graphics_pipeline()
             .vertex_shader(vert)
             .fragment_shader(frag)
             .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
             .build(first))

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(first, clear_color=CLEAR)
    cmd.bind_pipeline(pipe1)
    cmd.bind_vertex_buffer(vbuf)
    cmd.bind_index_buffer(ibuf)
    cmd.draw_indexed(3)
    cmd.end_rendering(first)
    ctx.submit(cmd)

    fullscreen = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    tex_frag = ctx.compile_shader(str(SHADER_DIR / "textured.frag"), bz.ShaderStage.FRAGMENT)
    second = ctx.create_render_target(64, 64)
    pipe2 = (ctx.graphics_pipeline()
             .vertex_shader(fullscreen)
             .fragment_shader(tex_frag)
             .texture(0, bz.ShaderStage.FRAGMENT, set=0)
             .build(second))

    pool = ctx.create_descriptor_pool(max_sets=4, textures=4)
    dset = pool.allocate_set(pipe2, set=0)
    dset.set_image(0, first.color[0])

    cmd2 = ctx.create_command_buffer()
    cmd2.begin()
    cmd2.begin_rendering(second, clear_color=[0, 0, 0, 1])
    cmd2.bind_pipeline(pipe2)
    cmd2.bind_descriptor_set(dset, pipe2, set=0)
    cmd2.draw(3)
    cmd2.end_rendering(second)
    ctx.submit(cmd2)

    np.testing.assert_array_equal(second.color[0].read(), first.color[0].read())


# ── the factory and the named views (0.23) ────────────────────────────────


def test_create_render_target_is_the_only_way_to_make_one(ctx):
    """The factory exists because every other resource comes from a create_*
    verb on the Context, and the constructor is gone rather than kept beside
    it — a second spelling of one call is a fork (the 0.18 audit).

    The refusal half lives in test_stubs.py, which runs on every wheel with no
    driver. This is the half that needs a device: the factory returns the class
    the stub and every annotation name, and the thing it returns draws."""
    made = ctx.create_render_target(8, 8)
    assert isinstance(made, bz.RenderTarget)
    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(made, clear_color=[1, 0, 0, 1])
    cmd.end_rendering(made)
    ctx.submit(cmd)
    assert np.all(made.color[0].read()[:, :, 0] == 255)


def test_create_render_target_takes_borrowed_images(ctx):
    mine = ctx.create_image(8, 8, bz.Format.RGBA8)
    target = ctx.create_render_target(color=[mine])
    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(target, clear_color=[0, 1, 0, 1])
    cmd.end_rendering(target)
    ctx.submit(cmd)
    assert np.all(mine.read()[:, :, 1] == 255)


def test_the_views_come_back_as_named_types(ctx):
    """layer() and all_layers() used to return opaque RenderTargetBase, so the
    stub could say nothing about them. pybind downcasts a polymorphic base to
    the registered derived type — registration alone is the feature."""
    target = ctx.create_render_target(8, 8, layers=2)
    assert type(target.layer(0)).__name__ == "SubresourceTarget"
    if ctx.supports(bz.Feature.MULTIVIEW):
        assert type(target.all_layers()).__name__ == "MultiviewTarget"
    assert isinstance(target.layer(1), bz.RenderTargetBase)
