"""MSAA: hardware multisampling with resolve, plus its riders.

The whole feature lives on the RenderTarget: samples=N allocates a multisampled
image that is resolved into the single-sample attachments target.color/target.depth
expose. A pipeline built against that target picks up the sample count automatically
(no separate knob). The proof MSAA actually happened is edge pixels: a solid white
triangle on black has only pure black/white pixels at samples=1 (hard edges) but
grows partial-coverage greys along its silhouette at samples>1.

sample_shading's positive path needs the SAMPLE_RATE_SHADING feature, which the
session Context does not enable — only its rejection is covered here; the enabled
path is exercised live in examples/15_msaa.
"""

import pathlib

import numpy as np
import pytest

import bazalt as bz

SHADER_DIR = pathlib.Path(__file__).parent / "shaders"


@pytest.fixture
def samples(ctx):
    """A valid MSAA count >= 2, or skip on a GPU that can't multisample."""
    s = ctx.max_samples()
    if s < 2:
        pytest.skip("GPU reports no MSAA support (max_samples == 1)")
    return min(4, s)


@pytest.fixture
def white_triangle(ctx):
    """The centre triangle from triangle_buffers, but solid white — so the only
    non-black/non-white pixels in the output are MSAA edge coverage."""
    vertices = [
        +0.0, -0.5, 0.0, 1.0, 1.0, 1.0,
        -0.5, +0.5, 0.0, 1.0, 1.0, 1.0,
        +0.5, +0.5, 0.0, 1.0, 1.0, 1.0,
    ]
    vbuf = ctx.create_buffer(vertices, bz.BufferType.VERTEX, bz.MemoryUsage.STATIC, bz.DataType.FLOAT)
    ibuf = ctx.create_buffer([0, 1, 2], bz.BufferType.INDEX, bz.MemoryUsage.STATIC, bz.DataType.UINT32)
    return vbuf, ibuf


def _render_white_triangle(ctx, triangle_shaders, white_triangle, samples, depth=None, keep_samples=False):
    """Render the solid-white triangle on black into a fresh target, return the
    resolved pixels (uint8 HxWx4). Optionally attach depth."""
    vert, frag = triangle_shaders
    vbuf, ibuf = white_triangle
    target = ctx.create_render_target(
        64, 64, color=bz.Format.RGBA8, depth=depth, samples=samples, keep_samples=keep_samples)
    builder = (ctx.graphics_pipeline()
               .vertex_shader(vert)
               .fragment_shader(frag)
               .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3]))
    if depth is not None:
        builder = builder.depth_test(True)
    pipe = builder.build(target)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(target, clear_color=[0, 0, 0, 1])
    cmd.bind_pipeline(pipe)
    cmd.bind_vertex_buffer(vbuf)
    cmd.bind_index_buffer(ibuf)
    cmd.draw_indexed(3)
    cmd.end_rendering(target)
    ctx.submit(cmd)
    return target


def _grey_edge_pixels(pixels):
    """Count pixels that are neither near-black nor near-white — i.e. partial
    coverage, which only MSAA resolve produces on a solid triangle."""
    r = pixels[:, :, 0].astype(np.int32)
    return int(np.count_nonzero((r > 20) & (r < 235)))


# ── max_samples ────────────────────────────────────────────────────────────


def test_max_samples_is_a_power_of_two(ctx):
    s = ctx.max_samples()
    assert s >= 1
    assert (s & (s - 1)) == 0, f"max_samples()={s} is not a power of two"


# ── the resolve actually happens ─────────────────────────────────────────────


def test_msaa_resolves_smooth_edges(ctx, triangle_shaders, white_triangle, samples):
    """A solid white triangle: samples=1 has hard black/white edges, samples>1
    grows grey partial-coverage pixels along the silhouette."""
    aa1 = _render_white_triangle(ctx, triangle_shaders, white_triangle, samples=1).color[0].read()
    aaN = _render_white_triangle(ctx, triangle_shaders, white_triangle, samples=samples).color[0].read()

    grey1 = _grey_edge_pixels(aa1)
    greyN = _grey_edge_pixels(aaN)

    assert grey1 == 0, "a solid triangle at samples=1 must have no partial-coverage pixels"
    assert greyN > 0, "MSAA must produce partial-coverage edge pixels"
    # Interior and far corners are unchanged — only the silhouette softens.
    assert list(aaN[32, 32]) == [255, 255, 255, 255], "triangle interior stays solid white"
    assert list(aaN[2, 2]) == [0, 0, 0, 255], "a corner stays the clear colour"


def test_msaa_resolved_image_is_readable(ctx, triangle_shaders, white_triangle, samples):
    """target.color[i] is the single-sample resolve image — an ordinary readable
    Image, samples reported as 1, not the multisampled one."""
    target = _render_white_triangle(ctx, triangle_shaders, white_triangle, samples=samples)
    assert target.color[0].samples == 1
    px = target.color[0].read()
    assert px.shape == (64, 64, 4)
    assert px.dtype == np.uint8


# ── the multisampled attachment itself (custom resolve) ──────────────────────


def test_multisampled_attachments_need_keep_samples(ctx, triangle_shaders, white_triangle, samples):
    """target.multisampled_color / .multisampled_depth are the images rendered
    into; target.color / .depth stay the resolve. They only appear with
    keep_samples=True, because without it the pass discards them."""
    plain = _render_white_triangle(ctx, triangle_shaders, white_triangle, samples=1, keep_samples=True)
    assert plain.multisampled_color == (), "no MSAA means nothing to keep"
    assert plain.multisampled_depth is None

    discarded = _render_white_triangle(ctx, triangle_shaders, white_triangle, samples=samples)
    assert discarded.multisampled_color == (), "the samples were discarded, so there is nothing to hand out"
    assert discarded.multisampled_depth is None

    target = _render_white_triangle(
        ctx, triangle_shaders, white_triangle, samples=samples, depth=bz.Format.D32F, keep_samples=True)
    assert len(target.multisampled_color) == 1
    assert target.multisampled_color[0].samples == samples
    assert target.color[0].samples == 1
    assert target.multisampled_depth.samples == samples
    assert target.depth.samples == 1


def test_custom_resolve_reads_individual_samples(ctx, triangle_shaders, white_triangle, samples):
    """The point of the feature: bind the multisampled image to a sampler2DMS and
    resolve it by hand. Sample 0 alone stays hard-edged where the average is grey,
    which nothing but per-sample access can produce.

    The barriers are what this really tests — the multisampled attachment has to
    leave the pass in a layout a shader can read, and the validation layers are
    the referee for that (see conftest)."""
    if samples != 4:
        pytest.skip("resolve_ms.frag averages exactly four samples")

    target = _render_white_triangle(ctx, triangle_shaders, white_triangle, samples=samples, keep_samples=True)

    fullscreen = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    resolve = ctx.compile_shader(str(SHADER_DIR / "resolve_ms.frag"), bz.ShaderStage.FRAGMENT)
    screen = ctx.create_render_target(64, 64)
    pipe = (ctx.graphics_pipeline()
            .vertex_shader(fullscreen)
            .fragment_shader(resolve)
            .texture(0, bz.ShaderStage.FRAGMENT)
            .build(screen))

    pool = ctx.create_descriptor_pool(max_sets=4, textures=4)
    dset = pool.allocate_set(pipe)
    # NEAREST because a multisampled image is fetched, never filtered.
    dset.set_image(0, target.multisampled_color[0], sampler=ctx.create_sampler(filter=bz.Filter.NEAREST))

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(screen, clear_color=[0, 0, 0, 1])
    cmd.bind_pipeline(pipe)
    cmd.bind_descriptor_set(dset, pipe)
    cmd.draw(3)
    cmd.end_rendering(screen)
    ctx.submit(cmd)

    pixels = screen.color[0].read()
    single = pixels[:, :, 0].astype(np.int32)
    averaged = pixels[:, :, 1].astype(np.int32)

    assert list(pixels[32, 32][:2]) == [255, 255], "triangle interior covers every sample"
    assert list(pixels[2, 2][:2]) == [0, 0], "a corner covers none of them"
    assert _grey_edge_pixels(pixels) == 0, "sample 0 on its own must stay hard-edged"
    assert int(np.count_nonzero((averaged > 20) & (averaged < 235))) > 0, "the average must find partial coverage"
    assert int(np.count_nonzero(single != averaged)) > 0, "the samples must differ from their own average"


# ── depth resolve (SAMPLE_ZERO) ──────────────────────────────────────────────


def test_msaa_depth_resolves_to_sampleable_depth(ctx, triangle_shaders, white_triangle, samples):
    """With MSAA, depth resolves too (SAMPLE_ZERO), so target.depth stays a
    readable single-sample depth image."""
    target = _render_white_triangle(ctx, triangle_shaders, white_triangle, samples=samples, depth=bz.Format.D32F)
    depth = target.depth.read()
    assert depth.shape == (64, 64)
    assert depth.dtype == np.float32
    assert depth[2, 2] == pytest.approx(1.0), "corner keeps the clear depth"
    assert depth[40, 32] == pytest.approx(0.0, abs=1e-5), "triangle interior wrote z=0"


# ── per-attachment clears ────────────────────────────────────────────────────


def test_per_attachment_clears(ctx):
    """Two colour attachments cleared to different colours in one pass."""
    mrt = ctx.create_render_target(16, 16, color=[bz.Format.RGBA8, bz.Format.RGBA8])
    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(mrt, clear_color=[[255, 0, 0, 255], [0, 255, 0, 255]])
    cmd.end_rendering(mrt)
    ctx.submit(cmd)

    assert list(mrt.color[0].read()[8, 8]) == [255, 0, 0, 255]
    assert list(mrt.color[1].read()[8, 8]) == [0, 255, 0, 255]


def test_single_clear_applies_to_all_attachments(ctx):
    """A single [r,g,b,a] clears every attachment — the pre-existing behaviour."""
    mrt = ctx.create_render_target(16, 16, color=[bz.Format.RGBA8, bz.Format.RGBA8])
    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(mrt, clear_color=[0, 0, 255, 255])
    cmd.end_rendering(mrt)
    ctx.submit(cmd)

    assert list(mrt.color[0].read()[8, 8]) == [0, 0, 255, 255]
    assert list(mrt.color[1].read()[8, 8]) == [0, 0, 255, 255]


# ── validation / guards ──────────────────────────────────────────────────────


def test_non_power_of_two_sample_count_is_refused(ctx):
    with pytest.raises(bz.UnsupportedError) as info:
        ctx.create_render_target(16, 16, samples=3)
    assert "max_samples" in str(info.value)


def test_sample_count_over_max_is_refused(ctx):
    with pytest.raises(bz.UnsupportedError):
        ctx.create_render_target(16, 16, samples=ctx.max_samples() * 2)


def test_sample_shading_without_feature_is_refused(ctx, triangle_shaders, samples):
    """sample_shading needs the SAMPLE_RATE_SHADING feature; the session Context
    does not enable it, so the build must say so rather than fail in validation."""
    vert, frag = triangle_shaders
    target = ctx.create_render_target(16, 16, samples=samples)
    with pytest.raises(bz.UnsupportedError) as info:
        (ctx.graphics_pipeline()
         .vertex_shader(vert)
         .fragment_shader(frag)
         .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
         .sample_shading(True)
         .build(target))
    assert "SAMPLE_RATE_SHADING" in str(info.value)


def test_render_target_debug_name_is_accepted(ctx):
    """name= is additive and must not disturb rendering (the name only shows up
    in validation messages, which this suite already asserts stay silent)."""
    target = ctx.create_render_target(16, 16, color=bz.Format.RGBA8, name="gbuffer")
    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(target, clear_color=[0.2, 0.2, 0.2, 1.0])
    cmd.end_rendering(target)
    ctx.submit(cmd)
    assert target.color[0].read().shape == (16, 16, 4)
