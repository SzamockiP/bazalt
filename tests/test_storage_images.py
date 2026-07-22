"""Storage images in compute: dispatch -> imageStore -> readback, and the
auto-barriers that make compute->compute and compute->sample work.

Like test_compute.py these are headless (dispatch -> read), so they run on a
software rasterizer in CI. The point of the release is the ResourceTracker
learning image layouts: every auto-barrier path here is also validated by the
sync-validation-as-assert ctx fixture, not just by the numeric result.
"""

import numpy as np
import pytest

import bazalt as bz

from conftest import SHADER_DIR


@pytest.fixture
def fullscreen_vert(ctx):
    return ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)


def _fill_pipeline(ctx):
    comp = ctx.compile_shader(str(SHADER_DIR / "store_const.comp"), bz.ShaderStage.COMPUTE)
    return ctx.compute_pipeline().shader(comp).storage_image(0).build()


# ── P1: descriptor plumbing ───────────────────────────────────────────────


def test_compute_pipeline_builds_with_a_storage_image(ctx):
    assert _fill_pipeline(ctx) is not None


def test_set_storage_image_on_sampler_binding_is_refused(ctx, fullscreen_vert):
    """A storage_image write to a combined-image-sampler binding is a mistake
    diagnosed at the call site, not left to the validation layers."""
    frag = ctx.compile_shader(str(SHADER_DIR / "textured.frag"), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, 16, 16)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(fullscreen_vert)
                .fragment_shader(frag)
                .texture(0, bz.ShaderStage.FRAGMENT, set=0)
                .build(target))
    pool = ctx.create_descriptor_pool(max_sets=4, samplers=4)
    dset = pool.allocate_set(pipeline, set=0)
    img = ctx.create_image(16, 16, bz.Format.RGBA8)

    with pytest.raises(bz.ResourceError):
        dset.set_storage_image(0, img)


# ── P3: compute -> readback ────────────────────────────────────────────────


def test_dispatch_fills_a_storage_image(ctx):
    pipeline = _fill_pipeline(ctx)
    img = ctx.create_image(16, 16, bz.Format.RGBA8)

    pool = ctx.create_descriptor_pool(max_sets=4, storage_images=4)
    dset = pool.allocate_set(pipeline, set=0)
    dset.set_storage_image(0, img)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.bind_pipeline(pipeline)
    cmd.bind_descriptor_set(dset, pipeline, set=0)
    cmd.dispatch(2, 2)  # 16/8 x 16/8
    ctx.submit(cmd)

    pixels = img.read()
    assert pixels.shape == (16, 16, 4)
    # (0.25, 0.5, 0.75, 1.0) * 255, UNORM rounding -> ±2.
    assert np.allclose(pixels[8, 8], [64, 128, 191, 255], atol=2), pixels[8, 8]


# ── P3: compute -> compute (image barrier between two dispatches) ──────────


def test_compute_to_compute_barrier(ctx):
    fill = _fill_pipeline(ctx)
    scale_comp = ctx.compile_shader(str(SHADER_DIR / "store_scale.comp"), bz.ShaderStage.COMPUTE)
    scale = ctx.compute_pipeline().shader(scale_comp).storage_image(0).build()

    img = ctx.create_image(16, 16, bz.Format.RGBA8)
    pool = ctx.create_descriptor_pool(max_sets=4, storage_images=4)
    fill_set = pool.allocate_set(fill, set=0)
    fill_set.set_storage_image(0, img)
    scale_set = pool.allocate_set(scale, set=0)
    scale_set.set_storage_image(0, img)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.bind_pipeline(fill)
    cmd.bind_descriptor_set(fill_set, fill, set=0)
    cmd.dispatch(2, 2)
    cmd.bind_pipeline(scale)
    cmd.bind_descriptor_set(scale_set, scale, set=0)
    cmd.dispatch(2, 2)  # reads what fill wrote -> needs an auto image barrier
    ctx.submit(cmd)

    # RGB halved, alpha untouched: (32, 64, 95, 255).
    assert np.allclose(img.read()[8, 8], [32, 64, 95, 255], atol=2), img.read()[8, 8]


# ── P3: compute -> graphics sample (layout GENERAL -> SHADER_READ_ONLY) ────


def test_compute_written_image_sampled_by_graphics(ctx, fullscreen_vert):
    fill = _fill_pipeline(ctx)
    frag = ctx.compile_shader(str(SHADER_DIR / "textured.frag"), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, 32, 32)
    gfx = (ctx.graphics_pipeline()
           .vertex_shader(fullscreen_vert)
           .fragment_shader(frag)
           .texture(0, bz.ShaderStage.FRAGMENT, set=0)
           .build(target))

    img = ctx.create_image(16, 16, bz.Format.RGBA8)
    pool = ctx.create_descriptor_pool(max_sets=4, storage_images=4, samplers=4)
    fill_set = pool.allocate_set(fill, set=0)
    fill_set.set_storage_image(0, img)
    sample_set = pool.allocate_set(gfx, set=0)
    sample_set.set_image(0, img)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.bind_pipeline(fill)
    cmd.bind_descriptor_set(fill_set, fill, set=0)
    cmd.dispatch(2, 2)
    cmd.begin_rendering(target)
    cmd.bind_pipeline(gfx)
    cmd.bind_descriptor_set(sample_set, gfx, set=0)
    cmd.draw(3)  # samples img -> auto GENERAL->SHADER_READ_ONLY, hoisted before begin_rendering
    cmd.end_rendering(target)
    ctx.submit(cmd)

    assert np.allclose(target.read_pixels()[16, 16, :3], [64, 128, 191], atol=2), \
        target.read_pixels()[16, 16]
