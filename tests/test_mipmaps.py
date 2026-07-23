"""Mipmaps: explicit levels on empty images and the standalone generator.

create_image(array/list, mipmaps=True) and load_image(mipmaps=...) cover the
upload path (tested in test_images.py). Here: empty mipped images and
cmd.generate_mipmaps(), which fills levels 1..N from mip 0 for images written by
compute or a render pass. The mip-transition barriers are audited by the
validation-as-assert ctx fixture, not just the sampled colour.
"""

import struct

import numpy as np
import pytest

import bazalt as bz

from conftest import SHADER_DIR


def test_empty_image_allocates_requested_levels(ctx):
    img = ctx.create_image(16, 16, bz.Format.RGBA8, mip_levels=5)
    assert img.mip_levels == 5


def test_mip_levels_capped_to_the_full_chain(ctx):
    """16x16 has 5 levels (16,8,4,2,1); asking for more is a ResourceError."""
    with pytest.raises(bz.ResourceError):
        ctx.create_image(16, 16, bz.Format.RGBA8, mip_levels=6)


def test_generate_mipmaps_requires_a_mipped_image(ctx):
    img = ctx.create_image(16, 16, bz.Format.RGBA8)  # single level
    cmd = ctx.create_command_buffer()
    cmd.begin()
    with pytest.raises(bz.ResourceError):
        cmd.generate_mipmaps(img)


def test_generate_mipmaps_refused_inside_a_rendering_scope(ctx):
    img = ctx.create_image(16, 16, bz.Format.RGBA8, mip_levels=5)
    target = bz.RenderTarget(ctx, 8, 8)
    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(target)
    with pytest.raises(bz.ResourceError, match="rendering scope"):
        cmd.generate_mipmaps(img)
    cmd.end_rendering(target)


def test_generate_mipmaps_fills_the_chain(ctx):
    """Empty mipped image, mip 0 written by compute, generate_mipmaps blits the
    rest. Sampling at a high LOD (clamped to the smallest level) returns mip 0's
    solid colour — proving the lower levels were generated and left sampleable.
    Every mip-transition barrier is audited by the ctx fixture."""
    fill_comp = ctx.compile_shader(str(SHADER_DIR / "store_const.comp"), bz.ShaderStage.COMPUTE)
    fill = ctx.compute_pipeline().shader(fill_comp).storage_image(0).build()

    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "sample_lod.frag"), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, 8, 8)
    sample = (ctx.graphics_pipeline()
              .vertex_shader(vert)
              .fragment_shader(frag)
              .texture(0, bz.ShaderStage.FRAGMENT, set=0)
              .push_constant(4, bz.ShaderStage.FRAGMENT)
              .build(target))

    img = ctx.create_image(16, 16, bz.Format.RGBA8, mip_levels=5)
    pool = ctx.create_descriptor_pool(max_sets=2, storage_images=1, samplers=1)
    fill_set = pool.allocate_set(fill, set=0)
    fill_set.set_storage_image(0, img)
    sample_set = pool.allocate_set(sample, set=0)
    sample_set.set_image(0, img)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.bind_pipeline(fill).bind_descriptor_set(fill_set, fill, set=0).dispatch(2, 2)
    # mip 0 is fresh out of compute (GENERAL): src=SHADER_WRITE.
    cmd.generate_mipmaps(img, src=bz.Access.SHADER_WRITE)
    cmd.begin_rendering(target)
    cmd.bind_pipeline(sample).bind_descriptor_set(sample_set, sample, set=0)
    cmd.push_constants(sample, 0, struct.pack("f", 32.0))  # clamp to the smallest mip
    cmd.draw(3)
    cmd.end_rendering(target)
    ctx.submit(cmd)

    # store_const writes (0.25, 0.5, 0.75); a solid colour box-downsamples to
    # itself at every level -> (64, 128, 191).
    got = target.read_pixels()[4, 4, :3]
    assert np.allclose(got, [64, 128, 191], atol=2), got
