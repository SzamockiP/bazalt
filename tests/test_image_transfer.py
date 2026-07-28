"""0.17: cmd.copy_image, cmd.clear_image, ctx.wait and the sampler border.

An image could be created, uploaded to, rendered into and read back, but never
copied to another one — which is what a history buffer (motion blur, a feedback
trail) and a compute ping-pong are made of.

Headless, so CI covers it.
"""

import pathlib

import numpy as np
import pytest

import bazalt as bz

SHADER_DIR = pathlib.Path(__file__).parent / "shaders"


def checkerboard():
    a = np.zeros((16, 16, 4), dtype=np.uint8)
    a[::2, ::2] = [255, 0, 0, 255]
    a[1::2, 1::2] = [0, 0, 255, 255]
    return a


def test_a_copy_reproduces_the_source(ctx):
    src = ctx.create_image(checkerboard())
    dst = ctx.create_image(16, 16, bz.Format.RGBA8)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.copy_image(src, dst)
    ctx.submit(cmd)

    assert np.array_equal(dst.read(), src.read())


def test_a_copy_of_a_compute_result_names_where_the_source_is(ctx):
    """A storage image lives in GENERAL, so the copy has to be told — the same
    vocabulary generate_mipmaps uses."""
    shader = ctx.compile_shader(str(SHADER_DIR / "store_const.comp"), bz.ShaderStage.COMPUTE)
    pipeline = ctx.compute_pipeline().shader(shader).storage_image(0).build()

    baked = ctx.create_image(16, 16, bz.Format.RGBA8)
    copy = ctx.create_image(16, 16, bz.Format.RGBA8)
    pool = ctx.create_descriptor_pool(max_sets=1, storage_images=1)
    dset = pool.allocate_set(pipeline, set=0)
    dset.set_storage_image(0, baked)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.bind_pipeline(pipeline)
    cmd.bind_descriptor_set(dset, pipeline, set=0)
    cmd.dispatch(16, 16)
    cmd.copy_image(baked, copy, src_access=bz.Access.SHADER_WRITE)
    ctx.submit(cmd)

    assert np.array_equal(copy.read(), baked.read())
    assert copy.read().any(), "the compute shader wrote nothing, so the copy proves nothing"


def test_a_copy_refuses_a_mismatch(ctx):
    src = ctx.create_image(checkerboard())
    smaller = ctx.create_image(8, 8, bz.Format.RGBA8)
    other_format = ctx.create_image(16, 16, bz.Format.R8)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with pytest.raises(bz.ResourceError, match="size, format and layer"):
        cmd.copy_image(src, smaller)
    with pytest.raises(bz.ResourceError, match="size, format and layer"):
        cmd.copy_image(src, other_format)


def test_a_copy_is_refused_inside_a_rendering_scope(ctx):
    target = bz.RenderTarget(ctx, 16, 16)
    src = ctx.create_image(checkerboard())
    dst = ctx.create_image(16, 16, bz.Format.RGBA8)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(target)
    with pytest.raises(bz.ResourceError, match="rendering scope"):
        cmd.copy_image(src, dst)
    with pytest.raises(bz.ResourceError, match="rendering scope"):
        cmd.clear_image(dst, [1.0, 0.0, 0.0, 1.0])
    cmd.end_rendering(target)


def test_clear_image_fills_every_texel(ctx):
    image = ctx.create_image(checkerboard())

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.clear_image(image, [0.0, 1.0, 0.0, 1.0])
    ctx.submit(cmd)

    pixels = image.read()
    assert np.all(pixels[..., 1] == 255) and np.all(pixels[..., 0] == 0)


def test_clear_image_refuses_depth(ctx):
    """The depth clear belongs to the pass that renders into it."""
    depth = ctx.create_image(16, 16, bz.Format.D32F)
    cmd = ctx.create_command_buffer()
    cmd.begin()
    with pytest.raises(bz.ResourceError, match="clear_depth"):
        cmd.clear_image(depth, [1.0, 1.0, 1.0, 1.0])


def test_wait_returns_after_the_work(ctx):
    """Nothing observable beyond "it returns and the result is there" — the
    submit already waits. It exists for the cases outside a frame."""
    image = ctx.create_image(checkerboard())
    dst = ctx.create_image(16, 16, bz.Format.RGBA8)
    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.copy_image(image, dst)
    ctx.submit(cmd)
    ctx.wait()
    assert np.array_equal(dst.read(), image.read())


def test_a_border_sampler_is_its_own_cache_entry(ctx):
    """The cache keys on the whole description, and the border colour and the
    lod bias are part of it — two different borders must not share a handle."""
    black = ctx.create_sampler(address_mode=bz.AddressMode.CLAMP_TO_BORDER)
    white = ctx.create_sampler(address_mode=bz.AddressMode.CLAMP_TO_BORDER,
                               border_color=bz.BorderColor.OPAQUE_WHITE)
    biased = ctx.create_sampler(address_mode=bz.AddressMode.CLAMP_TO_BORDER, mip_lod_bias=1.5)
    same = ctx.create_sampler(address_mode=bz.AddressMode.CLAMP_TO_BORDER)

    assert black is not white
    assert black is not biased
    assert black is same, "identical descriptions must still share one sampler"


def test_the_border_colour_is_what_a_sample_outside_the_image_reads(ctx):
    """The shadow-map fix, shown on a colour texture: CLAMP repeats the edge
    texel, CLAMP_TO_BORDER with a white border reads white."""
    texture = ctx.create_image(np.zeros((16, 16, 4), dtype=np.uint8))
    target = bz.RenderTarget(ctx, 32, 32)
    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "sample_outside.frag"), bz.ShaderStage.FRAGMENT)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .texture(0, bz.ShaderStage.FRAGMENT, 0)
                .build(target))

    def run(sampler):
        pool = ctx.create_descriptor_pool(max_sets=1, samplers=1)
        dset = pool.allocate_set(pipeline, set=0)
        dset.set_image(0, texture, sampler)
        cmd = ctx.create_command_buffer()
        cmd.begin()
        with cmd.rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0]) as c:
            c.bind_pipeline(pipeline).bind_descriptor_set(dset, pipeline, set=0).draw(3)
        ctx.submit(cmd)
        return int(target.color[0].read()[16, 16][0])

    clamped = ctx.create_sampler(address_mode=bz.AddressMode.CLAMP)
    bordered = ctx.create_sampler(address_mode=bz.AddressMode.CLAMP_TO_BORDER,
                                  border_color=bz.BorderColor.OPAQUE_WHITE)

    assert run(clamped) == 0, "CLAMP read something other than the black edge texel"
    assert run(bordered) == 255, "the white border was not read outside the image"
