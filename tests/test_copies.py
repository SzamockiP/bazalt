"""GPU-side copies: buffer to buffer, a fill, a resizing image blit, and
copy_image across the whole mip chain.

Before 0.18 the only way to move buffer bytes was a round trip through the host
or a compute shader written to do nothing but assign, and the only way to resize
an image was a fullscreen graphics pass. copy_image existed but copied mip 0
only, which is a copy of an image's top level rather than of the image.
"""

import numpy as np
import pytest

import bazalt as bz


def uint_buffer(ctx, values, usage=bz.BufferType.STORAGE):
    return ctx.create_buffer(list(values), usage, bz.MemoryUsage.STATIC, bz.DataType.UINT32)


def test_copy_buffer_moves_the_bytes(ctx):
    src = uint_buffer(ctx, [1, 2, 3, 4])
    dst = uint_buffer(ctx, [0, 0, 0, 0])

    cmd = ctx.create_command_buffer()
    cmd.begin().copy_buffer(src, dst)
    ctx.submit(cmd)

    assert list(dst.read("uint32")) == [1, 2, 3, 4]


def test_copy_buffer_honours_offsets_and_size(ctx):
    src = uint_buffer(ctx, [10, 20, 30, 40])
    dst = uint_buffer(ctx, [0, 0, 0, 0])

    # Two uint32s (8 bytes) from index 2 of the source into index 0.
    cmd = ctx.create_command_buffer()
    cmd.begin().copy_buffer(src, dst, src_offset=8, dst_offset=0, size=8)
    ctx.submit(cmd)

    assert list(dst.read("uint32")) == [30, 40, 0, 0]


def test_copy_buffer_rejects_a_region_that_does_not_fit(ctx):
    src = uint_buffer(ctx, [1, 2, 3, 4])
    dst = uint_buffer(ctx, [0, 0])

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with pytest.raises(bz.ResourceError):
        cmd.copy_buffer(src, dst)


def test_fill_buffer_zeroes(ctx):
    """The reason it exists: a counter has to start each frame at a known value,
    and saying so used to take a dispatch."""
    buf = uint_buffer(ctx, [7, 7, 7, 7])

    cmd = ctx.create_command_buffer()
    cmd.begin().fill_buffer(buf)
    ctx.submit(cmd)

    assert list(buf.read("uint32")) == [0, 0, 0, 0]


def test_fill_buffer_writes_the_given_word(ctx):
    """Two-sided: a fill that always wrote zero would pass the test above."""
    buf = uint_buffer(ctx, [0, 0, 0, 0])

    cmd = ctx.create_command_buffer()
    cmd.begin().fill_buffer(buf, 0xABCD)
    ctx.submit(cmd)

    assert list(buf.read("uint32")) == [0xABCD] * 4


def test_fill_buffer_honours_offset_and_size(ctx):
    buf = uint_buffer(ctx, [1, 1, 1, 1])

    cmd = ctx.create_command_buffer()
    cmd.begin().fill_buffer(buf, 9, offset=8, size=8)
    ctx.submit(cmd)

    assert list(buf.read("uint32")) == [1, 1, 9, 9]


def test_fill_buffer_rejects_an_unaligned_region(ctx):
    """vkCmdFillBuffer repeats one 32-bit word, so both ends must be multiples
    of 4. Saying so beats a validation message about the same thing."""
    buf = uint_buffer(ctx, [0, 0, 0, 0])

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with pytest.raises(bz.ResourceError):
        cmd.fill_buffer(buf, 0, offset=2)


def test_transfers_work_on_a_dynamic_buffer(ctx):
    """DYNAMIC buffers gained the transfer usage bits for this. Without them
    "which buffers can the GPU copy into" would be a second rule to remember,
    and each of these would be a validation error instead.

    Read back through a STATIC copy rather than off the DYNAMIC buffer itself:
    a DynamicBuffer holds one VkBuffer per frame slot and read() maps the
    current one, while the headless submit advances the ring — so which slot a
    later read sees is a question about the ring, not about the fill.
    """
    dynamic = ctx.create_buffer([1, 1, 1, 1], bz.BufferType.STORAGE,
                                bz.MemoryUsage.DYNAMIC, bz.DataType.UINT32)
    out = uint_buffer(ctx, [0, 0, 0, 0])

    cmd = ctx.create_command_buffer()
    cmd.begin().fill_buffer(dynamic, 5).copy_buffer(dynamic, out)
    ctx.submit(cmd)

    assert list(out.read("uint32")) == [5, 5, 5, 5]


def test_blit_image_downsamples(ctx):
    """A resize, which copy_image refuses and generate_mipmaps only does inside
    one image. A flat colour survives any filter, so the assertion is about the
    pixels arriving at all in the new size."""
    big = np.zeros((32, 32, 4), dtype=np.uint8)
    big[:, :, 0] = 255
    big[:, :, 3] = 255
    src = ctx.create_image(big)
    dst = ctx.create_image(8, 8, bz.Format.RGBA8)

    cmd = ctx.create_command_buffer()
    cmd.begin().blit_image(src, dst)
    ctx.submit(cmd)

    out = dst.read()
    assert out.shape == (8, 8, 4)
    assert out[4, 4, 0] > 200
    assert out[4, 4, 1] < 50


def test_blit_image_upsamples(ctx):
    small = np.zeros((4, 4, 4), dtype=np.uint8)
    small[:, :, 1] = 255
    small[:, :, 3] = 255
    src = ctx.create_image(small)
    dst = ctx.create_image(16, 16, bz.Format.RGBA8)

    cmd = ctx.create_command_buffer()
    cmd.begin().blit_image(src, dst, filter=bz.Filter.NEAREST)
    ctx.submit(cmd)

    out = dst.read()
    assert out.shape == (16, 16, 4)
    assert out[8, 8, 1] > 200


def test_blit_image_rejects_the_same_image_twice(ctx):
    img = ctx.create_image(8, 8, bz.Format.RGBA8)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with pytest.raises(bz.ResourceError):
        cmd.blit_image(img, img)


def test_blit_image_is_refused_inside_a_rendering_scope(ctx):
    src = ctx.create_image(8, 8, bz.Format.RGBA8)
    dst = ctx.create_image(4, 4, bz.Format.RGBA8)
    target = bz.RenderTarget(ctx, 8, 8)

    cmd = ctx.create_command_buffer()
    cmd.begin().begin_rendering(target, clear_color=[0, 0, 0, 1])
    with pytest.raises(bz.ResourceError):
        cmd.blit_image(src, dst)


def test_copy_image_copies_the_whole_mip_chain(ctx):
    """0.17 copied mip 0 only and left levels 1..N holding the destination's old
    pixels. Read back through a big LOD, which clamps to the smallest level:
    that is the level the old behaviour never touched.
    """
    red = np.zeros((16, 16, 4), dtype=np.uint8)
    red[:, :, 0] = 255
    red[:, :, 3] = 255
    src = ctx.create_image(red, mipmaps=True)
    assert src.mip_levels > 1

    dst = ctx.create_image(16, 16, bz.Format.RGBA8, mip_levels=src.mip_levels)

    cmd = ctx.create_command_buffer()
    cmd.begin().copy_image(src, dst)
    ctx.submit(cmd)

    # Sample the smallest level of the destination. Before the change this read
    # a level that was never written.
    from conftest import SHADER_DIR
    import struct

    fullscreen = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    lod_frag = ctx.compile_shader(str(SHADER_DIR / "sample_lod.frag"), bz.ShaderStage.FRAGMENT)
    screen = bz.RenderTarget(ctx, 8, 8)
    pipe = (ctx.graphics_pipeline()
            .vertex_shader(fullscreen)
            .fragment_shader(lod_frag)
            .texture(0, bz.ShaderStage.FRAGMENT, set=0)
            .push_constant(4, bz.ShaderStage.FRAGMENT)
            .build(screen))

    pool = ctx.create_descriptor_pool(max_sets=1, samplers=1)
    dset = pool.allocate_set(pipe, set=0)
    dset.set_image(0, dst, sampler=ctx.create_sampler(filter=bz.Filter.NEAREST))

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(screen, clear_color=[0, 0, 0, 1]):
        cmd.bind_pipeline(pipe)
        cmd.bind_descriptor_set(dset, pipe, 0)
        cmd.push_constants(pipe, 0, struct.pack("f", 32.0))  # clamps to the smallest mip
        cmd.draw(3)
    ctx.submit(cmd)

    out = screen.color[0].read()
    assert out[4, 4, 0] > 200, "the smallest mip of the destination was not copied"
    assert out[4, 4, 1] < 50
