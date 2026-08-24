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


def uint_buffer(ctx, values, usage=bz.BufferUsage.STORAGE):
    return ctx.create_buffer(list(values), usage, bz.MemoryUsage.STATIC, dtype=np.uint32)


def test_copy_buffer_moves_the_bytes(ctx):
    src = uint_buffer(ctx, [1, 2, 3, 4])
    dst = uint_buffer(ctx, [0, 0, 0, 0])

    g = ctx.graph()
    g.add_pass().copy_buffer(src, dst)
    ctx.submit(g)

    assert list(dst.read("uint32")) == [1, 2, 3, 4]


def test_copy_buffer_honours_offsets_and_size(ctx):
    src = uint_buffer(ctx, [10, 20, 30, 40])
    dst = uint_buffer(ctx, [0, 0, 0, 0])

    # Two uint32s (8 bytes) from index 2 of the source into index 0.
    g = ctx.graph()
    g.add_pass().copy_buffer(src, dst, src_offset=8, dst_offset=0, size=8)
    ctx.submit(g)

    assert list(dst.read("uint32")) == [30, 40, 0, 0]


def test_copy_buffer_rejects_a_region_that_does_not_fit(ctx):
    src = uint_buffer(ctx, [1, 2, 3, 4])
    dst = uint_buffer(ctx, [0, 0])

    g = ctx.graph()
    p = g.add_pass()
    with pytest.raises(bz.ResourceError):
        p.copy_buffer(src, dst)


def test_fill_buffer_zeroes(ctx):
    """The reason it exists: a counter has to start each frame at a known value,
    and saying so used to take a dispatch."""
    buf = uint_buffer(ctx, [7, 7, 7, 7])

    g = ctx.graph()
    g.add_pass().fill_buffer(buf)
    ctx.submit(g)

    assert list(buf.read("uint32")) == [0, 0, 0, 0]


def test_fill_buffer_writes_the_given_word(ctx):
    """Two-sided: a fill that always wrote zero would pass the test above."""
    buf = uint_buffer(ctx, [0, 0, 0, 0])

    g = ctx.graph()
    g.add_pass().fill_buffer(buf, 0xABCD)
    ctx.submit(g)

    assert list(buf.read("uint32")) == [0xABCD] * 4


def test_fill_buffer_honours_offset_and_size(ctx):
    buf = uint_buffer(ctx, [1, 1, 1, 1])

    g = ctx.graph()
    g.add_pass().fill_buffer(buf, 9, offset=8, size=8)
    ctx.submit(g)

    assert list(buf.read("uint32")) == [1, 1, 9, 9]


def test_fill_buffer_rejects_an_unaligned_region(ctx):
    """vkCmdFillBuffer repeats one 32-bit word, so both ends must be multiples
    of 4. Saying so beats a validation message about the same thing."""
    buf = uint_buffer(ctx, [0, 0, 0, 0])

    g = ctx.graph()
    p = g.add_pass()
    with pytest.raises(bz.ResourceError):
        p.fill_buffer(buf, 0, offset=2)


def test_transfers_work_on_a_dynamic_buffer(ctx):
    """DYNAMIC buffers gained the transfer usage bits for this. Without them
    "which buffers can the GPU copy into" would be a second rule to remember,
    and each of these would be a validation error instead.

    Read back through a STATIC copy rather than off the DYNAMIC buffer itself:
    a DynamicBuffer holds one VkBuffer per frame slot and read() maps the
    current one, while the headless submit advances the ring — so which slot a
    later read sees is a question about the ring, not about the fill.
    """
    dynamic = ctx.create_buffer([1, 1, 1, 1], bz.BufferUsage.STORAGE,
                                bz.MemoryUsage.DYNAMIC, dtype=np.uint32)
    out = uint_buffer(ctx, [0, 0, 0, 0])

    g = ctx.graph()
    g.add_pass().fill_buffer(dynamic, 5).copy_buffer(dynamic, out)
    ctx.submit(g)

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

    g = ctx.graph()
    g.add_pass().blit_image(src, dst)
    ctx.submit(g)

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

    g = ctx.graph()
    g.add_pass().blit_image(src, dst, filter=bz.Filter.NEAREST)
    ctx.submit(g)

    out = dst.read()
    assert out.shape == (16, 16, 4)
    assert out[8, 8, 1] > 200


def test_blit_image_rejects_the_same_image_twice(ctx):
    img = ctx.create_image(8, 8, bz.Format.RGBA8)

    g = ctx.graph()
    p = g.add_pass()
    with pytest.raises(bz.ResourceError):
        p.blit_image(img, img)


def test_blit_image_is_refused_inside_a_rendering_scope(ctx):
    src = ctx.create_image(8, 8, bz.Format.RGBA8)
    dst = ctx.create_image(4, 4, bz.Format.RGBA8)
    target = ctx.create_render_target(8, 8)

    g = ctx.graph()
    p = g.add_pass(target, clear_color=[0, 0, 0, 1])
    with pytest.raises(bz.StateError):
        p.blit_image(src, dst)


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

    g = ctx.graph()
    g.add_pass().copy_image(src, dst)
    ctx.submit(g)

    # Sample the smallest level of the destination. Before the change this read
    # a level that was never written.
    from conftest import SHADER_DIR
    import struct

    fullscreen = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    lod_frag = ctx.compile_shader(str(SHADER_DIR / "sample_lod.frag"), bz.ShaderStage.FRAGMENT)
    screen = ctx.create_render_target(8, 8)
    pipe = (ctx.graphics_pipeline()
            .vertex_shader(fullscreen)
            .fragment_shader(lod_frag)
            .texture(0, bz.ShaderStage.FRAGMENT, set=0)
            .push_constant(4, bz.ShaderStage.FRAGMENT)
            .build(screen))

    pool = ctx.create_descriptor_pool(max_sets=1, textures=1)
    dset = pool.allocate_set(pipe, set=0)
    dset.set_image(0, dst, sampler=ctx.create_sampler(filter=bz.Filter.NEAREST))

    sample = ctx.graph()
    with sample.add_pass(screen, clear_color=[0, 0, 0, 1]) as p:
        p.bind_pipeline(pipe)
        p.bind_descriptor_set(dset, pipe, 0)
        p.push_constants(pipe, 0, struct.pack("f", 32.0))  # clamps to the smallest mip
        p.draw(3)
    ctx.submit(sample)

    out = screen.color[0].read()
    assert out[4, 4, 0] > 200, "the smallest mip of the destination was not copied"
    assert out[4, 4, 1] < 50


def test_update_buffer_writes_the_bytes(ctx):
    """The command-stream patch: bytes and arrays land at their offset with no
    staging buffer and no second submit."""
    buf = ctx.create_buffer(8 * 4, bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    g = ctx.graph()
    with g.add_pass(name="patch") as p:
        p.fill_buffer(buf, 0)
        p.update_buffer(buf, np.arange(2, dtype=np.uint32))
        p.update_buffer(buf, b"\x07\x00\x00\x00", offset=8)
    ctx.submit(g)
    assert buf.read(np.uint32).tolist() == [0, 1, 7, 0, 0, 0, 0, 0]


def test_update_buffer_refuses_too_much_or_unaligned(ctx):
    """vkCmdUpdateBuffer's own limits, named at the call with the fix."""
    big = ctx.create_buffer(1 << 20, bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    g = ctx.graph()
    with g.add_pass() as p:
        with pytest.raises(bz.ResourceError, match="65536"):
            p.update_buffer(big, bytes(65540))
        with pytest.raises(bz.ResourceError, match="multiples of 4"):
            p.update_buffer(big, bytes(6))
        with pytest.raises(bz.ResourceError, match="multiples of 4"):
            p.update_buffer(big, bytes(8), offset=2)


def test_copy_buffer_to_image_fills_a_mip_and_the_mirror_reads_it_back(ctx):
    """One (layer, mip) each way, tightly packed. The round trip through a
    second buffer and image.read(mip=1) must agree byte for byte."""
    img = ctx.create_image(32, 32, bz.Format.RGBA8, mip_levels=2)
    pattern = np.arange(16 * 16 * 4, dtype=np.uint8)
    src = ctx.create_buffer(pattern, bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    dst = ctx.create_buffer(16 * 16 * 4, bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    g = ctx.graph()
    with g.add_pass(name="roundtrip") as p:
        p.copy_buffer_to_image(src, img, mip=1)
        p.copy_image_to_buffer(img, dst, mip=1)
    ctx.submit(g)
    assert np.array_equal(dst.read(np.uint8), pattern)
    assert np.array_equal(img.read(mip=1).ravel(), pattern)


def test_copy_buffer_to_image_refuses_a_missing_subresource_or_a_short_buffer(ctx):
    img = ctx.create_image(16, 16, bz.Format.RGBA8)
    buf = ctx.create_buffer(8, bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    g = ctx.graph()
    with g.add_pass() as p:
        with pytest.raises(bz.ResourceError, match="does not exist"):
            p.copy_buffer_to_image(buf, img, layer=3)
        with pytest.raises(bz.ResourceError, match="cannot hold"):
            p.copy_buffer_to_image(buf, img)
        with pytest.raises(bz.ResourceError, match="does not exist"):
            p.copy_image_to_buffer(img, buf, mip=1)


def test_the_buffer_image_copies_work_in_a_manual_pass(ctx):
    """The image side records its own transitions (the copy_image shape), so
    auto_barriers=False changes nothing about these verbs."""
    img = ctx.create_image(8, 8, bz.Format.RGBA8)
    src = ctx.create_buffer(np.full(8 * 8 * 4, 9, np.uint8),
                            bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    dst = ctx.create_buffer(8 * 8 * 4, bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    g = ctx.graph()
    with g.add_pass(name="manual", auto_barriers=False) as p:
        p.copy_buffer_to_image(src, img)
        p.copy_image_to_buffer(img, dst)
    ctx.submit(g)
    assert (dst.read(np.uint8) == 9).all()
