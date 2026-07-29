"""Offsets near the type maximum must be refused, not wrapped around.

Every one of these bounds checks used to read `offset + length > size`. The
operands are unsigned, so an offset near the type maximum wrapped the sum to a
small number and the check passed — the check was a bypass. The worst of them
reached a memcpy, the rest reached Vulkan.

The assertion is that each verb raises a bazalt exception. A validation error
would mean the argument got past the check, and the ctx fixture fails the test
on one of those, so the two halves of "refused cleanly" are both covered.
"""

import numpy as np
import pytest

import bazalt as bz

U64_MAX = 2**64 - 1
U32_MAX = 2**32 - 1


def test_dynamic_buffer_update_refuses_a_wrapping_offset(ctx):
    """The one that reached memcpy: a wild destination pointer, not a slow path."""
    buf = ctx.create_buffer([0.0] * 16, bz.BufferType.UNIFORM,
                            bz.MemoryUsage.DYNAMIC, bz.DataType.FLOAT)
    payload = np.zeros(4, dtype=np.float32)

    # offset + 16 wraps to 6, which is <= the 64-byte buffer.
    with pytest.raises(bz.ResourceError):
        buf.update(payload, offset=U64_MAX - 9)


def test_image_update_refuses_a_wrapping_region(ctx):
    """x + w wraps below the mip width, and x then casts to a negative
    int32_t straight into VkBufferImageCopy.imageOffset."""
    image = ctx.create_image(8, 8, bz.Format.RGBA8)
    pixels = np.zeros((8, 4, 4), dtype=np.uint8)

    # x + 4 wraps to 2, which is < 8.
    with pytest.raises(bz.ResourceError):
        image.update(pixels, region=(U32_MAX - 1, 0, 4, 8))


def test_image_read_refuses_a_wrapping_layer(ctx):
    """image.read() was the sixth site, and the one 0.20 missed: `base_layer +
    layers` wrapped to 0, the check passed, and baseArrayLayer=4294967295
    reached vkCmdCopyImageToBuffer. It returned an array of layer 0."""
    image = ctx.create_image(np.zeros((4, 4, 4), dtype=np.uint8))

    with pytest.raises(bz.ResourceError):
        image.read(layer=U32_MAX)


def test_copy_buffer_refuses_a_wrapping_offset(ctx):
    src = ctx.create_buffer([0.0] * 16, bz.BufferType.STORAGE,
                            bz.MemoryUsage.STATIC, bz.DataType.FLOAT)
    dst = ctx.create_buffer([0.0] * 16, bz.BufferType.STORAGE,
                            bz.MemoryUsage.STATIC, bz.DataType.FLOAT)
    cmd = ctx.create_command_buffer()
    cmd.begin()

    with pytest.raises(bz.ResourceError):
        cmd.copy_buffer(src, dst, src_offset=U64_MAX - 9, size=16)
    with pytest.raises(bz.ResourceError):
        cmd.copy_buffer(src, dst, dst_offset=U64_MAX - 9, size=16)


def test_fill_buffer_refuses_a_wrapping_offset(ctx):
    buf = ctx.create_buffer([0.0] * 16, bz.BufferType.STORAGE,
                            bz.MemoryUsage.STATIC, bz.DataType.FLOAT)
    cmd = ctx.create_command_buffer()
    cmd.begin()

    # Multiple of 4, so it gets past the alignment check and reaches the size one.
    with pytest.raises(bz.ResourceError):
        cmd.fill_buffer(buf, 0, offset=U64_MAX - 15, size=16)


def test_draw_indirect_refuses_a_wrapping_offset(ctx, triangle_shaders):
    vert, frag = triangle_shaders
    target = bz.RenderTarget(ctx, 64, 64)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                .build(target))
    args = ctx.create_buffer([3, 1, 0, 0], bz.BufferType.STORAGE,
                             bz.MemoryUsage.STATIC, bz.DataType.UINT32)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(target, clear_color=[0, 0, 0, 1])
    cmd.bind_pipeline(pipeline)

    # One VkDrawIndirectCommand is 16 bytes, so offset + 16 wraps to 0.
    with pytest.raises(bz.ResourceError):
        cmd.draw_indirect(args, offset=U64_MAX - 15, count=1)


def test_the_ordinary_offsets_still_work(ctx):
    """The guard rejects the wrap, not the feature. Without this the whole file
    would pass on a bounds check hardcoded to refuse everything."""
    buf = ctx.create_buffer([0.0] * 16, bz.BufferType.UNIFORM,
                            bz.MemoryUsage.DYNAMIC, bz.DataType.FLOAT)
    buf.update(np.zeros(4, dtype=np.float32), offset=48)  # last 16 bytes, exact fit
    with pytest.raises(bz.ResourceError):
        buf.update(np.zeros(4, dtype=np.float32), offset=52)  # one float past the end
