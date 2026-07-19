"""Rendering with no window at all.

This is what the RenderTarget abstraction bought: before it, CommandBuffer's
recorded commands took a SwapchainRenderer&, so drawing anywhere other than a
window was impossible — and so was testing any of this.
"""

import numpy as np
import pytest

import bazalt as bz

CLEAR = [0.1, 0.2, 0.3, 1.0]
CLEAR_RGB = np.array([26, 51, 77])


def draw_triangle(ctx, target, shaders, buffers, clear=CLEAR):
    vert, frag = shaders
    vbuf, ibuf = buffers
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                .build(target))

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(target, clear_color=clear)
    cmd.bind_pipeline(pipeline)
    cmd.bind_vertex_buffer(vbuf)
    cmd.bind_index_buffer(ibuf)
    cmd.draw_indexed(3)
    cmd.end_rendering(target)
    ctx.submit(cmd)
    return target.read_pixels()


def test_render_target_reports_its_size(ctx):
    target = bz.RenderTarget(ctx, 64, 48)
    assert (target.width, target.height) == (64, 48)


def test_read_pixels_shape_and_dtype(ctx):
    target = bz.RenderTarget(ctx, 32, 16)
    # A clear-only pass counts as rendering; reading a never-rendered target
    # is an error (see test_read_pixels_before_any_render_is_an_error).
    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(target, clear_color=CLEAR)
    cmd.end_rendering(target)
    ctx.submit(cmd)

    pixels = target.read_pixels()
    assert pixels.shape == (16, 32, 4)
    assert pixels.dtype == np.uint8


def test_read_pixels_before_any_render_is_an_error(ctx):
    """Before the first submit the image is UNDEFINED — reading it back would
    return whatever the driver left in VRAM, which *sometimes* looks right.

    read_pixels used to do exactly that (the rendered_ flag was never set)."""
    target = bz.RenderTarget(ctx, 32, 32)
    with pytest.raises(bz.ResourceError) as info:
        target.read_pixels()
    assert "no contents" in str(info.value)


def test_recorded_but_unsubmitted_commands_do_not_count_as_rendering(ctx):
    """Only a submit flips the rendered flag; recording alone must not."""
    target = bz.RenderTarget(ctx, 32, 32)
    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(target, clear_color=CLEAR)
    cmd.end_rendering(target)
    # No ctx.submit(cmd).
    with pytest.raises(bz.ResourceError):
        target.read_pixels()


def test_clear_colour_reaches_the_image(ctx, triangle_shaders, triangle_buffers):
    target = bz.RenderTarget(ctx, 64, 64)
    pixels = draw_triangle(ctx, target, triangle_shaders, triangle_buffers)
    assert np.allclose(pixels[2, 2, :3], CLEAR_RGB, atol=2)


def test_triangle_is_actually_drawn(ctx, triangle_shaders, triangle_buffers):
    target = bz.RenderTarget(ctx, 64, 64)
    pixels = draw_triangle(ctx, target, triangle_shaders, triangle_buffers)
    centre = pixels[32, 32, :3]
    assert not np.allclose(centre, CLEAR_RGB, atol=2), "centre still shows the clear colour"
    assert centre.sum() > 100


def test_vertex_colours_are_interpolated(ctx, triangle_shaders, triangle_buffers):
    """Red at the top vertex, green bottom-left, blue bottom-right."""
    target = bz.RenderTarget(ctx, 64, 64)
    pixels = draw_triangle(ctx, target, triangle_shaders, triangle_buffers)

    near_top = pixels[20, 32, :3]
    bottom_left = pixels[46, 20, :3]
    bottom_right = pixels[46, 44, :3]

    assert near_top[0] == max(near_top), f"top should be reddest, got {near_top}"
    assert bottom_left[1] == max(bottom_left), f"bottom-left should be greenest, got {bottom_left}"
    assert bottom_right[2] == max(bottom_right), f"bottom-right should be bluest, got {bottom_right}"


def test_depth_target_renders(ctx, triangle_shaders, triangle_buffers):
    target = bz.RenderTarget(ctx, 64, 64, depth=bz.Format.D32F)
    pixels = draw_triangle(ctx, target, triangle_shaders, triangle_buffers)
    assert not np.allclose(pixels[32, 32, :3], CLEAR_RGB, atol=2)


def test_uint16_indices_draw_correctly(ctx, triangle_shaders, triangle_buffers):
    """bind_index_buffer used to hardcode VK_INDEX_TYPE_UINT32.

    A UINT16 index buffer was therefore read as UINT32 at half the count, which
    drew nothing or garbage, silently.
    """
    vbuf, _ = triangle_buffers
    ibuf16 = ctx.create_buffer([0, 1, 2], bz.BufferType.INDEX, bz.MemoryUsage.STATIC,
                               bz.DataType.UINT16)
    target = bz.RenderTarget(ctx, 64, 64)
    pixels = draw_triangle(ctx, target, triangle_shaders, (vbuf, ibuf16))
    assert not np.allclose(pixels[32, 32, :3], CLEAR_RGB, atol=2)


def test_two_targets_from_one_context(ctx, triangle_shaders, triangle_buffers):
    """Multiple render targets coexist — only SwapchainRenderers are restricted."""
    a = bz.RenderTarget(ctx, 32, 32)
    b = bz.RenderTarget(ctx, 48, 48)
    pa = draw_triangle(ctx, a, triangle_shaders, triangle_buffers)
    pb = draw_triangle(ctx, b, triangle_shaders, triangle_buffers)
    assert pa.shape == (32, 32, 4)
    assert pb.shape == (48, 48, 4)


def test_target_is_reusable_after_readback(ctx, triangle_shaders, triangle_buffers):
    """read_pixels must leave the image in final_layout so it stays samplable.

    Also the observable contract of the layout fix: the source barrier now
    declares the true layout instead of UNDEFINED, so the driver may not
    discard the rendered contents between two reads."""
    target = bz.RenderTarget(ctx, 32, 32)
    first = draw_triangle(ctx, target, triangle_shaders, triangle_buffers)
    second = target.read_pixels()
    assert np.array_equal(first, second)
    assert not np.allclose(first[16, 16, :3], CLEAR_RGB, atol=2), \
        "both reads agreeing on a blank image would prove nothing"
