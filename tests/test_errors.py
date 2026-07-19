"""The exception type is the contract for whether you can carry on.

Everything used to arrive as a bare RuntimeError, so a typo in a shader was
indistinguishable from a lost device and neither could be handled.
"""

import numpy as np
import pytest

import bazalt as bz


def test_every_error_shares_one_base(ctx):
    for exc in (bz.InitializationError, bz.DeviceLostError, bz.OutOfMemoryError,
                bz.ShaderError, bz.WindowError, bz.ResourceError):
        assert issubclass(exc, bz.BazaltError)
    assert issubclass(bz.BazaltError, Exception)


def test_shader_error_carries_path_and_line(ctx, tmp_path):
    """Hot reload in 0.6 depends on this being catchable and locatable."""
    bad = tmp_path / "bad.frag"
    bad.write_text("#version 450\nvoid main() { not_a_real_symbol; }\n")

    with pytest.raises(bz.ShaderError) as info:
        ctx.compile_shader(str(bad), bz.ShaderStage.FRAGMENT)

    assert info.value.path == str(bad)
    assert info.value.line == 2


def test_shader_error_is_recoverable(ctx, tmp_path, triangle_shaders):
    """A typo must not end the process — that is the whole point of hot reload."""
    bad = tmp_path / "typo.frag"
    bad.write_text("#version 450\nvoid main() { oops }\n")

    try:
        ctx.compile_shader(str(bad), bz.ShaderStage.FRAGMENT)
    except bz.ShaderError:
        pass

    # Still usable afterwards.
    target = bz.RenderTarget(ctx, 16, 16)
    assert target.width == 16


def test_missing_shader_file_is_a_resource_error_not_a_shader_error(ctx):
    """A file that isn't there is a different problem from one that won't compile."""
    with pytest.raises(bz.ResourceError):
        ctx.compile_shader("does_not_exist.frag", bz.ShaderStage.VERTEX)


def test_missing_image_reports_why(ctx):
    with pytest.raises(bz.ResourceError) as info:
        ctx.load_image("no_such_image.png")
    # stb knows whether it was missing or corrupt; the message should say.
    assert "no_such_image.png" in str(info.value)


def test_invalid_validation_mode_names_the_valid_ones():
    with pytest.raises(ValueError) as info:
        bz.Context(validation="nonsense")
    for mode in ("auto", "on", "off"):
        assert mode in str(info.value)


def test_second_live_context_is_refused(ctx):
    """volk's function pointers are global, so a second Context corrupts the first.

    This used to be an access violation with no diagnostic.
    """
    with pytest.raises(bz.InitializationError) as info:
        bz.Context(validation="off")
    assert "one bazalt Context" in str(info.value)


def test_empty_buffer_list_is_rejected(ctx):
    with pytest.raises(bz.ResourceError):
        ctx.create_buffer([], bz.BufferType.VERTEX, bz.MemoryUsage.STATIC)


def test_descriptor_pool_needs_at_least_one_descriptor(ctx):
    with pytest.raises(bz.ResourceError):
        ctx.create_descriptor_pool(max_sets=1)


def test_pipeline_without_shaders_is_a_shader_error(ctx):
    target = bz.RenderTarget(ctx, 16, 16)
    with pytest.raises(bz.ShaderError):
        ctx.graphics_pipeline().build(target)


def test_static_buffer_update_is_a_resource_error(ctx):
    """update() on a STATIC buffer used to raise a bare RuntimeError, invisible
    to `except bz.BazaltError`."""
    buf = ctx.create_buffer([1.0, 2.0, 3.0], bz.BufferType.VERTEX, bz.MemoryUsage.STATIC)
    with pytest.raises(bz.ResourceError) as info:
        buf.update([4.0, 5.0, 6.0])
    assert "DYNAMIC" in str(info.value)


def test_oversized_dynamic_update_is_a_resource_error(ctx):
    """The message must name both sizes, or the user is left guessing."""
    buf = ctx.create_buffer([0.0, 0.0, 0.0, 0.0], bz.BufferType.UNIFORM,
                            bz.MemoryUsage.DYNAMIC)  # 16 bytes
    with pytest.raises(bz.ResourceError) as info:
        buf.update([0.0] * 16)  # 64 bytes
    assert "64" in str(info.value) and "16" in str(info.value)


def test_set_buffer_on_nonexistent_binding_is_a_resource_error(ctx, triangle_shaders):
    """A typo'd binding index used to be silently *assumed* to be a uniform
    buffer, producing a descriptor write the layout never declared."""
    vert, frag = triangle_shaders
    target = bz.RenderTarget(ctx, 16, 16)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                .uniform_buffer(0, bz.ShaderStage.FRAGMENT, set=0)
                .build(target))
    pool = ctx.create_descriptor_pool(max_sets=4, uniform_buffers=4)
    dset = pool.allocate_set(pipeline, set=0)
    ubuf = ctx.create_buffer([0.0] * 4, bz.BufferType.UNIFORM, bz.MemoryUsage.STATIC)

    with pytest.raises(bz.ResourceError) as info:
        dset.set_buffer(5, ubuf)
    assert "5" in str(info.value)


def test_set_buffer_on_image_binding_points_to_set_image(ctx, triangle_shaders):
    vert, frag = triangle_shaders
    target = bz.RenderTarget(ctx, 16, 16)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                .texture(0, bz.ShaderStage.FRAGMENT, set=0)
                .build(target))
    pool = ctx.create_descriptor_pool(max_sets=4, samplers=4)
    dset = pool.allocate_set(pipeline, set=0)
    ubuf = ctx.create_buffer([0.0] * 4, bz.BufferType.UNIFORM, bz.MemoryUsage.STATIC)

    with pytest.raises(bz.ResourceError) as info:
        dset.set_buffer(0, ubuf)
    assert "set_image" in str(info.value)


def test_dynamic_buffer_in_static_set_is_a_resource_error(ctx, triangle_shaders):
    """A DYNAMIC buffer has one backing buffer per frame; a static set can only
    point at one of them. The error must steer towards allocate_frame_set."""
    vert, frag = triangle_shaders
    target = bz.RenderTarget(ctx, 16, 16)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                .uniform_buffer(0, bz.ShaderStage.FRAGMENT, set=0)
                .build(target))
    pool = ctx.create_descriptor_pool(max_sets=4, uniform_buffers=4)
    static_set = pool.allocate_set(pipeline, set=0)
    dynamic = ctx.create_buffer([0.0] * 4, bz.BufferType.UNIFORM, bz.MemoryUsage.DYNAMIC)

    with pytest.raises(bz.ResourceError) as info:
        static_set.set_buffer(0, dynamic)
    assert "allocate_frame_set" in str(info.value)
