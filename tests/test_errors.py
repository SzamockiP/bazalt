"""The exception type is the contract for whether you can carry on.

Everything used to arrive as a bare RuntimeError, so a typo in a shader was
indistinguishable from a lost device and neither could be handled.
"""

import numpy as np
import pytest

import bazalt as bz
from conftest import SHADER_DIR


def test_every_error_shares_one_base(ctx):
    for exc in (bz.InitializationError, bz.DeviceLostError, bz.OutOfMemoryError,
                bz.ShaderError, bz.WindowError, bz.ResourceError,
                bz.StateError, bz.UnsupportedError):
        assert issubclass(exc, bz.BazaltError)
    assert issubclass(bz.BazaltError, Exception)


def test_the_three_recoverable_kinds_are_siblings(ctx):
    """0.23 split StateError (fix the call order) and UnsupportedError (this
    GPU cannot) out of ResourceError (fix the data). An `except
    bz.ResourceError` written for data problems must not swallow the other
    two, so they are siblings, not subclasses."""
    for exc in (bz.StateError, bz.UnsupportedError):
        assert not issubclass(exc, bz.ResourceError)
        assert not issubclass(bz.ResourceError, exc)
    assert not issubclass(bz.StateError, bz.UnsupportedError)


def test_a_sequencing_error_is_a_state_error(ctx):
    """The same mistake spelled two ways gets the same type: a barrier inside
    a rendering scope is 'right call, wrong moment', not a resource fault."""
    buf = ctx.create_buffer(1024, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    target = bz.RenderTarget(ctx, 8, 8)
    cmd = ctx.create_command_buffer()
    cmd.begin().begin_rendering(target, clear_color=[0, 0, 0, 1])
    with pytest.raises(bz.StateError):
        cmd.barrier(buf, bz.Access.SHADER_WRITE, bz.Access.SHADER_READ)


def test_index_is_keyword_only_on_the_set_verbs(ctx):
    """set_image(0, img, 3) used to read as 'index 3' and pass 3 as a sampler.
    Since 0.23 index must be spelled, on all three verbs — one rule."""
    comp = ctx.compile_shader(str(SHADER_DIR / "double.comp"), bz.ShaderStage.COMPUTE)
    pipe = ctx.compute_pipeline().shader(comp).storage_buffer(0).build()
    pool = ctx.create_descriptor_pool(max_sets=1, storage_buffers=1)
    dset = pool.allocate_set(pipe, set=0)
    buf = ctx.create_buffer(64, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    dset.set_buffer(0, buf, index=0)
    with pytest.raises(TypeError):
        dset.set_buffer(0, buf, 0)


def test_create_buffer_takes_data_as_a_keyword(ctx):
    """The first parameter is `data` on all three overloads since 0.23, so the
    keyword spelling works whichever body the argument picks."""
    by_list = ctx.create_buffer(data=[1.0, 2.0], type=bz.BufferType.VERTEX,
                                usage=bz.MemoryUsage.STATIC)
    by_array = ctx.create_buffer(data=np.zeros(4, np.float32),
                                 type=bz.BufferType.VERTEX,
                                 usage=bz.MemoryUsage.STATIC)
    by_size = ctx.create_buffer(data=64, type=bz.BufferType.STORAGE,
                                usage=bz.MemoryUsage.STATIC)
    assert np.array_equal(by_array.read(np.float32), np.zeros(4, np.float32))
    assert by_list is not None and by_size is not None


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


def test_second_live_context_is_allowed(ctx, extra_context):
    """The inverse of what this asserted until 0.15.

    volk's function pointers are process globals, so a second Context used to be
    refused outright — without the guard it silently redirected the first one's
    GPU calls at its own device. Per-Context dispatch tables removed the reason,
    so the guard went with it. The rest of the story is in test_multi_context.py.
    """
    second = extra_context()
    assert second.device_name
    assert ctx.device_name


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


def test_read_and_update_agree_on_the_exception_type(ctx):
    """The same mistake on the same object got two different exception types
    until 0.20: read(layer=) raised bz.ResourceError from the C++ core and
    update(layer=) raised ValueError from the binding lambda. So did the
    C-contiguous rule, which Buffer.update reported as ResourceError and
    Image.update as ValueError."""
    img = ctx.create_image(np.zeros((4, 4, 4), dtype=np.uint8))
    pixels = np.zeros((4, 4, 4), dtype=np.uint8)

    with pytest.raises(bz.ResourceError):
        img.read(layer=3)
    with pytest.raises(bz.ResourceError):
        img.update(pixels, layer=3)

    strided = np.zeros((4, 8, 4), dtype=np.uint8)[:, ::2]
    with pytest.raises(bz.ResourceError):
        img.update(strided)
    buf = ctx.create_buffer([0.0] * 8, bz.BufferType.UNIFORM, bz.MemoryUsage.DYNAMIC)
    with pytest.raises(bz.ResourceError):
        buf.update(np.zeros((2, 4), dtype=np.float32)[:, ::2])


def test_a_malformed_argument_is_still_a_ValueError(ctx):
    """The other half of the rule: an argument wrong on its own stays out of the
    BazaltError hierarchy, so `except bz.BazaltError` never hides a typo."""
    img = ctx.create_image(np.zeros((4, 4, 4), dtype=np.uint8))

    with pytest.raises(ValueError):
        img.update(np.zeros((4, 4, 4), dtype=np.uint8), region=(0, 0, 4))
    with pytest.raises(ValueError):
        bz.Context(frames_in_flight=9)
    with pytest.raises(ValueError):
        bz.Context(validation="nonsense")
    assert not issubclass(bz.BazaltError, ValueError)
