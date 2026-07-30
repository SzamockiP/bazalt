"""Buffer uploads, and the stride bug that used to corrupt them silently."""

import numpy as np
import pytest

import bazalt as bz

from conftest import SHADER_DIR


def test_contiguous_array_uploads(ctx):
    data = np.arange(12, dtype=np.float32)
    buf = ctx.create_buffer(data, bz.BufferType.VERTEX, bz.MemoryUsage.STATIC)
    assert buf is not None


@pytest.mark.parametrize("make_strided, label", [
    (lambda: np.arange(24, dtype=np.float32).reshape(4, 6).T, "transposed"),
    (lambda: np.arange(20, dtype=np.float32)[::2], "every other element"),
    (lambda: np.arange(36, dtype=np.float32).reshape(6, 6)[:, ::2], "strided columns"),
])
def test_strided_arrays_are_rejected_not_silently_mangled(ctx, make_strided, label):
    """create_buffer used to memcpy size*itemsize bytes and ignore strides.

    A transposed or sliced view therefore uploaded whatever happened to be
    adjacent in memory, with no error at all.
    """
    arr = make_strided()
    assert not arr.flags["C_CONTIGUOUS"], f"{label} should not be contiguous"

    with pytest.raises(bz.ResourceError) as info:
        ctx.create_buffer(arr, bz.BufferType.VERTEX, bz.MemoryUsage.STATIC)
    assert "contiguous" in str(info.value)


def test_error_message_suggests_the_fix(ctx):
    arr = np.arange(24, dtype=np.float32).reshape(4, 6).T
    with pytest.raises(bz.ResourceError) as info:
        ctx.create_buffer(arr, bz.BufferType.VERTEX, bz.MemoryUsage.STATIC)
    assert "ascontiguousarray" in str(info.value)

    # And that suggestion actually works.
    ctx.create_buffer(np.ascontiguousarray(arr), bz.BufferType.VERTEX, bz.MemoryUsage.STATIC)


def test_size_one_dimensions_are_not_false_positives(ctx):
    """A dimension of extent 1 has an arbitrary stride; numpy fills in junk.

    Comparing it against the packed layout reports non-contiguity that isn't real.
    """
    arr = np.zeros((1, 8), dtype=np.float32)
    ctx.create_buffer(arr, bz.BufferType.VERTEX, bz.MemoryUsage.STATIC)


def test_strided_update_is_rejected(ctx):
    buf = ctx.create_buffer(64, bz.BufferType.UNIFORM, bz.MemoryUsage.DYNAMIC)
    arr = np.arange(32, dtype=np.float32).reshape(4, 8).T
    with pytest.raises(bz.ResourceError):
        buf.update(arr)


def test_list_overload_still_wins_over_the_buffer_overload(ctx):
    """A Python list must not be captured by the array overload.

    numpy would convert it to float64, so vertex data would silently double in
    size and be misread by the shader.
    """
    buf = ctx.create_buffer([1.0, 2.0, 3.0, 4.0], bz.BufferType.VERTEX,
                            bz.MemoryUsage.DYNAMIC, bz.DataType.FLOAT)
    # 4 float32s, not 4 float64s.
    assert buf is not None
    buf.update([5.0, 6.0, 7.0, 8.0])


@pytest.mark.parametrize("dtype", [bz.DataType.FLOAT, bz.DataType.UINT32,
                                   bz.DataType.UINT16, bz.DataType.INT32])
def test_every_data_type_can_be_created_and_updated(ctx, dtype):
    """UINT16 could be created but not updated — the update path threw."""
    buf = ctx.create_buffer([1, 2, 3] if dtype != bz.DataType.FLOAT else [1.0, 2.0, 3.0],
                            bz.BufferType.STORAGE, bz.MemoryUsage.DYNAMIC, dtype)
    buf.update([4, 5, 6] if dtype != bz.DataType.FLOAT else [4.0, 5.0, 6.0], dtype)


def test_sized_buffer_without_data(ctx):
    buf = ctx.create_buffer(256, bz.BufferType.UNIFORM, bz.MemoryUsage.DYNAMIC)
    assert buf is not None


def test_bytes_upload(ctx):
    buf = ctx.create_buffer(64, bz.BufferType.UNIFORM, bz.MemoryUsage.DYNAMIC)
    buf.update(b"\x00" * 64)


# ── read() — 0.5 ──────────────────────────────────────────────────────────


def test_static_buffer_reads_back_what_was_uploaded(ctx):
    """STATIC read() is a GPU round trip: device-local memory is not mappable."""
    data = np.arange(16, dtype=np.float32)
    buf = ctx.create_buffer(data, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    np.testing.assert_array_equal(buf.read(np.float32), data)


def test_dynamic_buffer_read_returns_the_latest_update(ctx):
    """DYNAMIC read() maps the current frame's copy — no GPU involved."""
    buf = ctx.create_buffer(16, bz.BufferType.UNIFORM, bz.MemoryUsage.DYNAMIC)
    payload = np.array([1.5, -2.0, 3.25, 0.0], dtype=np.float32)
    buf.update(payload)  # numpy straight in, no bytes(...) dance
    np.testing.assert_array_equal(buf.read(np.float32), payload)


def test_read_dtype_is_respected(ctx):
    data = np.arange(8, dtype=np.uint32)
    buf = ctx.create_buffer(data, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    out = buf.read(np.uint32)
    assert out.dtype == np.uint32
    np.testing.assert_array_equal(out, data)


# ── every type is usable in both memory usages (0.21) ─────────────────────────


@pytest.mark.parametrize("memory", [bz.MemoryUsage.STATIC, bz.MemoryUsage.DYNAMIC])
def test_a_vertex_buffer_binds_in_either_memory_usage(ctx, memory):
    """A DYNAMIC vertex buffer is geometry rebuilt every frame, which is what
    DYNAMIC is FOR — and it did not work: DynamicBuffer asked "is it STORAGE?"
    and gave everything else uniform-buffer usage, so binding one was
    VUID-vkCmdBindVertexBuffers-pBuffers-00627 and the draw read undefined data.

    Parametrized over both usages rather than testing the broken one alone,
    because the bug was the two paths disagreeing. The referee is the
    validation-as-assert fixture: a missing usage bit is an error there, not a
    wrong pixel."""
    vertices = np.array([
        -0.5, -0.5, 0.0, 1.0, 0.0, 0.0,
        -0.5, +0.5, 0.0, 0.0, 1.0, 0.0,
        +0.5, +0.5, 0.0, 0.0, 0.0, 1.0,
    ], dtype=np.float32)
    vbuf = ctx.create_buffer(vertices, bz.BufferType.VERTEX, memory)
    ibuf = ctx.create_buffer(np.array([0, 1, 2], dtype=np.uint32), bz.BufferType.INDEX, memory)

    vert = ctx.compile_shader(str(SHADER_DIR / "triangle.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "triangle.frag"), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, 32, 32)
    pipe = (ctx.graphics_pipeline()
            .vertex_shader(vert).fragment_shader(frag)
            .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
            .build(target))

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(target, clear_color=[0, 0, 0, 1]) as c:
        c.bind_pipeline(pipe).bind_vertex_buffer(vbuf).bind_index_buffer(ibuf).draw_indexed(3)
    ctx.submit(cmd)

    painted = int(np.count_nonzero(target.color[0].read()[:, :, :3].any(axis=2)))
    assert painted > 100, f"{memory} vertex buffer drew {painted} pixels"


@pytest.mark.parametrize("buffer_type", [bz.BufferType.VERTEX, bz.BufferType.INDEX,
                                         bz.BufferType.UNIFORM, bz.BufferType.STORAGE])
def test_both_memory_usages_accept_every_buffer_type(ctx, buffer_type):
    """The creation half of the same rule: no combination of type and memory
    usage is refused or silently given the wrong usage flags."""
    data = np.arange(16, dtype=np.uint32)
    for memory in (bz.MemoryUsage.STATIC, bz.MemoryUsage.DYNAMIC):
        buf = ctx.create_buffer(data, buffer_type, memory)
        # Read back rather than merely constructed: TRANSFER_SRC is part of the
        # usage every type gets, and a missing bit shows up here.
        assert buf.read(np.uint32).tolist() == data.tolist()
