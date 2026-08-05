"""A buffer reached by its address instead of a descriptor (0.26).

The reason the feature exists is a limit: `ctx.limits.max_storage_buffer` caps
what one descriptor may see, and an address is not a descriptor. So the tests
that matter are "the shader reads the right bytes through a pointer" and "the
address survives the round trip through a push constant".
"""

import struct

import numpy as np
import pytest

import bazalt as bz

READ_SHADER = """
#version 450
#extension GL_EXT_buffer_reference : require
layout(local_size_x = 64) in;

layout(buffer_reference, std430) readonly buffer Words { uint v[]; };
layout(buffer_reference, std430) buffer Out { uint v[]; };
layout(push_constant, std430) uniform PC { Words src; Out dst; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    dst.v[i] = src.v[i] * 2u + 1u;
}
"""


@pytest.fixture
def address_ctx(extra_context):
    """A Context with the feature, or a skip when the GPU has not got it."""
    context = extra_context(optional=[bz.Feature.BUFFER_ADDRESS])
    if not context.supports(bz.Feature.BUFFER_ADDRESS):
        pytest.skip("this GPU has no bufferDeviceAddress")
    return context


def test_address_is_not_null(address_ctx):
    buf = address_ctx.create_buffer(
        np.arange(64, dtype=np.uint32), bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    assert buf.address != 0
    # Stable: it is where the buffer IS, not a token handed out per call.
    assert buf.address == buf.address


def test_two_buffers_have_different_addresses(address_ctx):
    a = address_ctx.create_buffer(1024, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    b = address_ctx.create_buffer(1024, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    assert a.address != b.address


def test_shader_reads_through_the_address(address_ctx):
    """The whole feature, end to end: no descriptor set anywhere in here."""
    n = 256
    pipeline = (address_ctx.compute_pipeline()
                .shader(address_ctx.compile_shader(source=READ_SHADER, stage=bz.ShaderStage.COMPUTE))
                .push_constant(16)
                .build())

    src = address_ctx.create_buffer(
        np.arange(n, dtype=np.uint32), bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    dst = address_ctx.create_buffer(
        np.zeros(n, dtype=np.uint32), bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)

    cmd = address_ctx.create_command_buffer()
    cmd.begin()
    (cmd.bind_pipeline(pipeline)
        .push_constants(pipeline, 0, struct.pack("<QQ", src.address, dst.address))
        .dispatch(n // 64))
    address_ctx.submit(cmd)

    assert np.array_equal(dst.read(np.uint32), np.arange(n, dtype=np.uint32) * 2 + 1)


def test_a_shader_may_use_64_bit_integers(extra_context):
    """Feature.SHADER_INT64, which is a separate row from the address on purpose.

    Passing an address around needs no 64-bit arithmetic — the pointer type
    hides it — but declaring a uint64_t does, and a shader that counts past 4
    billion elements wants one.
    """
    context = extra_context(optional=[bz.Feature.SHADER_INT64])
    if not context.supports(bz.Feature.SHADER_INT64):
        pytest.skip("this GPU has no shaderInt64")

    source = """
    #version 450
    #extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
    layout(local_size_x = 1) in;
    layout(set = 0, binding = 0, std430) buffer Out { uint v[]; };
    layout(push_constant, std430) uniform PC { uint64_t big; };
    void main() {
        // Only correct if the arithmetic really happened at 64 bits.
        v[0] = uint(big >> 32);
        v[1] = uint(big & 0xFFFFFFFFul);
    }
    """
    pipeline = (context.compute_pipeline()
                .shader(context.compile_shader(source=source, stage=bz.ShaderStage.COMPUTE))
                .storage_buffer(0)
                .push_constant(8)
                .build())

    out = context.create_buffer(
        np.zeros(4, dtype=np.uint32), bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    bound = context.create_descriptor_pool().allocate_set(pipeline)
    bound.set_buffer(0, out)

    value = (12345 << 32) | 67890
    cmd = context.create_command_buffer()
    cmd.begin()
    (cmd.bind_pipeline(pipeline)
        .bind_descriptor_set(bound, pipeline)
        .push_constants(pipeline, 0, struct.pack("<Q", value))
        .dispatch(1))
    context.submit(cmd)

    got = out.read(np.uint32)
    assert (int(got[0]), int(got[1])) == (12345, 67890)


def test_a_dynamic_buffer_has_an_address_too(address_ctx):
    buf = address_ctx.create_buffer(256, bz.BufferType.STORAGE, bz.MemoryUsage.DYNAMIC)
    assert buf.address != 0


def test_address_without_the_feature_raises(extra_context):
    """The flag is set at creation, so a Context without it can never answer."""
    plain = extra_context()
    buf = plain.create_buffer(256, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    with pytest.raises(bz.UnsupportedError, match="BUFFER_ADDRESS"):
        buf.address


def test_a_buffer_larger_than_one_staging_chunk_uploads_whole(ctx):
    """The upload is cut into 64 MiB pieces, so the seams are what can break.

    Not about the address at all — it is the other half of the same release,
    and it is what stops a big buffer from needing as much host memory as GPU
    memory. 160 MiB crosses two seams.
    """
    words = 40 * 1024 * 1024  # 160 MiB of uint32
    data = np.arange(words, dtype=np.uint32)
    buf = ctx.create_buffer(data, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)

    got = buf.read(np.uint32)
    assert len(got) == words
    # Check the piece boundaries specifically: a wrong dstOffset shows up here
    # and nowhere in the first megabyte.
    chunk_words = (64 * 1024 * 1024) // 4
    for seam in (chunk_words - 1, chunk_words, 2 * chunk_words - 1, 2 * chunk_words, words - 1):
        assert got[seam] == seam, f"word {seam} came back as {got[seam]}"
