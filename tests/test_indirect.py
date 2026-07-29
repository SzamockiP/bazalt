"""Indirect draw and dispatch (0.19).

The arguments come out of a storage buffer the GPU can write, so a compute pass
decides what gets drawn and the CPU never learns the answer. That is the whole
feature: culling, LOD selection and particle compaction stop needing a readback
between the pass that decides and the draw that obeys.

The interesting test is the first one, and what makes it a real test rather than a
smoke test is that nothing reads the argument buffer back. `stripe.vert` paints one
stripe per instance, so the painted pixel count is exactly proportional to the
`instanceCount` the compute shader accumulated — checked against a CPU-side count
of the same candidates. A draw that ignored the buffer, or a compute pass that
wrote the wrong word, changes that number.

bazalt declares no struct type for the arguments. numpy already writes
VkDrawIndirectCommand, a std430 GLSL struct is byte-identical to it, and a dtype
that exists only to be converted fails the scope test's second question.
"""

import struct

import numpy as np
import pytest

import bazalt as bz

from conftest import SHADER_DIR

# 8 stripes across a 64-wide target, so each instance paints exactly 8x64 = 512
# pixels and the counts below are exact rather than approximate.
TOTAL_STRIPES = 8
TARGET = 64
STRIPE_PIXELS = (TARGET // TOTAL_STRIPES) * TARGET

# VkDrawIndirectCommand is 4 uint32s; the buffer holds one of them.
DRAW_ARGS_BYTES = 16


def painted(target):
    px = target.color[0].read()
    return int(np.count_nonzero(px[:, :, :3].any(axis=2)))


def args_buffer(ctx, size=DRAW_ARGS_BYTES):
    """A storage buffer for draw arguments. STORAGE is the only type that carries
    the indirect usage flag, which is also the type a compute shader needs it to
    be."""
    return ctx.create_buffer(size, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)


# ── the feature ───────────────────────────────────────────────────────────────

def test_compute_decides_how_many_instances_are_drawn(ctx):
    """The whole chain in one recording: fill_buffer zeroes the arguments, a
    compute pass accumulates instanceCount atomically, and draw_indirect draws
    exactly that many stripes.

    No cmd.barrier() anywhere. The compute write and the command processor's read
    are ordered by the automatic tracker, and DRAW_INDIRECT is earlier than any
    shader stage — so getting that barrier wrong is a wrong picture, not a slow one.
    """
    scores = np.array([0.9, 0.1, 0.8, 0.7, 0.05, 0.95, 0.2, 0.6], dtype=np.float32)
    threshold = 0.5
    expected = int((scores > threshold).sum())
    assert 0 < expected < TOTAL_STRIPES  # the test is pointless at either extreme

    args = args_buffer(ctx)
    candidates = ctx.create_buffer(scores, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)

    comp = ctx.compile_shader(str(SHADER_DIR / "cull_args.comp"), bz.ShaderStage.COMPUTE)
    cull = (ctx.compute_pipeline()
            .shader(comp)
            .storage_buffer(0, set=0)
            .storage_buffer(1, set=0)
            .push_constant(8)
            .build())

    vert = ctx.compile_shader(str(SHADER_DIR / "stripe.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, TARGET, TARGET)
    draw = (ctx.graphics_pipeline()
            .vertex_shader(vert)
            .fragment_shader(frag)
            .push_constant(4, bz.ShaderStage.VERTEX)
            .build(target))

    pool = ctx.create_descriptor_pool(max_sets=4, storage_buffers=8)
    dset = pool.allocate_set(cull, set=0)
    dset.set_buffer(0, args)
    dset.set_buffer(1, candidates)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    # The prerequisite that landed in 0.18 for exactly this: the counter has to
    # start each frame at a known value, and saying so used to need a dispatch
    # whose whole body was an assignment.
    cmd.fill_buffer(args, 0)
    cmd.bind_pipeline(cull)
    cmd.bind_descriptor_set(dset, cull, set=0)
    cmd.push_constants(cull, 0, struct.pack("If", len(scores), threshold))
    cmd.dispatch(1)
    with cmd.rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0]) as c:
        c.bind_pipeline(draw)
        c.push_constants(draw, 0, struct.pack("I", TOTAL_STRIPES))
        c.draw_indirect(args)
    ctx.submit(cmd)
    ctx.wait()

    assert painted(target) == expected * STRIPE_PIXELS

    # And the buffer really does hold what the GPU decided, so the pixel count
    # above is not passing for some unrelated reason.
    written = args.read(np.uint32)
    assert written[0] == 6            # vertexCount
    assert written[1] == expected     # instanceCount


def test_a_zero_instance_count_draws_nothing(ctx):
    """The GPU-side way to say "draw nothing" is instanceCount = 0, which is why
    count=0 on the verb is refused instead: the two would be a second spelling, and
    only one of them can be decided on the GPU."""
    scores = np.zeros(8, dtype=np.float32)
    args = args_buffer(ctx)
    candidates = ctx.create_buffer(scores, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)

    comp = ctx.compile_shader(str(SHADER_DIR / "cull_args.comp"), bz.ShaderStage.COMPUTE)
    cull = (ctx.compute_pipeline().shader(comp)
            .storage_buffer(0, set=0).storage_buffer(1, set=0).push_constant(8).build())

    vert = ctx.compile_shader(str(SHADER_DIR / "stripe.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, TARGET, TARGET)
    draw = (ctx.graphics_pipeline().vertex_shader(vert).fragment_shader(frag)
            .push_constant(4, bz.ShaderStage.VERTEX).build(target))

    pool = ctx.create_descriptor_pool(max_sets=4, storage_buffers=8)
    dset = pool.allocate_set(cull, set=0)
    dset.set_buffer(0, args)
    dset.set_buffer(1, candidates)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.fill_buffer(args, 0)
    cmd.bind_pipeline(cull)
    cmd.bind_descriptor_set(dset, cull, set=0)
    cmd.push_constants(cull, 0, struct.pack("If", len(scores), 0.5))
    cmd.dispatch(1)
    with cmd.rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0]) as c:
        c.bind_pipeline(draw)
        c.push_constants(draw, 0, struct.pack("I", TOTAL_STRIPES))
        c.draw_indirect(args)
    ctx.submit(cmd)
    ctx.wait()

    assert painted(target) == 0


def test_cpu_written_arguments_work_too(ctx):
    """Nothing about the verb requires a compute pass: arguments written from
    Python are the ordinary case for a mesh whose instance count the CPU knows, and
    they prove numpy's layout matches VkDrawIndirectCommand without a bazalt dtype.
    """
    instances = 3
    args = ctx.create_buffer(
        np.array([6, instances, 0, 0], dtype=np.uint32),
        bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)

    vert = ctx.compile_shader(str(SHADER_DIR / "stripe.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, TARGET, TARGET)
    draw = (ctx.graphics_pipeline().vertex_shader(vert).fragment_shader(frag)
            .push_constant(4, bz.ShaderStage.VERTEX).build(target))

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0]) as c:
        c.bind_pipeline(draw)
        c.push_constants(draw, 0, struct.pack("I", TOTAL_STRIPES))
        c.draw_indirect(args)
    ctx.submit(cmd)
    ctx.wait()

    assert painted(target) == instances * STRIPE_PIXELS


def test_dispatch_indirect_takes_its_group_count_from_the_gpu(ctx):
    """Two compute passes: the first writes the group count, the second runs with
    it. bump_counter.comp increments once per invocation at local_size 1, so the
    counter IS the group count the second dispatch ran with."""
    want = 5
    groups = ctx.create_buffer(12, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    counter = ctx.create_buffer(4, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)

    writer_shader = ctx.compile_shader(str(SHADER_DIR / "write_groups.comp"), bz.ShaderStage.COMPUTE)
    writer = (ctx.compute_pipeline().shader(writer_shader)
              .storage_buffer(0, set=0).push_constant(4).build())
    bump_shader = ctx.compile_shader(str(SHADER_DIR / "bump_counter.comp"), bz.ShaderStage.COMPUTE)
    bump = ctx.compute_pipeline().shader(bump_shader).storage_buffer(0, set=0).build()

    pool = ctx.create_descriptor_pool(max_sets=4, storage_buffers=8)
    writer_set = pool.allocate_set(writer, set=0)
    writer_set.set_buffer(0, groups)
    bump_set = pool.allocate_set(bump, set=0)
    bump_set.set_buffer(0, counter)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.fill_buffer(counter, 0)
    cmd.bind_pipeline(writer)
    cmd.bind_descriptor_set(writer_set, writer, set=0)
    cmd.push_constants(writer, 0, struct.pack("I", want))
    cmd.dispatch(1)
    cmd.bind_pipeline(bump)
    cmd.bind_descriptor_set(bump_set, bump, set=0)
    cmd.dispatch_indirect(groups)
    ctx.submit(cmd)
    ctx.wait()

    assert int(counter.read(np.uint32)[0]) == want


def test_draw_indexed_indirect_uses_the_index_buffer(ctx):
    """The indexed twin, with its own 20-byte argument struct — and note
    vertexOffset is SIGNED, which is why the layout is documented rather than
    guessed."""
    instances = 2
    # indexCount, instanceCount, firstIndex, vertexOffset (int32), firstInstance
    args = ctx.create_buffer(
        np.array([6, instances, 0, 0, 0], dtype=np.uint32),
        bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    indices = ctx.create_buffer(
        np.array([0, 1, 2, 3, 4, 5], dtype=np.uint32),
        bz.BufferType.INDEX, bz.MemoryUsage.STATIC)

    vert = ctx.compile_shader(str(SHADER_DIR / "stripe.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, TARGET, TARGET)
    draw = (ctx.graphics_pipeline().vertex_shader(vert).fragment_shader(frag)
            .push_constant(4, bz.ShaderStage.VERTEX).build(target))

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0]) as c:
        c.bind_pipeline(draw)
        c.bind_index_buffer(indices)
        c.push_constants(draw, 0, struct.pack("I", TOTAL_STRIPES))
        c.draw_indexed_indirect(args)
    ctx.submit(cmd)
    ctx.wait()

    assert painted(target) == instances * STRIPE_PIXELS


# ── what it refuses, and why ──────────────────────────────────────────────────

def test_only_a_storage_buffer_can_hold_the_arguments(ctx):
    """Only BufferType.STORAGE carries VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, so the
    refusal names the fix instead of leaving the layers to report a usage flag. A
    compute shader writing the arguments needs a storage buffer anyway."""
    uniform = ctx.create_buffer(64, bz.BufferType.UNIFORM, bz.MemoryUsage.STATIC)
    cmd = ctx.create_command_buffer()
    cmd.begin()
    with pytest.raises(bz.ResourceError, match="BufferType.STORAGE"):
        cmd.dispatch_indirect(uniform)


def test_the_offset_must_be_aligned_and_in_range(ctx):
    args = args_buffer(ctx)
    cmd = ctx.create_command_buffer()
    cmd.begin()

    with pytest.raises(bz.ResourceError, match="multiple of 4"):
        cmd.dispatch_indirect(args, offset=2)
    # 16-byte buffer, one 16-byte struct: offset 4 leaves only 12 bytes.
    with pytest.raises(bz.ResourceError, match="buffer is 16"):
        cmd.draw_indirect(args, offset=4)
    with pytest.raises(bz.ResourceError, match="at least 1"):
        cmd.draw_indirect(args, count=0)


def test_multi_draw_needs_its_feature(ctx, extra_context):
    """count>1 is multiDrawIndirect, a feature bit rather than free core Vulkan —
    the fourth release in a row where something that looks like plain command
    recording turns out to have one. This is also the release that finally gives
    Feature.MULTI_DRAW_INDIRECT an API to be reachable through: it has been in the
    table since 0.5 with nothing behind it.
    """
    # Room for two argument structs, so the failure is the feature and not the size.
    args = args_buffer(ctx, size=DRAW_ARGS_BYTES * 2)
    cmd = ctx.create_command_buffer()
    cmd.begin()
    if ctx.supports(bz.Feature.MULTI_DRAW_INDIRECT):
        pytest.skip("the session Context has MULTI_DRAW_INDIRECT, so it cannot refuse")
    with pytest.raises(bz.ResourceError, match="MULTI_DRAW_INDIRECT"):
        cmd.draw_indirect(args, count=2)


def test_multi_draw_works_with_the_feature(ctx, extra_context):
    """Two argument structs in one call, on a Context that asked for the feature."""
    multi = extra_context(optional=[bz.Feature.MULTI_DRAW_INDIRECT])
    if not multi.supports(bz.Feature.MULTI_DRAW_INDIRECT):
        pytest.skip("GPU reports no multiDrawIndirect")

    # Two commands: 1 instance then 2, so 3 stripes total. A single-command draw
    # would paint 1 and a wrong stride would paint garbage.
    args = multi.create_buffer(
        np.array([6, 1, 0, 0,
                  6, 2, 0, 0], dtype=np.uint32),
        bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)

    vert = multi.compile_shader(str(SHADER_DIR / "stripe.vert"), bz.ShaderStage.VERTEX)
    frag = multi.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(multi, TARGET, TARGET)
    draw = (multi.graphics_pipeline().vertex_shader(vert).fragment_shader(frag)
            .push_constant(4, bz.ShaderStage.VERTEX).build(target))

    cmd = multi.create_command_buffer()
    cmd.begin()
    with cmd.rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0]) as c:
        c.bind_pipeline(draw)
        c.push_constants(draw, 0, struct.pack("I", TOTAL_STRIPES))
        c.draw_indirect(args, count=2)
    multi.submit(cmd)
    multi.wait()

    # Both commands draw instances starting at 0, so the second's 2 stripes overlap
    # the first's 1: the union is stripes 0 and 1.
    assert painted(target) == 2 * STRIPE_PIXELS


def test_a_buffer_from_another_context_is_refused(ctx, extra_context):
    other = extra_context()
    foreign = other.create_buffer(16, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    cmd = ctx.create_command_buffer()
    cmd.begin()
    with pytest.raises(bz.ResourceError, match="different Context"):
        cmd.dispatch_indirect(foreign)


def test_indirect_read_is_a_manual_barrier_access(ctx):
    """Access.INDIRECT_READ exists so manual mode can express the same hazard the
    tracker handles: it is a buffer access, so cmd.barrier(image, INDIRECT_READ)
    gets the existing buffer-only message rather than a layout nobody asked for."""
    args = args_buffer(ctx)
    img = ctx.create_image(16, 16, bz.Format.RGBA8)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.barrier(args, bz.Access.SHADER_WRITE, bz.Access.INDIRECT_READ)
    with pytest.raises(bz.ResourceError, match="apply to buffers only"):
        cmd.barrier(img, bz.Access.SHADER_WRITE, bz.Access.INDIRECT_READ)
    ctx.submit(cmd)
    ctx.wait()
