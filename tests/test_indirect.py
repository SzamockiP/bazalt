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
    return ctx.create_buffer(size, bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)


# ── the feature ───────────────────────────────────────────────────────────────

def test_compute_decides_how_many_instances_are_drawn(ctx):
    """The whole chain in one graph: fill_buffer zeroes the arguments, a
    compute pass accumulates instanceCount atomically, and draw_indirect draws
    exactly that many stripes.

    No p.barrier() anywhere. The compute write and the command processor's read
    are ordered by the automatic tracker, and DRAW_INDIRECT is earlier than any
    shader stage — so getting that barrier wrong is a wrong picture, not a slow one.
    """
    scores = np.array([0.9, 0.1, 0.8, 0.7, 0.05, 0.95, 0.2, 0.6], dtype=np.float32)
    threshold = 0.5
    expected = int((scores > threshold).sum())
    assert 0 < expected < TOTAL_STRIPES  # the test is pointless at either extreme

    args = args_buffer(ctx)
    candidates = ctx.create_buffer(scores, bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)

    comp = ctx.compile_shader(str(SHADER_DIR / "cull_args.comp"), bz.ShaderStage.COMPUTE)
    cull = (ctx.compute_pipeline()
            .shader(comp)
            .storage_buffer(0, set=0)
            .storage_buffer(1, set=0)
            .push_constant(8)
            .build())

    vert = ctx.compile_shader(str(SHADER_DIR / "stripe.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    target = ctx.create_render_target(TARGET, TARGET)
    draw = (ctx.graphics_pipeline()
            .vertex_shader(vert)
            .fragment_shader(frag)
            .push_constant(4, bz.ShaderStage.VERTEX)
            .build(target))

    pool = ctx.create_descriptor_pool(max_sets=4, storage_buffers=8)
    dset = pool.allocate_set(cull, set=0)
    dset.set_buffer(0, args)
    dset.set_buffer(1, candidates)

    g = ctx.graph()
    with g.add_pass() as p:
        # The prerequisite that landed in 0.18 for exactly this: the counter has to
        # start each frame at a known value, and saying so used to need a dispatch
        # whose whole body was an assignment.
        p.fill_buffer(args, 0)
        p.bind_pipeline(cull)
        p.bind_descriptor_set(dset, cull, set=0)
        p.push_constants(cull, 0, struct.pack("If", len(scores), threshold))
        p.dispatch(1)
    with g.add_pass(target, clear_color=[0.0, 0.0, 0.0, 1.0]) as p:
        p.bind_pipeline(draw)
        p.push_constants(draw, 0, struct.pack("I", TOTAL_STRIPES))
        p.draw_indirect(args)
    ctx.submit(g)

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
    candidates = ctx.create_buffer(scores, bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)

    comp = ctx.compile_shader(str(SHADER_DIR / "cull_args.comp"), bz.ShaderStage.COMPUTE)
    cull = (ctx.compute_pipeline().shader(comp)
            .storage_buffer(0, set=0).storage_buffer(1, set=0).push_constant(8).build())

    vert = ctx.compile_shader(str(SHADER_DIR / "stripe.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    target = ctx.create_render_target(TARGET, TARGET)
    draw = (ctx.graphics_pipeline().vertex_shader(vert).fragment_shader(frag)
            .push_constant(4, bz.ShaderStage.VERTEX).build(target))

    pool = ctx.create_descriptor_pool(max_sets=4, storage_buffers=8)
    dset = pool.allocate_set(cull, set=0)
    dset.set_buffer(0, args)
    dset.set_buffer(1, candidates)

    g = ctx.graph()
    with g.add_pass() as p:
        p.fill_buffer(args, 0)
        p.bind_pipeline(cull)
        p.bind_descriptor_set(dset, cull, set=0)
        p.push_constants(cull, 0, struct.pack("If", len(scores), 0.5))
        p.dispatch(1)
    with g.add_pass(target, clear_color=[0.0, 0.0, 0.0, 1.0]) as p:
        p.bind_pipeline(draw)
        p.push_constants(draw, 0, struct.pack("I", TOTAL_STRIPES))
        p.draw_indirect(args)
    ctx.submit(g)

    assert painted(target) == 0


def test_cpu_written_arguments_work_too(ctx):
    """Nothing about the verb requires a compute pass: arguments written from
    Python are the ordinary case for a mesh whose instance count the CPU knows, and
    they prove numpy's layout matches VkDrawIndirectCommand without a bazalt dtype.
    """
    instances = 3
    args = ctx.create_buffer(
        np.array([6, instances, 0, 0], dtype=np.uint32),
        bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)

    vert = ctx.compile_shader(str(SHADER_DIR / "stripe.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    target = ctx.create_render_target(TARGET, TARGET)
    draw = (ctx.graphics_pipeline().vertex_shader(vert).fragment_shader(frag)
            .push_constant(4, bz.ShaderStage.VERTEX).build(target))

    g = ctx.graph()
    with g.add_pass(target, clear_color=[0.0, 0.0, 0.0, 1.0]) as p:
        p.bind_pipeline(draw)
        p.push_constants(draw, 0, struct.pack("I", TOTAL_STRIPES))
        p.draw_indirect(args)
    ctx.submit(g)

    assert painted(target) == instances * STRIPE_PIXELS


def test_dispatch_indirect_takes_its_group_count_from_the_gpu(ctx):
    """Two compute passes: the first writes the group count, the second runs with
    it. bump_counter.comp increments once per invocation at local_size 1, so the
    counter IS the group count the second dispatch ran with."""
    want = 5
    groups = ctx.create_buffer(12, bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    counter = ctx.create_buffer(4, bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)

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

    g = ctx.graph()
    p = g.add_pass()
    p.fill_buffer(counter, 0)
    p.bind_pipeline(writer)
    p.bind_descriptor_set(writer_set, writer, set=0)
    p.push_constants(writer, 0, struct.pack("I", want))
    p.dispatch(1)
    p.bind_pipeline(bump)
    p.bind_descriptor_set(bump_set, bump, set=0)
    p.dispatch_indirect(groups)
    ctx.submit(g)

    assert int(counter.read(np.uint32)[0]) == want


def test_draw_indexed_indirect_uses_the_index_buffer(ctx):
    """The indexed twin, with its own 20-byte argument struct — and note
    vertexOffset is SIGNED, which is why the layout is documented rather than
    guessed."""
    instances = 2
    # indexCount, instanceCount, firstIndex, vertexOffset (int32), firstInstance
    args = ctx.create_buffer(
        np.array([6, instances, 0, 0, 0], dtype=np.uint32),
        bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    indices = ctx.create_buffer(
        np.array([0, 1, 2, 3, 4, 5], dtype=np.uint32),
        bz.BufferUsage.INDEX, bz.MemoryUsage.STATIC)

    vert = ctx.compile_shader(str(SHADER_DIR / "stripe.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    target = ctx.create_render_target(TARGET, TARGET)
    draw = (ctx.graphics_pipeline().vertex_shader(vert).fragment_shader(frag)
            .push_constant(4, bz.ShaderStage.VERTEX).build(target))

    g = ctx.graph()
    with g.add_pass(target, clear_color=[0.0, 0.0, 0.0, 1.0]) as p:
        p.bind_pipeline(draw)
        p.bind_index_buffer(indices)
        p.push_constants(draw, 0, struct.pack("I", TOTAL_STRIPES))
        p.draw_indexed_indirect(args)
    ctx.submit(g)

    assert painted(target) == instances * STRIPE_PIXELS


# ── what it refuses, and why ──────────────────────────────────────────────────

def test_only_a_storage_buffer_can_hold_the_arguments(ctx):
    """Only BufferUsage.STORAGE carries VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, so the
    refusal names the fix instead of leaving the layers to report a usage flag. A
    compute shader writing the arguments needs a storage buffer anyway."""
    uniform = ctx.create_buffer(64, bz.BufferUsage.UNIFORM, bz.MemoryUsage.STATIC)
    g = ctx.graph()
    p = g.add_pass()
    with pytest.raises(bz.ResourceError, match="BufferUsage.STORAGE"):
        p.dispatch_indirect(uniform)


def test_the_offset_must_be_aligned_and_in_range(ctx):
    args = args_buffer(ctx)
    g = ctx.graph()
    p = g.add_pass()
    with pytest.raises(bz.ResourceError, match="multiple of 4"):
        p.dispatch_indirect(args, offset=2)

    # The draw verbs need a render pass.
    r = g.add_pass(ctx.create_render_target(8, 8))
    # 16-byte buffer, one 16-byte struct: offset 4 leaves only 12 bytes.
    with pytest.raises(bz.ResourceError, match="buffer is 16"):
        r.draw_indirect(args, offset=4)
    with pytest.raises(bz.ResourceError, match="at least 1"):
        r.draw_indirect(args, count=0)


def test_multi_draw_needs_its_feature(ctx, extra_context):
    """count>1 is multiDrawIndirect, a feature bit rather than free core Vulkan —
    the fourth release in a row where something that looks like plain command
    recording turns out to have one. This is also the release that finally gives
    Feature.MULTI_DRAW_INDIRECT an API to be reachable through: it has been in the
    table since 0.5 with nothing behind it.
    """
    # Room for two argument structs, so the failure is the feature and not the size.
    args = args_buffer(ctx, size=DRAW_ARGS_BYTES * 2)
    g = ctx.graph()
    p = g.add_pass(ctx.create_render_target(8, 8))
    if ctx.supports(bz.Feature.MULTI_DRAW_INDIRECT):
        pytest.skip("the session Context has MULTI_DRAW_INDIRECT, so it cannot refuse")
    with pytest.raises(bz.UnsupportedError, match="MULTI_DRAW_INDIRECT"):
        p.draw_indirect(args, count=2)


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
        bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)

    vert = multi.compile_shader(str(SHADER_DIR / "stripe.vert"), bz.ShaderStage.VERTEX)
    frag = multi.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    target = multi.create_render_target(TARGET, TARGET)
    draw = (multi.graphics_pipeline().vertex_shader(vert).fragment_shader(frag)
            .push_constant(4, bz.ShaderStage.VERTEX).build(target))

    g = multi.graph()
    with g.add_pass(target, clear_color=[0.0, 0.0, 0.0, 1.0]) as p:
        p.bind_pipeline(draw)
        p.push_constants(draw, 0, struct.pack("I", TOTAL_STRIPES))
        p.draw_indirect(args, count=2)
    multi.submit(g)

    # Both commands draw instances starting at 0, so the second's 2 stripes overlap
    # the first's 1: the union is stripes 0 and 1.
    assert painted(target) == 2 * STRIPE_PIXELS


def test_a_buffer_from_another_context_is_refused(ctx, extra_context):
    other = extra_context()
    foreign = other.create_buffer(16, bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    g = ctx.graph()
    p = g.add_pass()
    with pytest.raises(bz.ResourceError, match="different Context"):
        p.dispatch_indirect(foreign)


def test_indirect_read_is_a_manual_barrier_access(ctx):
    """Access.INDIRECT_READ exists so manual mode can express the same hazard the
    tracker handles: it is a buffer access, so p.barrier(image, INDIRECT_READ)
    gets the existing buffer-only message rather than a layout nobody asked for."""
    args = args_buffer(ctx)
    img = ctx.create_image(16, 16, bz.Format.RGBA8)

    g = ctx.graph()
    p = g.add_pass()
    p.barrier(args, bz.Access.SHADER_WRITE, bz.Access.INDIRECT_READ)
    with pytest.raises(bz.ResourceError, match="apply to buffers only"):
        p.barrier(img, bz.Access.SHADER_WRITE, bz.Access.INDIRECT_READ)
    ctx.submit(g)


# ── a GPU-decided draw COUNT (0.21) ───────────────────────────────────────────


def count_context(extra_context):
    """Both features: the count buffer needs DRAW_INDIRECT_COUNT, and `count`
    becomes maxDrawCount, which is still multiDrawIndirect when it is above 1."""
    ctx = extra_context(optional=[bz.Feature.DRAW_INDIRECT_COUNT,
                                  bz.Feature.MULTI_DRAW_INDIRECT])
    if not ctx.supports(bz.Feature.DRAW_INDIRECT_COUNT):
        pytest.skip("GPU reports no drawIndirectCount")
    if not ctx.supports(bz.Feature.MULTI_DRAW_INDIRECT):
        pytest.skip("GPU reports no multiDrawIndirect")
    return ctx


def stripe_pipeline(ctx, target):
    vert = ctx.compile_shader(str(SHADER_DIR / "stripe.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    return (ctx.graphics_pipeline().vertex_shader(vert).fragment_shader(frag)
            .push_constant(4, bz.ShaderStage.VERTEX).build(target))


def one_stripe_each(ctx, commands):
    """`commands` argument structs, command i painting stripe i alone.

    firstInstance is what separates them: gl_InstanceIndex counts from it, so
    command i draws exactly stripe i and the painted area is proportional to how
    many commands ran — which is the number under test."""
    words = []
    for i in range(commands):
        words += [6, 1, 0, i]  # vertexCount, instanceCount, firstVertex, firstInstance
    return ctx.create_buffer(np.array(words, dtype=np.uint32),
                             bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)


def count_buffer(ctx, value):
    return ctx.create_buffer(np.array([value], dtype=np.uint32),
                             bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)


@pytest.mark.parametrize("issued", [1, 3])
def test_the_count_buffer_decides_how_many_commands_run(extra_context, issued):
    """Two-sided by parametrization: the same four argument structs and the same
    maxDrawCount, differing only in the number sitting in the count buffer. A
    draw that ignored the buffer would paint four stripes both times."""
    ctx = count_context(extra_context)
    target = ctx.create_render_target(TARGET, TARGET)
    draw = stripe_pipeline(ctx, target)
    args = one_stripe_each(ctx, 4)
    counts = count_buffer(ctx, issued)

    g = ctx.graph()
    with g.add_pass(target, clear_color=[0.0, 0.0, 0.0, 1.0]) as p:
        p.bind_pipeline(draw)
        p.push_constants(draw, 0, struct.pack("I", TOTAL_STRIPES))
        p.draw_indirect(args, count=4, count_buffer=counts)
    ctx.submit(g)

    assert painted(target) == issued * STRIPE_PIXELS


def test_the_count_is_clamped_to_the_maximum(extra_context):
    """The buffer says 4 and maxDrawCount says 2, so two commands run. The spec
    clamps rather than reading past the arguments the caller sized for, which is
    also why check_indirect_ bounds-checks against `count` and not against the
    number in the buffer — the CPU never learns that one."""
    ctx = count_context(extra_context)
    target = ctx.create_render_target(TARGET, TARGET)
    draw = stripe_pipeline(ctx, target)
    args = one_stripe_each(ctx, 4)
    counts = count_buffer(ctx, 4)

    g = ctx.graph()
    with g.add_pass(target, clear_color=[0.0, 0.0, 0.0, 1.0]) as p:
        p.bind_pipeline(draw)
        p.push_constants(draw, 0, struct.pack("I", TOTAL_STRIPES))
        p.draw_indirect(args, count=2, count_buffer=counts)
    ctx.submit(g)

    assert painted(target) == 2 * STRIPE_PIXELS


def test_compute_writes_the_count(extra_context):
    """The point of the feature: nothing on the CPU knows how many draws happen.
    The same cull_args.comp accumulates a counter, and this time the counter IS
    the draw count rather than an instance count.

    No p.barrier(): the compute write and the command processor's read of the
    COUNT buffer are ordered by the tracker, which had to learn that the count
    buffer is read at DRAW_INDIRECT like the arguments are."""
    scores = np.array([0.9, 0.1, 0.8, 0.7, 0.05, 0.95, 0.2, 0.6], dtype=np.float32)
    threshold = 0.5
    expected = int((scores > threshold).sum())

    ctx = count_context(extra_context)
    counts = ctx.create_buffer(DRAW_ARGS_BYTES, bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    candidates = ctx.create_buffer(scores, bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)

    comp = ctx.compile_shader(str(SHADER_DIR / "cull_args.comp"), bz.ShaderStage.COMPUTE)
    cull = (ctx.compute_pipeline().shader(comp)
            .storage_buffer(0, set=0).storage_buffer(1, set=0)
            .push_constant(8).build())
    pool = ctx.create_descriptor_pool(max_sets=2, storage_buffers=4)
    dset = pool.allocate_set(cull, set=0)
    dset.set_buffer(0, counts)
    dset.set_buffer(1, candidates)

    target = ctx.create_render_target(TARGET, TARGET)
    draw = stripe_pipeline(ctx, target)
    args = one_stripe_each(ctx, TOTAL_STRIPES)

    g = ctx.graph()
    with g.add_pass() as p:
        p.fill_buffer(counts, 0)
        p.bind_pipeline(cull)
        p.bind_descriptor_set(dset, cull, set=0)
        p.push_constants(cull, 0, struct.pack("If", len(scores), threshold))
        p.dispatch(1)
    with g.add_pass(target, clear_color=[0.0, 0.0, 0.0, 1.0]) as p:
        p.bind_pipeline(draw)
        p.push_constants(draw, 0, struct.pack("I", TOTAL_STRIPES))
        # cull_args.comp accumulates into word 1 (instanceCount), so the count the
        # draw reads sits 4 bytes in.
        p.draw_indirect(args, count=TOTAL_STRIPES, count_buffer=counts, count_offset=4)
    ctx.submit(g)

    assert painted(target) == expected * STRIPE_PIXELS


# ── refusals ──────────────────────────────────────────────────────────────────


def test_a_count_buffer_needs_its_feature(ctx):
    if ctx.supports(bz.Feature.DRAW_INDIRECT_COUNT):
        pytest.skip("the session Context has DRAW_INDIRECT_COUNT, so it cannot refuse")
    args = args_buffer(ctx)
    counts = args_buffer(ctx)
    g = ctx.graph()
    p = g.add_pass(ctx.create_render_target(8, 8))
    with pytest.raises(bz.UnsupportedError, match="DRAW_INDIRECT_COUNT"):
        p.draw_indirect(args, count_buffer=counts)


def test_a_count_buffer_must_be_a_storage_buffer(extra_context):
    ctx = count_context(extra_context)
    args = ctx.create_buffer(DRAW_ARGS_BYTES, bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    wrong = ctx.create_buffer(16, bz.BufferUsage.UNIFORM, bz.MemoryUsage.STATIC)
    g = ctx.graph()
    p = g.add_pass(ctx.create_render_target(8, 8))
    with pytest.raises(bz.ResourceError, match="STORAGE"):
        p.draw_indirect(args, count_buffer=wrong)


def test_a_count_offset_must_be_aligned_and_inside_the_buffer(extra_context):
    ctx = count_context(extra_context)
    args = ctx.create_buffer(DRAW_ARGS_BYTES, bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    counts = ctx.create_buffer(8, bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    g = ctx.graph()
    p = g.add_pass(ctx.create_render_target(8, 8))
    with pytest.raises(bz.ResourceError, match="multiple of 4"):
        p.draw_indirect(args, count_buffer=counts, count_offset=2)
    # 4 bytes at offset 8 need 12, and the buffer is 8. The near-maximum offset
    # is the one that catches an unsigned overflow rather than an off-by-one.
    with pytest.raises(bz.ResourceError, match="count buffer is 8"):
        p.draw_indirect(args, count_buffer=counts, count_offset=8)
    with pytest.raises(bz.ResourceError, match="count buffer is 8"):
        p.draw_indirect(args, count_buffer=counts, count_offset=2**64 - 4)


# ── stride= (0.25) ───────────────────────────────────────────────────────


def test_indirect_stride_steps_over_per_draw_data(extra_context):
    """stride= is what lets draw arguments be INTERLEAVED with per-draw data
    instead of living in their own packed array. Two commands 32 bytes apart,
    with 16 bytes of the caller's own payload between them — the same two stripes
    the packed multi-draw above produces, so a stride the driver read wrongly
    would paint something else."""
    multi = extra_context(optional=[bz.Feature.MULTI_DRAW_INDIRECT])
    if not multi.supports(bz.Feature.MULTI_DRAW_INDIRECT):
        pytest.skip("GPU reports no multiDrawIndirect")

    # VkDrawIndirectCommand is 4 words; the 4 after each are the caller's.
    words = np.zeros(12, dtype=np.uint32)
    words[0:4] = [6, 1, 0, 0]
    words[4:8] = [0xDEAD, 0, 0, 0]
    words[8:12] = [6, 2, 0, 0]
    args = multi.create_buffer(words, bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)

    vert = multi.compile_shader(str(SHADER_DIR / "stripe.vert"), bz.ShaderStage.VERTEX)
    frag = multi.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    target = multi.create_render_target(TARGET, TARGET)
    draw = (multi.graphics_pipeline().vertex_shader(vert).fragment_shader(frag)
            .push_constant(4, bz.ShaderStage.VERTEX).build(target))

    g = multi.graph()
    with g.add_pass(target, clear_color=[0.0, 0.0, 0.0, 1.0]) as p:
        p.bind_pipeline(draw)
        p.push_constants(draw, 0, struct.pack("I", TOTAL_STRIPES))
        p.draw_indirect(args, count=2, stride=32)
    multi.submit(g)

    # The same picture the packed version above produces: both commands start at
    # instance 0, so the union is stripes 0 and 1. A stride the driver read
    # differently would paint some other number of them.
    assert painted(target) == 2 * STRIPE_PIXELS


def test_a_stride_smaller_than_the_struct_is_refused(ctx):
    """stride= exists to leave room BETWEEN the argument structs, so it can only
    be larger. Smaller would make consecutive commands overlap."""
    buffer = ctx.create_buffer(256, bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    g = ctx.graph()
    p = g.add_pass(ctx.create_render_target(8, 8))
    with pytest.raises(bz.ResourceError, match="stride"):
        p.draw_indirect(buffer, count=2, stride=8)
    with pytest.raises(bz.ResourceError, match="multiple of 4"):
        p.draw_indirect(buffer, count=2, stride=17)


def test_the_last_command_needs_only_its_own_struct(extra_context):
    """A buffer sized exactly for the data must not be refused: the padding a
    stride leaves is between commands, and the last one has nothing after it."""
    multi = extra_context(optional=[bz.Feature.MULTI_DRAW_INDIRECT])
    if not multi.supports(bz.Feature.MULTI_DRAW_INDIRECT):
        pytest.skip("GPU reports no multiDrawIndirect")
    # Two 16-byte commands at a stride of 32 need 32 + 16 = 48 bytes, not 64.
    buffer = multi.create_buffer(48, bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    g = multi.graph()
    p = g.add_pass(multi.create_render_target(8, 8))
    p.draw_indirect(buffer, count=2, stride=32)
