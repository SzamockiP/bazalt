"""Automatic barriers between resources, and the manual mode that disables them.

Auto barriers are computed at record time: deferred recording fixes the usage
sequence constructively, so a barrier computed once is correct for every
replay. The validation-as-assert fixture is the referee — a wrong or missing
auto barrier surfaces as a validation error and fails the test.

The one thing core validation CANNOT see is a missing barrier (that takes
synchronization validation), which is why the manual-mode negative test spins
up a second Context with validation="sync" of its own.
"""

import os
import struct

import numpy as np
import pytest

import bazalt as bz

from conftest import SHADER_DIR


@pytest.fixture
def fullscreen_vert(ctx):
    return ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)


@pytest.fixture
def double_pipeline(ctx):
    comp = ctx.compile_shader(str(SHADER_DIR / "double.comp"), bz.ShaderStage.COMPUTE)
    return ctx.compute_pipeline().shader(comp).storage_buffer(0).build()


@pytest.fixture
def add_one_pipeline(ctx):
    comp = ctx.compile_shader(str(SHADER_DIR / "add_one.comp"), bz.ShaderStage.COMPUTE)
    return ctx.compute_pipeline().shader(comp).storage_buffer(0).build()


def make_set(ctx, pipeline, sbuf):
    pool = ctx.create_descriptor_pool(max_sets=8, storage_buffers=8)
    dset = pool.allocate_set(pipeline, set=0)
    dset.set_buffer(0, sbuf)
    return pool, dset


# ── auto mode ─────────────────────────────────────────────────────────────


def test_dispatch_to_dispatch_gets_a_barrier(ctx, double_pipeline, add_one_pipeline):
    """(x * 2) + 1 requires the second dispatch to see the first one's writes."""
    data = np.arange(64, dtype=np.float32)
    sbuf = ctx.create_buffer(data, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    pool_a, dset_a = make_set(ctx, double_pipeline, sbuf)
    pool_b, dset_b = make_set(ctx, add_one_pipeline, sbuf)

    cmd = ctx.create_command_buffer()
    (cmd.begin()
        .bind_pipeline(double_pipeline)
        .bind_descriptor_set(dset_a, double_pipeline, set=0)
        .dispatch(1)
        .bind_pipeline(add_one_pipeline)
        .bind_descriptor_set(dset_b, add_one_pipeline, set=0)
        .dispatch(1))
    ctx.submit(cmd)

    assert np.allclose(sbuf.read(np.float32), data * 2 + 1)


def test_dispatch_to_draw_via_descriptor_read(ctx, fullscreen_vert, double_pipeline):
    """Compute writes an SSBO, the fragment shader reads it — RAW across bind points."""
    frag = ctx.compile_shader(str(SHADER_DIR / "ssbo.frag"), bz.ShaderStage.FRAGMENT)
    target = ctx.create_render_target(32, 32)
    gfx = (ctx.graphics_pipeline()
           .vertex_shader(fullscreen_vert)
           .fragment_shader(frag)
           .storage_buffer(0, bz.ShaderStage.FRAGMENT, set=0)
           .build(target))

    # double.comp turns (0, 0, 0.5, 0.5) into the (0, 0, 1, 1) ssbo.frag paints.
    sbuf = ctx.create_buffer(np.array([0.0, 0.0, 0.5, 0.5], dtype=np.float32),
                             bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    comp_pool, comp_set = make_set(ctx, double_pipeline, sbuf)
    gfx_pool, gfx_set = make_set(ctx, gfx, sbuf)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.bind_pipeline(double_pipeline)
    cmd.bind_descriptor_set(comp_set, double_pipeline, set=0)
    cmd.dispatch(1)
    with cmd.rendering(target):
        cmd.bind_pipeline(gfx).bind_descriptor_set(gfx_set, gfx, set=0).draw(3)
    ctx.submit(cmd)

    assert np.allclose(target.color[0].read()[16, 16, :3], [0, 0, 255], atol=2)


def test_dispatch_to_vertex_fetch_hoists_the_barrier(ctx, double_pipeline):
    """Compute writes vertices, the draw consumes them via bind_vertex_buffer.

    The bind happens inside the rendering scope, so the barrier must be
    hoisted before begin_rendering — vkCmdPipelineBarrier is illegal inside
    dynamic rendering, and the validation fixture would catch it there.
    """
    comp = ctx.compile_shader(str(SHADER_DIR / "write_vertices.comp"), bz.ShaderStage.COMPUTE)
    write_verts = ctx.compute_pipeline().shader(comp).storage_buffer(0).build()

    vert = ctx.compile_shader(str(SHADER_DIR / "pos2.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    target = ctx.create_render_target(32, 32)
    gfx = (ctx.graphics_pipeline()
           .vertex_shader(vert)
           .fragment_shader(frag)
           .vertex_format([bz.VertexFormat.FLOAT2])
           .build(target))

    # Garbage in: the triangle only covers the screen if the dispatch's writes
    # actually reached the vertex fetch.
    verts = ctx.create_buffer(np.zeros(6, dtype=np.float32),
                              bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    pool, dset = make_set(ctx, write_verts, verts)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.bind_pipeline(write_verts)
    cmd.bind_descriptor_set(dset, write_verts, set=0)
    cmd.dispatch(1)
    with cmd.rendering(target):
        cmd.bind_pipeline(gfx).bind_vertex_buffer(verts).draw(3)
    ctx.submit(cmd)

    assert np.allclose(target.color[0].read()[16, 16, :3], [255, 0, 0], atol=2)


def test_draw_then_dispatch_is_write_after_read(ctx, double_pipeline):
    """The dispatch must wait for the draw that reads the buffer it writes."""
    comp = ctx.compile_shader(str(SHADER_DIR / "write_vertices.comp"), bz.ShaderStage.COMPUTE)
    write_verts = ctx.compute_pipeline().shader(comp).storage_buffer(0).build()

    vert = ctx.compile_shader(str(SHADER_DIR / "pos2.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    target = ctx.create_render_target(32, 32)
    gfx = (ctx.graphics_pipeline()
           .vertex_shader(vert)
           .fragment_shader(frag)
           .vertex_format([bz.VertexFormat.FLOAT2])
           .build(target))

    tri = np.array([-1.0, -1.0, -1.0, 3.0, 3.0, -1.0], dtype=np.float32)
    verts = ctx.create_buffer(tri, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    pool, dset = make_set(ctx, write_verts, verts)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(target):
        cmd.bind_pipeline(gfx).bind_vertex_buffer(verts).draw(3)
    cmd.bind_pipeline(write_verts)
    cmd.bind_descriptor_set(dset, write_verts, set=0)
    cmd.dispatch(1)
    ctx.submit(cmd)

    assert np.allclose(target.color[0].read()[16, 16, :3], [255, 0, 0], atol=2)
    assert np.allclose(verts.read(np.float32), tri)  # rewrote the same values


# ── manual mode ───────────────────────────────────────────────────────────


def test_manual_mode_with_explicit_barriers_is_clean(ctx, double_pipeline, add_one_pipeline):
    """Same dispatch chain as the auto test, barriers spelled by hand."""
    data = np.arange(64, dtype=np.float32)
    sbuf = ctx.create_buffer(data, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    pool_a, dset_a = make_set(ctx, double_pipeline, sbuf)
    pool_b, dset_b = make_set(ctx, add_one_pipeline, sbuf)

    cmd = ctx.create_command_buffer(auto_barriers=False)
    (cmd.begin()
        .bind_pipeline(double_pipeline)
        .bind_descriptor_set(dset_a, double_pipeline, set=0)
        .dispatch(1)
        .barrier(sbuf, bz.Access.SHADER_WRITE, bz.Access.SHADER_READ)
        .barrier(sbuf, bz.Access.SHADER_WRITE, bz.Access.SHADER_WRITE)
        .bind_pipeline(add_one_pipeline)
        .bind_descriptor_set(dset_b, add_one_pipeline, set=0)
        .dispatch(1))
    ctx.submit(cmd)

    assert np.allclose(sbuf.read(np.float32), data * 2 + 1)


def test_barrier_inside_rendering_scope_is_refused(ctx, triangle_shaders, triangle_buffers):
    vert, frag = triangle_shaders
    vbuf, ibuf = triangle_buffers
    target = ctx.create_render_target(16, 16)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                .build(target))
    sbuf = ctx.create_buffer(np.zeros(4, dtype=np.float32),
                             bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)

    cmd = ctx.create_command_buffer(auto_barriers=False)
    cmd.begin()
    cmd.begin_rendering(target)
    with pytest.raises(bz.StateError, match="rendering scope"):
        cmd.barrier(sbuf, bz.Access.SHADER_WRITE, bz.Access.VERTEX_READ)
    cmd.end_rendering(target)


def test_context_wide_manual_mode_is_inherited(ctx):
    """create_command_buffer() without the kwarg takes the Context's default."""
    assert ctx.auto_barriers is True
    # The per-CB override is the only way to go manual on this (auto) Context;
    # the flag's plumbing from ContextConfig is exercised by run_sync_case.
    cmd = ctx.create_command_buffer(auto_barriers=False)
    assert cmd is not None


# ── sync validation: the proof that manual mode is really manual ─────────


def sync_setup(auto):
    """A sync-validation Context plus one compute pipeline over one storage buffer.

    In-process since 0.15. This used to be a script-in-a-string run through
    subprocess, purely because sync validation needs its own Context and only one
    could be alive per process — so this doubles as the sharpest proof that two
    Contexts with different validation settings now coexist: the session Context
    is right here, on "auto", while this one runs the sync layer.

    Returns the pieces rather than a result, because the recordings built on top
    of it differ: one recording with two dispatches, or two recordings with one
    each.
    """
    hazards = []
    log = bz.Logger(min_severity=bz.Severity.INFO)

    @log.on_message
    def _(msg):
        # "hazard detected" on recent layers, "Hazard WRITE_AFTER_WRITE" on
        # older ones.
        if msg.source == bz.Source.VALIDATION and "hazard" in msg.text.lower():
            hazards.append(msg.text)

    context = bz.Context(log, validation="sync", auto_barriers=auto)
    assert context.auto_barriers is auto

    comp = context.compile_shader(str(SHADER_DIR / "double.comp"), bz.ShaderStage.COMPUTE)
    pipeline = context.compute_pipeline().shader(comp).storage_buffer(0).build()
    sbuf = context.create_buffer(np.arange(64, dtype=np.float32),
                                 bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    pool = context.create_descriptor_pool(max_sets=8, storage_buffers=8)
    dset = pool.allocate_set(pipeline, set=0)
    dset.set_buffer(0, sbuf)
    return context, log, hazards, pipeline, sbuf, dset


def run_sync_case(mode):
    """One dispatch-pair recording, returning the hazards the layer reported."""
    context, log, hazards, pipeline, sbuf, dset = sync_setup(auto=mode == "auto")

    cmd = context.create_command_buffer()
    cmd.begin()
    cmd.bind_pipeline(pipeline)
    cmd.bind_descriptor_set(dset, pipeline, set=0)
    cmd.dispatch(1)
    if mode == "barrier":
        cmd.barrier(sbuf, bz.Access.SHADER_WRITE, bz.Access.SHADER_READ)
        cmd.barrier(sbuf, bz.Access.SHADER_WRITE, bz.Access.SHADER_WRITE)
    cmd.dispatch(1)
    context.submit(cmd)

    log.flush()
    return hazards


def run_cross_recording_case(auto):
    """A compute recording that WRITES a buffer, then a graphics recording that
    only READS it, as two separate CommandBuffers.

    The read-only half is the point, and it took a wrong test to find out why. The
    replay wrap-around barrier at the top of execute() (0.19) already covers a
    recording that writes anything tracked, so two writing recordings are ordered
    and prove nothing. `tracked_writes_` is per RECORDING, not per buffer, so the
    hole is exactly a recording that writes nothing the tracker sees — a draw,
    whose only writes are attachments. That is the shape of
    examples/28_gpu_culling, which is why that example carried a manual barrier.

    Both submits are asynchronous, so the queue really does have both in flight; a
    blocking submit would resolve the dependency by waiting and prove nothing
    either.
    """
    context, log, hazards, pipeline, sbuf, dset = sync_setup(auto)

    writer = context.create_command_buffer()
    writer.begin()
    writer.bind_pipeline(pipeline)
    writer.bind_descriptor_set(dset, pipeline, set=0)
    writer.dispatch(1)

    # The reader draws straight out of the storage buffer: BufferType.STORAGE
    # carries VERTEX_BUFFER_BIT, so no second resource is needed to express
    # "compute produced these vertices".
    vert = context.compile_shader(str(SHADER_DIR / "triangle.vert"), bz.ShaderStage.VERTEX)
    frag = context.compile_shader(str(SHADER_DIR / "triangle.frag"), bz.ShaderStage.FRAGMENT)
    target = context.create_render_target(16, 16)
    draw_pipeline = (context.graphics_pipeline()
                     .vertex_shader(vert)
                     .fragment_shader(frag)
                     .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                     .build(target))
    reader = context.create_command_buffer()
    reader.begin()
    with reader.rendering(target):
        reader.bind_pipeline(draw_pipeline)
        reader.bind_vertex_buffer(sbuf)
        reader.draw(3)

    context.submit(writer, wait=False)
    context.submit(reader, wait=False)
    context.wait()

    log.flush()
    return hazards


@pytest.mark.skipif(
    os.environ.get("BAZALT_SYNCVAL_UNSUPPORTED") == "1",
    reason="this environment's validation layer cannot report shader-access "
           "sync hazards (verified: 1.4.313, still the newest LunarG packages "
           "for Ubuntu noble, stays silent even with the settings file forcing "
           "validate_sync + syncval_shader_accesses_heuristic; 1.4.350 reports "
           "them — the messenger itself was proven alive)")
def test_missing_barrier_in_manual_mode_trips_sync_validation(ctx):
    """If this test fails, manual mode is not really manual (or sync validation
    is not really on) — either way the mode would be a lie.

    CI computes the skip from the installed layer version rather than declaring
    one, so this starts running there by itself when a new enough package
    appears (tech debt #4)."""
    hazards = run_sync_case("nobarrier")
    assert hazards


def test_explicit_barrier_in_manual_mode_satisfies_sync_validation(ctx):
    assert run_sync_case("barrier") == []


def test_auto_barriers_satisfy_sync_validation(ctx):
    """The auto tracker's barriers hold up under the same referee that catches
    the missing ones — not just under core validation, which is blind here."""
    assert run_sync_case("auto") == []


# ── the first-use floor: hazards from OUTSIDE one recording (0.24) ───────


@pytest.mark.skipif(
    os.environ.get("BAZALT_SYNCVAL_UNSUPPORTED") == "1",
    reason="this environment's validation layer cannot report shader-access "
           "sync hazards (see the skip on the manual-mode case above)")
def test_cross_recording_hazard_is_real_in_manual_mode(ctx):
    """The negative control, and the whole test below depends on it.

    Without it, a clean auto-mode run would be indistinguishable from a layer that
    does not look across command buffers at all — the shape of proof the 0.19
    lesson names, where the safe direction of being wrong silently reports
    success."""
    assert run_cross_recording_case(auto=False)


def test_cross_recording_hazard_is_barriered_automatically(ctx):
    """A read-only recording is ordered against a write it cannot see. The
    first-use floor is what covers it, and before 0.24 this reported a hazard."""
    assert run_cross_recording_case(auto=True) == []


def test_a_transfer_write_can_be_named_by_hand(extra_context):
    """cmd.fill_buffer writes; until 0.26 no Access could say so.

    The automatic tracker always knew — it puts a TRANSFER_WRITE floor under a
    buffer's first reader. What it cannot see is a buffer reached by ADDRESS,
    and that is exactly where the manual verb is the only tool: zero a counter
    with fill_buffer, read it from a dispatch through a pointer, and the hazard
    had no spelling at all.

    Sync validation is the referee here, not the values: it is the mode that
    reports a missing barrier.
    """
    context = extra_context(validation="sync", optional=[bz.Feature.BUFFER_ADDRESS])
    if not context.supports(bz.Feature.BUFFER_ADDRESS):
        pytest.skip("this GPU has no bufferDeviceAddress")

    source = """
    #version 450
    #extension GL_EXT_buffer_reference : require
    layout(local_size_x = 1) in;
    layout(buffer_reference, std430) buffer Counter { uint v[]; };
    layout(push_constant, std430) uniform PC { Counter counter; };
    void main() { counter.v[1] = counter.v[0] + 7u; }
    """
    pipeline = (context.compute_pipeline()
                .shader(context.compile_shader(source=source, stage=bz.ShaderStage.COMPUTE))
                .push_constant(8)
                .build())

    buf = context.create_buffer(
        np.full(4, 99, dtype=np.uint32), bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)

    cmd = context.create_command_buffer()
    cmd.begin()
    cmd.fill_buffer(buf, 5, size=4)
    cmd.barrier(buf, bz.Access.TRANSFER_WRITE, bz.Access.SHADER_READ)
    (cmd.bind_pipeline(pipeline)
        .push_constants(pipeline, 0, struct.pack("<Q", buf.address))
        .dispatch(1))
    context.submit(cmd)

    got = buf.read(np.uint32)
    assert (int(got[0]), int(got[1])) == (5, 12)


def test_transfer_read_is_spelled_too(extra_context):
    """The other half: a shader writes, a copy reads."""
    context = extra_context()
    src = context.create_buffer(
        np.arange(8, dtype=np.uint32), bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    dst = context.create_buffer(
        np.zeros(8, dtype=np.uint32), bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)

    cmd = context.create_command_buffer()
    cmd.begin()
    cmd.barrier(src, bz.Access.SHADER_WRITE, bz.Access.TRANSFER_READ)
    cmd.copy_buffer(src, dst)
    context.submit(cmd)

    assert np.array_equal(dst.read(np.uint32), np.arange(8, dtype=np.uint32))
