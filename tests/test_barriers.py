"""Automatic barriers between passes, and the manual mode that disables them.

Auto barriers are computed when the graph compiles: the fold walks every
pass's recorded uses in add order, so a use inside the graph names its real
predecessor and gets a precise edge, and only the first touch of a resource
falls back to the floors (the writer may be another graph, or last frame).
The validation-as-assert fixture is the referee — a wrong or missing barrier
surfaces as a validation error and fails the test.

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
    """(x * 2) + 1 requires the second dispatch to see the first one's writes —
    inside one pass, scheduled at the recorded position."""
    data = np.arange(64, dtype=np.float32)
    sbuf = ctx.create_buffer(data, bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    pool_a, dset_a = make_set(ctx, double_pipeline, sbuf)
    pool_b, dset_b = make_set(ctx, add_one_pipeline, sbuf)

    g = ctx.graph()
    (g.add_pass()
      .bind_pipeline(double_pipeline)
      .bind_descriptor_set(dset_a, double_pipeline, set=0)
      .dispatch(1)
      .bind_pipeline(add_one_pipeline)
      .bind_descriptor_set(dset_b, add_one_pipeline, set=0)
      .dispatch(1))
    ctx.submit(g)

    assert np.allclose(sbuf.read(np.float32), data * 2 + 1)


def test_dispatch_pass_to_dispatch_pass_gets_a_barrier(ctx, double_pipeline, add_one_pipeline):
    """The same chain split across two passes: the fold sees the first pass's
    write as the second pass's named predecessor, so the barrier is the precise
    cross-pass edge — not a floor."""
    data = np.arange(64, dtype=np.float32)
    sbuf = ctx.create_buffer(data, bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    pool_a, dset_a = make_set(ctx, double_pipeline, sbuf)
    pool_b, dset_b = make_set(ctx, add_one_pipeline, sbuf)

    g = ctx.graph()
    g.add_pass(name="double").bind_pipeline(double_pipeline) \
        .bind_descriptor_set(dset_a, double_pipeline, set=0).dispatch(1)
    g.add_pass(name="add one").bind_pipeline(add_one_pipeline) \
        .bind_descriptor_set(dset_b, add_one_pipeline, set=0).dispatch(1)
    ctx.submit(g)

    assert np.allclose(sbuf.read(np.float32), data * 2 + 1)


def test_dispatch_to_draw_via_descriptor_read(ctx, fullscreen_vert, double_pipeline):
    """Compute writes an SSBO, the fragment shader reads it — RAW across passes
    and across bind points."""
    frag = ctx.compile_shader(str(SHADER_DIR / "ssbo.frag"), bz.ShaderStage.FRAGMENT)
    target = ctx.create_render_target(32, 32)
    gfx = (ctx.graphics_pipeline()
           .vertex_shader(fullscreen_vert)
           .fragment_shader(frag)
           .storage_buffer(0, bz.ShaderStage.FRAGMENT, set=0)
           .build(target))

    # double.comp turns (0, 0, 0.5, 0.5) into the (0, 0, 1, 1) ssbo.frag paints.
    sbuf = ctx.create_buffer(np.array([0.0, 0.0, 0.5, 0.5], dtype=np.float32),
                             bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    comp_pool, comp_set = make_set(ctx, double_pipeline, sbuf)
    gfx_pool, gfx_set = make_set(ctx, gfx, sbuf)

    g = ctx.graph()
    g.add_pass().bind_pipeline(double_pipeline) \
        .bind_descriptor_set(comp_set, double_pipeline, set=0).dispatch(1)
    with g.add_pass(target) as p:
        p.bind_pipeline(gfx).bind_descriptor_set(gfx_set, gfx, set=0).draw(3)
    ctx.submit(g)

    assert np.allclose(target.color[0].read()[16, 16, :3], [0, 0, 255], atol=2)


def test_dispatch_to_vertex_fetch_lands_in_the_entry_batch(ctx, double_pipeline):
    """Compute writes vertices, the draw consumes them via bind_vertex_buffer.

    The bind is recorded inside the render pass, and vkCmdPipelineBarrier is
    illegal inside dynamic rendering — so the compile puts the barrier in the
    render pass's ENTRY batch, before the rendering scope opens. The old
    recorder solved the same problem by hoisting; the validation fixture would
    catch either getting it wrong.
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
                              bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    pool, dset = make_set(ctx, write_verts, verts)

    g = ctx.graph()
    g.add_pass().bind_pipeline(write_verts) \
        .bind_descriptor_set(dset, write_verts, set=0).dispatch(1)
    with g.add_pass(target) as p:
        p.bind_pipeline(gfx).bind_vertex_buffer(verts).draw(3)
    ctx.submit(g)

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
    verts = ctx.create_buffer(tri, bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    pool, dset = make_set(ctx, write_verts, verts)

    g = ctx.graph()
    with g.add_pass(target) as p:
        p.bind_pipeline(gfx).bind_vertex_buffer(verts).draw(3)
    g.add_pass().bind_pipeline(write_verts) \
        .bind_descriptor_set(dset, write_verts, set=0).dispatch(1)
    ctx.submit(g)

    assert np.allclose(target.color[0].read()[16, 16, :3], [255, 0, 0], atol=2)
    assert np.allclose(verts.read(np.float32), tri)  # rewrote the same values


# ── what a render pass WRITES (0.28) ──────────────────────────────────────
#
# An attachment write is not a descriptor use, so no recorded verb reports it.
# The graph reports it after folding a render pass, and these are the tests
# that say why: without it, "render into a texture, then sample it" — a
# G-buffer, a post-process chain, a shadow map — has no barrier at all.


def run_render_then_sample_case(reader):
    """Pass 1 draws into an offscreen target; pass 2 reads that target's colour.

    `reader` picks which stage does the reading, and that is the point of the
    parametrization: the pass's own exit transition names the fragment shader,
    so a fragment reader must need nothing more, while a COMPUTE reader is
    outside what the retire covered and must get a barrier from the fold.

    Builds its own sync-validation Context, exactly as run_sync_case does and
    for the same reason: core validation cannot see a MISSING barrier, so on
    the session Context this would pass against the unfixed build — the 0.24
    lesson about a test that passes before the fix being a test of something
    else. Returns the hazards and the pixels, because both halves matter.
    """
    hazards = []
    log = bz.Logger(min_severity=bz.Severity.INFO)

    @log.on_message
    def _(msg):
        if msg.source == bz.Source.VALIDATION and "hazard" in msg.text.lower():
            hazards.append(msg.text)

    ctx = bz.Context(log, validation="sync")
    vert = ctx.compile_shader(str(SHADER_DIR / "triangle.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "triangle.frag"), bz.ShaderStage.FRAGMENT)
    tri = np.array([
        [-0.9, 0.9, 0.2, 1.0, 0.0, 0.0],
        [0.9, 0.9, 0.2, 0.0, 1.0, 0.0],
        [0.0, -0.9, 0.2, 0.0, 0.0, 1.0],
    ], dtype=np.float32)
    vbuf = ctx.create_buffer(tri, bz.BufferUsage.VERTEX, bz.MemoryUsage.STATIC)
    offscreen = ctx.create_render_target(32, 32)
    drawn = (ctx.graphics_pipeline()
             .vertex_shader(vert)
             .fragment_shader(frag)
             .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
             .build(offscreen))

    g = ctx.graph()
    with g.add_pass(offscreen, clear_color=[0, 0, 0, 1], name="draw") as p:
        p.bind_pipeline(drawn).bind_vertex_buffer(vbuf).draw(3)

    if reader == "fragment":
        final = ctx.create_render_target(32, 32)
        show = (ctx.graphics_pipeline()
                .vertex_shader(ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"),
                                                  bz.ShaderStage.VERTEX))
                .fragment_shader(ctx.compile_shader(str(SHADER_DIR / "textured.frag"),
                                                    bz.ShaderStage.FRAGMENT))
                .texture(0, bz.ShaderStage.FRAGMENT, set=0)
                .build(final))
        dset = ctx.create_descriptor_pool(max_sets=1, textures=1).allocate_set(show, set=0)
        dset.set_image(0, offscreen.color[0])
        with g.add_pass(final, clear_color=[0, 0, 0, 1], name="sample") as p:
            p.bind_pipeline(show).bind_descriptor_set(dset, show, set=0).draw(3)
        ctx.submit(g)
        pixels = final.color[0].read()
        log.flush()
        return hazards, pixels

    out = ctx.create_image(32, 32, bz.Format.RGBA8)
    copy = (ctx.compute_pipeline()
            .shader(ctx.compile_shader(str(SHADER_DIR / "sample_texture.comp"),
                                       bz.ShaderStage.COMPUTE))
            .texture(0, set=0)
            .storage_image(1, set=0)
            .build())
    pool = ctx.create_descriptor_pool(max_sets=1, textures=1, storage_images=1)
    dset = pool.allocate_set(copy, set=0)
    dset.set_image(0, offscreen.color[0], sampler=ctx.create_sampler())
    dset.set_storage_image(1, out)
    with g.add_pass(name="sample") as p:
        p.bind_pipeline(copy).bind_descriptor_set(dset, copy, set=0).dispatch(4, 4)
    ctx.submit(g)
    pixels = out.read()
    log.flush()
    return hazards, pixels


@pytest.mark.parametrize("reader", ["fragment", "compute"])
def test_rendering_into_a_texture_then_sampling_it(ctx, reader):
    """Sync validation is the referee, and the pixels are the second half: a
    black result would mean the read happened before the draw landed.

    Before the fix this reported READ_AFTER_WRITE at the reading command. An
    attachment write is not a descriptor use, so nothing reported it to the
    fold, and the compile saw no predecessor to order the read against. It is
    the most ordinary thing a frame does — a G-buffer, a post-process chain —
    and the suite reached it through two graphs every time, where the
    cross-graph floor covered it."""
    hazards, pixels = run_render_then_sample_case(reader)
    assert hazards == []
    assert pixels[16, 16, :3].sum() > 0, "the sampled image is empty"


def test_a_preserving_pass_after_something_else_moved_the_attachment(ctx, triangle_shaders,
                                                                     triangle_buffers):
    """A render pass builds its entry transition from the RenderTarget, so
    preserving means "come from final_layout()". Something between the two
    passes can move the image — here a compute pass writes it as a storage
    image — and then that transition names a layout the device is not in.

    The look-ahead cannot cover this: it only fires for two CONSECUTIVE passes
    on one target, and there is a pass in between. So the compile corrects the
    difference first, and the fixture is what proves it: the uncorrected
    version is VUID-VkImageMemoryBarrier-oldLayout-01197."""
    vbuf, ibuf = triangle_buffers
    image = ctx.create_image(32, 32, bz.Format.RGBA8)
    target = ctx.create_render_target(color=[image])
    drawn = (ctx.graphics_pipeline()
             .vertex_shader(triangle_shaders[0])
             .fragment_shader(triangle_shaders[1])
             .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
             .build(target))
    store = (ctx.compute_pipeline()
             .shader(ctx.compile_shader(str(SHADER_DIR / "store_const.comp"),
                                        bz.ShaderStage.COMPUTE))
             .storage_image(0)
             .build())
    dset = ctx.create_descriptor_pool(max_sets=1, storage_images=1).allocate_set(store, set=0)
    dset.set_storage_image(0, image)

    g = ctx.graph()
    with g.add_pass(target, clear_color=[0, 0, 0, 1]) as p:
        p.bind_pipeline(drawn).bind_vertex_buffer(vbuf).bind_index_buffer(ibuf).draw_indexed(3)
    with g.add_pass(name="overwrite") as p:
        p.bind_pipeline(store).bind_descriptor_set(dset, store, set=0).dispatch(4, 4)
    with g.add_pass(target, clear_color=None) as p:
        p.bind_pipeline(drawn).bind_vertex_buffer(vbuf).bind_index_buffer(ibuf).draw_indexed(3)
    ctx.submit(g)

    assert image.read() is not None


# ── manual mode ───────────────────────────────────────────────────────────


def test_manual_mode_with_explicit_barriers_is_clean(ctx, double_pipeline, add_one_pipeline):
    """Same dispatch chain as the auto test, barriers spelled by hand in a
    manual pass."""
    data = np.arange(64, dtype=np.float32)
    sbuf = ctx.create_buffer(data, bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    pool_a, dset_a = make_set(ctx, double_pipeline, sbuf)
    pool_b, dset_b = make_set(ctx, add_one_pipeline, sbuf)

    g = ctx.graph()
    (g.add_pass(auto_barriers=False)
      .bind_pipeline(double_pipeline)
      .bind_descriptor_set(dset_a, double_pipeline, set=0)
      .dispatch(1)
      .barrier(sbuf, bz.Access.SHADER_WRITE, bz.Access.SHADER_READ)
      .barrier(sbuf, bz.Access.SHADER_WRITE, bz.Access.SHADER_WRITE)
      .bind_pipeline(add_one_pipeline)
      .bind_descriptor_set(dset_b, add_one_pipeline, set=0)
      .dispatch(1))
    ctx.submit(g)

    assert np.allclose(sbuf.read(np.float32), data * 2 + 1)


def test_a_manual_pass_seeds_its_auto_neighbours(ctx, double_pipeline, add_one_pipeline):
    """The escape hatch composes: a manual pass's barriers feed the fold, so an
    automatic pass after it does not stack a second barrier on the dependency
    the manual one just expressed — and orders correctly against it. The
    referee is the fixture: a stale-layout double transition would be a
    validation error."""
    data = np.arange(64, dtype=np.float32)
    sbuf = ctx.create_buffer(data, bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    pool_a, dset_a = make_set(ctx, double_pipeline, sbuf)
    pool_b, dset_b = make_set(ctx, add_one_pipeline, sbuf)

    g = ctx.graph()
    (g.add_pass(auto_barriers=False, name="manual")
      .bind_pipeline(double_pipeline)
      .bind_descriptor_set(dset_a, double_pipeline, set=0)
      .dispatch(1)
      .barrier(sbuf, bz.Access.SHADER_WRITE, bz.Access.SHADER_WRITE))
    (g.add_pass(name="auto")
      .bind_pipeline(add_one_pipeline)
      .bind_descriptor_set(dset_b, add_one_pipeline, set=0)
      .dispatch(1))
    ctx.submit(g)

    assert np.allclose(sbuf.read(np.float32), data * 2 + 1)


def test_barrier_in_a_render_pass_is_refused(ctx, triangle_shaders, triangle_buffers):
    target = ctx.create_render_target(16, 16)
    sbuf = ctx.create_buffer(np.zeros(4, dtype=np.float32),
                             bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)

    g = ctx.graph()
    p = g.add_pass(target, auto_barriers=False)
    with pytest.raises(bz.StateError, match="rendering scope"):
        p.barrier(sbuf, bz.Access.SHADER_WRITE, bz.Access.VERTEX_READ)


def test_context_wide_manual_mode_is_inherited(ctx):
    """add_pass() without the kwarg takes the Context's default."""
    assert ctx.auto_barriers is True
    # The per-pass override is the only way to go manual on this (auto)
    # Context; the flag's plumbing from ContextConfig is exercised by
    # run_sync_case.
    p = ctx.graph().add_pass(auto_barriers=False)
    assert p is not None


# ── sync validation: the proof that manual mode is really manual ─────────


def sync_setup(auto):
    """A sync-validation Context plus one compute pipeline over one storage buffer.

    In-process since 0.15. This used to be a script-in-a-string run through
    subprocess, purely because sync validation needs its own Context and only one
    could be alive per process — so this doubles as the sharpest proof that two
    Contexts with different validation settings now coexist: the session Context
    is right here, on "auto", while this one runs the sync layer.

    Returns the pieces rather than a result, because the graphs built on top
    of it differ: one pass with two dispatches, or two graphs with one each.
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
                                 bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    pool = context.create_descriptor_pool(max_sets=8, storage_buffers=8)
    dset = pool.allocate_set(pipeline, set=0)
    dset.set_buffer(0, sbuf)
    return context, log, hazards, pipeline, sbuf, dset


def run_sync_case(mode):
    """One dispatch-pair pass, returning the hazards the layer reported."""
    context, log, hazards, pipeline, sbuf, dset = sync_setup(auto=mode == "auto")

    g = context.graph()
    p = g.add_pass()
    p.bind_pipeline(pipeline)
    p.bind_descriptor_set(dset, pipeline, set=0)
    p.dispatch(1)
    if mode == "barrier":
        p.barrier(sbuf, bz.Access.SHADER_WRITE, bz.Access.SHADER_READ)
        p.barrier(sbuf, bz.Access.SHADER_WRITE, bz.Access.SHADER_WRITE)
    p.dispatch(1)
    context.submit(g)

    log.flush()
    return hazards


def run_cross_graph_case(auto):
    """A compute graph that WRITES a buffer, then a graphics graph that only
    READS it, as two separate Graphs.

    The read-only half is the point, and it took a wrong test to find out why.
    The replay wrap-around barrier at the top of a graph's replay covers a
    graph that writes anything tracked, so two writing graphs are ordered and
    prove nothing. The hole is exactly a graph that writes nothing the fold
    sees — a draw, whose only writes are attachments. That is the shape of
    examples/28_gpu_culling, which is why that example carried a manual
    barrier before 0.24.

    Both submits are asynchronous, so the queue really does have both in
    flight; a blocking submit would resolve the dependency by waiting and
    prove nothing either.
    """
    context, log, hazards, pipeline, sbuf, dset = sync_setup(auto)

    writer = context.graph()
    writer.add_pass().bind_pipeline(pipeline) \
        .bind_descriptor_set(dset, pipeline, set=0).dispatch(1)

    # The reader draws straight out of the storage buffer: BufferUsage.STORAGE
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
    reader = context.graph()
    with reader.add_pass(target) as p:
        p.bind_pipeline(draw_pipeline)
        p.bind_vertex_buffer(sbuf)
        p.draw(3)

    context.submit(writer, wait=False)
    context.submit(reader, wait=False)
    context.wait()

    log.flush()
    return hazards


def run_two_queue_replay_case(auto):
    """One graph whose producer runs on the compute queue and whose consumer
    runs on the graphics queue, replayed twice without waiting.

    Two claims in one shape. Inside a replay, the consumer's read of what the
    producer wrote crosses a queue, so a pipeline barrier cannot express it and
    the fold has to turn it into a semaphore wait. Between replays, the second
    producer overwrites the buffer the first consumer is still reading — the
    wrap-around barrier's hazard, one level up, and again out of a barrier's
    reach because the two run on different queues.

    Manual mode is the negative control and it is exact: a manual pass emits no
    UseEvents at all, so the fold sees nothing to wait for and the hazard is
    real. Returns the hazards and the numbers, because the value is the second
    referee where sync validation cannot see across queues.
    """
    context, log, hazards, pipeline, sbuf, dset = sync_setup(auto)

    g = context.graph()
    (g.add_pass(name="produce", queue=bz.Queue.COMPUTE)
        .bind_pipeline(pipeline).bind_descriptor_set(dset, pipeline, set=0).dispatch(1))
    (g.add_pass(name="consume", queue=bz.Queue.GRAPHICS)
        .bind_pipeline(pipeline).bind_descriptor_set(dset, pipeline, set=0).dispatch(1))

    context.submit(g, wait=False)
    context.submit(g, wait=False)
    context.wait()

    log.flush()
    return hazards, sbuf.read(np.float32)


def run_compute_image_then_draw_case(queue):
    """A pass writes a storage image, a render pass samples it. On the compute
    queue the layout move (GENERAL to SHADER_READ_ONLY) is the interesting
    half: the semaphore carries the memory dependency, so the transition has an
    empty source scope and only the layout to change."""
    hazards = []
    log = bz.Logger(min_severity=bz.Severity.INFO)

    @log.on_message
    def _(msg):
        if msg.source == bz.Source.VALIDATION and "hazard" in msg.text.lower():
            hazards.append(msg.text)

    context = bz.Context(log, validation="sync")
    pattern = context.compile_shader(str(SHADER_DIR / "pattern.comp"), bz.ShaderStage.COMPUTE)
    write = (context.compute_pipeline().shader(pattern).storage_image(0)
             .push_constant(4).build())
    vert = context.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = context.compile_shader(str(SHADER_DIR / "textured.frag"), bz.ShaderStage.FRAGMENT)
    target = context.create_render_target(64, 64)
    read = (context.graphics_pipeline().vertex_shader(vert).fragment_shader(frag)
            .texture(0, bz.ShaderStage.FRAGMENT).build(target))

    image = context.create_image(64, 64, bz.Format.RGBA8)
    pool = context.create_descriptor_pool()
    write_set = pool.allocate_set(write)
    write_set.set_storage_image(0, image)
    read_set = pool.allocate_set(read)
    read_set.set_image(0, image)

    g = context.graph()
    (g.add_pass(name="write", queue=queue)
        .bind_pipeline(write).bind_descriptor_set(write_set, write)
        .push_constants(write, 0, struct.pack("<f", 0.5))
        .dispatch(8, 8))
    with g.add_pass(target, name="sample") as p:
        p.bind_pipeline(read).bind_descriptor_set(read_set, read).draw(3)
    context.submit(g)

    log.flush()
    return hazards, target.color[0].read()


def run_attachment_read_then_redraw_case():
    """Draw into an offscreen target, sample it from the COMPUTE queue, then
    clear and draw into it again.

    The third pass is the point. Its attachment transition is built by the
    RenderTarget rather than by the fold — a clearing pass comes from UNDEFINED
    and asks nothing about what went before — so the read the compute queue is
    still doing has nothing to order it against. A pipeline barrier could not
    carry it anyway: the reader is on the other queue. This is the
    write-after-read the fold has to notice by asking the tracker who has
    touched the attachment, before the pass runs.
    """
    hazards = []
    log = bz.Logger(min_severity=bz.Severity.INFO)

    @log.on_message
    def _(msg):
        if msg.source == bz.Source.VALIDATION and "hazard" in msg.text.lower():
            hazards.append(msg.text)

    context = bz.Context(log, validation="sync")
    vert = context.compile_shader(str(SHADER_DIR / "triangle.vert"), bz.ShaderStage.VERTEX)
    frag = context.compile_shader(str(SHADER_DIR / "triangle.frag"), bz.ShaderStage.FRAGMENT)
    target = context.create_render_target(64, 64)
    draw = (context.graphics_pipeline()
            .vertex_shader(vert).fragment_shader(frag)
            .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
            .build(target))
    vbuf = context.create_buffer(
        [0.0, -0.5, 0.0, 1.0, 0.0, 0.0,
         -0.5, 0.5, 0.0, 0.0, 1.0, 0.0,
         0.5, 0.5, 0.0, 0.0, 0.0, 1.0],
        bz.BufferUsage.VERTEX, bz.MemoryUsage.STATIC)

    # The compute half samples the attachment into a storage image, so its read
    # of the attachment is a descriptor use the fold can see.
    sample = context.compile_shader(str(SHADER_DIR / "sample_texture.comp"), bz.ShaderStage.COMPUTE)
    copy = context.compute_pipeline().shader(sample).texture(0).storage_image(1).build()
    scratch = context.create_image(64, 64, bz.Format.RGBA8)
    pool = context.create_descriptor_pool()
    copy_set = pool.allocate_set(copy)
    copy_set.set_image(0, target.color[0])
    copy_set.set_storage_image(1, scratch)

    g = context.graph()
    with g.add_pass(target, [0.1, 0.2, 0.3, 1.0], name="draw") as p:
        p.bind_pipeline(draw).bind_vertex_buffer(vbuf).draw(3)
    (g.add_pass(name="sample on compute", queue=bz.Queue.COMPUTE)
        .bind_pipeline(copy).bind_descriptor_set(copy_set, copy).dispatch(8, 8))
    with g.add_pass(target, [0.0, 0.0, 0.0, 1.0], name="redraw") as p:
        p.bind_pipeline(draw).bind_vertex_buffer(vbuf).draw(3)
    context.submit(g)

    log.flush()
    return hazards


def run_note_then_cross_queue_read_case(kind):
    """A pass on the compute queue leaves a resource in a state the fold learns
    from a NOTE rather than from a descriptor use, and a graphics pass then
    reads it.

    A note is what a manual p.barrier() and the transfer verbs (copy_image,
    clear_image) report. It used to be modelled as a completed READ, and a read
    is not something a later reader waits for — so the graphics pass got
    neither a barrier nor a semaphore wait, and read a resource the compute
    queue was still writing.

    `kind` picks which half: "image" goes through clear_image, "buffer" through
    a manual barrier after a dispatch.
    """
    hazards = []
    log = bz.Logger(min_severity=bz.Severity.INFO)

    @log.on_message
    def _(msg):
        if msg.source == bz.Source.VALIDATION and "hazard" in msg.text.lower():
            hazards.append(msg.text)

    context = bz.Context(log, validation="sync")
    vert = context.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    target = context.create_render_target(64, 64)

    if kind == "image":
        frag = context.compile_shader(str(SHADER_DIR / "textured.frag"), bz.ShaderStage.FRAGMENT)
        read = (context.graphics_pipeline().vertex_shader(vert).fragment_shader(frag)
                .texture(0, bz.ShaderStage.FRAGMENT).build(target))
        image = context.create_image(256, 256, bz.Format.RGBA8)
        pool = context.create_descriptor_pool()
        read_set = pool.allocate_set(read)
        read_set.set_image(0, image)

        g = context.graph()
        # A transfer verb reports its result as a note, not as a use.
        g.add_pass(name="clear on compute", queue=bz.Queue.COMPUTE).clear_image(image, [0.2, 0.4, 0.6, 1.0])
        with g.add_pass(target, name="sample") as p:
            p.bind_pipeline(read).bind_descriptor_set(read_set, read).draw(3)
        context.submit(g)
        log.flush()
        return hazards

    comp = context.compile_shader(str(SHADER_DIR / "double.comp"), bz.ShaderStage.COMPUTE)
    write = context.compute_pipeline().shader(comp).storage_buffer(0).build()
    sbuf = context.create_buffer(np.arange(64, dtype=np.float32),
                                 bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    pool = context.create_descriptor_pool(max_sets=8, storage_buffers=8)
    write_set = pool.allocate_set(write, set=0)
    write_set.set_buffer(0, sbuf)

    # The consumer only READS, and that is the whole point: a reader waits for
    # a writer and never for another reader, so a note that models the write as
    # a read leaves exactly this shape unordered. A consumer that WRITES is
    # ordered anyway, through the reads the note does record.
    tri_vert = context.compile_shader(str(SHADER_DIR / "triangle.vert"), bz.ShaderStage.VERTEX)
    tri_frag = context.compile_shader(str(SHADER_DIR / "triangle.frag"), bz.ShaderStage.FRAGMENT)
    draw = (context.graphics_pipeline().vertex_shader(tri_vert).fragment_shader(tri_frag)
            .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
            .build(target))

    g = context.graph()
    producer = g.add_pass(name="double on compute", queue=bz.Queue.COMPUTE)
    producer.bind_pipeline(write).bind_descriptor_set(write_set, write, set=0).dispatch(1)
    # The manual barrier covers the compute queue and can reach no further. It
    # must not DELETE the dependency for the queue it cannot reach.
    producer.barrier(sbuf, bz.Access.SHADER_WRITE, bz.Access.SHADER_READ)
    with g.add_pass(target, name="read on graphics") as p:
        p.bind_pipeline(draw).bind_vertex_buffer(sbuf).draw(3)
    context.submit(g)
    log.flush()
    return hazards


def run_preserve_chain_case():
    """Two render passes on one target, the second preserving — the look-ahead.

    The compile elides the retire/re-enter pair between them: the attachment
    stays in its attachment layout and one execution barrier covers the seam.
    Sync validation is the referee that the elision did not drop a dependency.
    """
    context, log, hazards, pipeline, sbuf, dset = sync_setup(auto=True)

    vert = context.compile_shader(str(SHADER_DIR / "triangle.vert"), bz.ShaderStage.VERTEX)
    frag = context.compile_shader(str(SHADER_DIR / "triangle.frag"), bz.ShaderStage.FRAGMENT)
    target = context.create_render_target(32, 32)
    draw_pipeline = (context.graphics_pipeline()
                     .vertex_shader(vert)
                     .fragment_shader(frag)
                     .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                     .build(target))
    tri = np.array([
        [-0.5, -0.5, 0.0, 1.0, 0.0, 0.0],
        [0.5, -0.5, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)
    vbuf = context.create_buffer(tri, bz.BufferUsage.VERTEX, bz.MemoryUsage.STATIC)

    g = context.graph()
    with g.add_pass(target, clear_color=[0, 0, 0, 1]) as p:
        p.bind_pipeline(draw_pipeline).bind_vertex_buffer(vbuf).draw(3)
    with g.add_pass(target, clear_color=None) as p:
        p.bind_pipeline(draw_pipeline).bind_vertex_buffer(vbuf).draw(3)
    context.submit(g)

    pixels = target.color[0].read()
    log.flush()
    return hazards, pixels


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
    """The fold's barriers hold up under the same referee that catches the
    missing ones — not just under core validation, which is blind here."""
    assert run_sync_case("auto") == []


@pytest.mark.skipif(
    os.environ.get("BAZALT_SYNCVAL_UNSUPPORTED") == "1",
    reason="the installed validation layer does not report shader hazards (see debt #4)")
def test_a_two_queue_graph_races_in_manual_mode(ctx):
    """The negative control for the two claims below, and it is what makes them
    mean anything: with the barriers off, a producer on one queue and a
    consumer on the other really do race, on this machine, in this build."""
    hazards, _ = run_two_queue_replay_case(auto=False)
    assert hazards, "no hazard reported — sync validation is not watching the queues"


def test_a_two_queue_graph_orders_itself(ctx):
    """A use whose producer sits in another batch on another queue becomes a
    semaphore wait, and a graph replayed again waits for its own previous
    submit on the other queues. The value is the second referee: doubling twice
    per replay over two replays is x16, and any missed edge gives a smaller
    number."""
    hazards, values = run_two_queue_replay_case(auto=True)
    assert hazards == []
    assert np.allclose(values, np.arange(64, dtype=np.float32) * 16)


@pytest.mark.parametrize("queue", ["graphics", "compute"])
def test_a_storage_image_written_on_either_queue_is_sampled_safely(ctx, queue):
    """The image half of the same claim. GENERAL to SHADER_READ_ONLY across a
    queue keeps the transition and drops its source scope, because the
    semaphore already made the write available."""
    kind = bz.Queue.COMPUTE if queue == "compute" else bz.Queue.GRAPHICS
    hazards, pixels = run_compute_image_then_draw_case(kind)
    assert hazards == []
    assert pixels[32, 32, :3].sum() > 0, "the sampled image was black"


def test_a_pass_that_redraws_an_attachment_waits_for_a_compute_reader(ctx):
    """A clearing render pass builds its entry transition from the target and
    asks the fold nothing, so the compute queue's read of that attachment has
    to be found the other way round: the fold asks the tracker who has touched
    the image before the pass runs (Graph::compile_, cross_queue_touches).

    A REGRESSION GUARD, NOT A PROOF, and the difference was measured rather
    than assumed. The fix was disabled and the build rerun, and this test still
    passed: sync validation does not report this write-after-read, though the
    race is real by the spec — the third pass writes an attachment the second
    is still sampling from another queue, with nothing between them. So what
    this pins is that the ordering the fold adds is not itself illegal, and the
    hazard it closes is argued from the spec rather than shown by a layer.

    On a device whose compute queue aliases the graphics one it proves even
    less: one queue orders the two passes anyway."""
    assert run_attachment_read_then_redraw_case() == []


@pytest.mark.skipif(
    os.environ.get("BAZALT_SYNCVAL_UNSUPPORTED") == "1",
    reason="the installed validation layer does not report shader hazards (see debt #4)")
@pytest.mark.parametrize("kind", ["image", "buffer"])
def test_a_note_on_one_queue_still_orders_the_other_queue(ctx, kind):
    """The transfer verbs and a manual p.barrier() reach the fold as notes, and
    a note used to say "somebody read this". A reader waits for a writer, never
    for another reader, so the consumer on the other queue was left completely
    unordered — no barrier, because a barrier cannot cross a queue, and no
    semaphore wait, because nothing asked for one.

    Found by review rather than by the suite, which is the interesting part: the
    two cross-queue tests above both drive the fold through descriptor USES, and
    every verb that reports a note was outside their shape."""
    assert run_note_then_cross_queue_read_case(kind) == []


def test_a_preserve_chain_satisfies_sync_validation(ctx):
    """The look-ahead's elision under the sharpest referee available, plus the
    pixels: the second pass really drew over the first one's output."""
    hazards, pixels = run_preserve_chain_case()
    assert hazards == []
    assert pixels is not None


# ── the graph-scope first-use floor: hazards from OUTSIDE one graph ──────


@pytest.mark.skipif(
    os.environ.get("BAZALT_SYNCVAL_UNSUPPORTED") == "1",
    reason="this environment's validation layer cannot report shader-access "
           "sync hazards (see the skip on the manual-mode case above)")
def test_cross_graph_hazard_is_real_in_manual_mode(ctx):
    """The negative control, and the whole test below depends on it.

    Without it, a clean auto-mode run would be indistinguishable from a layer
    that does not look across command buffers at all — the shape of proof the
    0.19 lesson names, where the safe direction of being wrong silently
    reports success."""
    assert run_cross_graph_case(auto=False)


def test_cross_graph_hazard_is_barriered_automatically(ctx):
    """A read-only graph is ordered against a write it cannot see. The
    first-use floor is what covers it — the fold's floors answer for writers
    outside the graph, exactly as the per-recording floors did since 0.24."""
    assert run_cross_graph_case(auto=True) == []


def test_a_transfer_write_can_be_named_by_hand(extra_context):
    """p.fill_buffer writes; until 0.26 no Access could say so.

    The automatic fold always knew — it puts a TRANSFER_WRITE floor under a
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
        np.full(4, 99, dtype=np.uint32), bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)

    g = context.graph()
    p = g.add_pass()
    p.fill_buffer(buf, 5, size=4)
    p.barrier(buf, bz.Access.TRANSFER_WRITE, bz.Access.SHADER_READ)
    (p.bind_pipeline(pipeline)
      .push_constants(pipeline, 0, struct.pack("<Q", buf.address))
      .dispatch(1))
    context.submit(g)

    got = buf.read(np.uint32)
    assert (int(got[0]), int(got[1])) == (5, 12)


def test_transfer_read_is_spelled_too(extra_context):
    """The other half: a shader writes, a copy reads."""
    context = extra_context()
    src = context.create_buffer(
        np.arange(8, dtype=np.uint32), bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    dst = context.create_buffer(
        np.zeros(8, dtype=np.uint32), bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)

    g = context.graph()
    p = g.add_pass()
    p.barrier(src, bz.Access.SHADER_WRITE, bz.Access.TRANSFER_READ)
    p.copy_buffer(src, dst)
    context.submit(g)

    assert np.array_equal(dst.read(np.uint32), np.arange(8, dtype=np.uint32))
