"""The graph itself: sealing, reset, recompilation, enabled, remove.

The pass-graph API of 0.28. What the other files test is what the passes
RECORD; this one tests the graph's own contract — when a pass stops accepting
commands, what invalidates the compiled barriers, and what the two structural
verbs (enabled, remove) do to a frame.
"""

import numpy as np
import pytest

import bazalt as bz

from conftest import SHADER_DIR


def add_one(ctx):
    """A dispatch that adds 1.0 to every element, plus the buffer it works on."""
    comp = ctx.compile_shader(str(SHADER_DIR / "add_one.comp"), bz.ShaderStage.COMPUTE)
    pipeline = ctx.compute_pipeline().shader(comp).storage_buffer(0).build()
    buf = ctx.create_buffer(np.zeros(4, dtype=np.float32),
                            bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    pool = ctx.create_descriptor_pool(max_sets=8, storage_buffers=8)
    dset = pool.allocate_set(pipeline, set=0)
    dset.set_buffer(0, buf)
    return pipeline, buf, dset, pool


def counting_pass(graph, pipeline, dset, **kwargs):
    p = graph.add_pass(**kwargs)
    p.bind_pipeline(pipeline).bind_descriptor_set(dset, pipeline, set=0).dispatch(1)
    return p


# ── sealing ───────────────────────────────────────────────────────────────


def test_a_removed_pass_refuses_every_verb(ctx):
    pipeline, buf, dset, pool = add_one(ctx)
    g = ctx.graph()
    p = counting_pass(g, pipeline, dset)
    g.remove(p)
    with pytest.raises(bz.StateError, match="removed"):
        p.dispatch(1)


def test_reset_removes_the_passes(ctx):
    pipeline, buf, dset, pool = add_one(ctx)
    g = ctx.graph()
    p = counting_pass(g, pipeline, dset)
    g.reset()
    with pytest.raises(bz.StateError, match="removed"):
        p.dispatch(1)


def test_removing_a_foreign_pass_is_refused(ctx):
    pipeline, buf, dset, pool = add_one(ctx)
    g = ctx.graph()
    other = ctx.graph()
    p = counting_pass(g, pipeline, dset)
    with pytest.raises(bz.StateError, match="does not belong"):
        other.remove(p)


# ── what a submit actually runs ───────────────────────────────────────────


def test_passes_run_in_add_order(ctx):
    """The graph never reorders. On one queue a topological sort could only
    produce the order the caller wrote or a surprise, and determinism is a
    prototyping feature."""
    comp_double = ctx.compile_shader(str(SHADER_DIR / "double.comp"), bz.ShaderStage.COMPUTE)
    comp_add = ctx.compile_shader(str(SHADER_DIR / "add_one.comp"), bz.ShaderStage.COMPUTE)
    doubler = ctx.compute_pipeline().shader(comp_double).storage_buffer(0).build()
    adder = ctx.compute_pipeline().shader(comp_add).storage_buffer(0).build()

    buf = ctx.create_buffer(np.full(4, 3.0, dtype=np.float32),
                            bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    pool = ctx.create_descriptor_pool(max_sets=8, storage_buffers=8)
    d_double = pool.allocate_set(doubler, set=0)
    d_double.set_buffer(0, buf)
    d_add = pool.allocate_set(adder, set=0)
    d_add.set_buffer(0, buf)

    g = ctx.graph()
    g.add_pass(name="double").bind_pipeline(doubler) \
        .bind_descriptor_set(d_double, doubler, set=0).dispatch(1)
    g.add_pass(name="add").bind_pipeline(adder) \
        .bind_descriptor_set(d_add, adder, set=0).dispatch(1)
    ctx.submit(g)

    # (3 * 2) + 1, not (3 + 1) * 2.
    assert np.allclose(buf.read(np.float32), 7.0)


def test_a_graph_replays_every_submit(ctx):
    """Record once, submit every frame: the retained idiom."""
    pipeline, buf, dset, pool = add_one(ctx)
    g = ctx.graph()
    counting_pass(g, pipeline, dset)

    for _ in range(3):
        ctx.submit(g)

    assert np.allclose(buf.read(np.float32), 3.0)


def test_reset_and_rebuild_is_the_other_idiom(ctx):
    """The rebuild-per-frame idiom, and the successor of cmd.begin(): the
    per-slot command buffers are reused, so it allocates nothing on the
    device."""
    pipeline, buf, dset, pool = add_one(ctx)
    g = ctx.graph()

    for _ in range(3):
        g.reset()
        counting_pass(g, pipeline, dset)
        ctx.submit(g)

    assert np.allclose(buf.read(np.float32), 3.0)


# ── enabled and remove change what the next submit runs ───────────────────


def test_a_disabled_pass_does_not_run(ctx):
    pipeline, buf, dset, pool = add_one(ctx)
    g = ctx.graph()
    p = counting_pass(g, pipeline, dset)

    ctx.submit(g)
    assert np.allclose(buf.read(np.float32), 1.0)

    p.enabled = False
    ctx.submit(g)
    assert np.allclose(buf.read(np.float32), 1.0), "the disabled pass ran"

    p.enabled = True
    ctx.submit(g)
    assert np.allclose(buf.read(np.float32), 2.0), "the re-enabled pass did not run"


def test_enabled_is_readable_and_defaults_to_true(ctx):
    pipeline, buf, dset, pool = add_one(ctx)
    p = counting_pass(ctx.graph(), pipeline, dset)
    assert p.enabled is True
    p.enabled = False
    assert p.enabled is False


def test_a_pass_reports_the_name_it_was_given(ctx):
    """The name is a debug label — it reaches a capture, and it reads back off
    the handle. It is never a key: passes are handles, and what orders them is
    what they use."""
    g = ctx.graph()
    assert g.add_pass(name="shadow").name == "shadow"
    assert g.add_pass().name == ""


def test_a_removed_pass_does_not_run(ctx):
    pipeline, buf, dset, pool = add_one(ctx)
    g = ctx.graph()
    first = counting_pass(g, pipeline, dset)
    counting_pass(g, pipeline, dset)

    g.remove(first)
    ctx.submit(g)

    assert np.allclose(buf.read(np.float32), 1.0), "the removed pass still ran"


def test_a_pass_added_after_a_submit_recompiles(ctx):
    """Adding a pass to a graph that already ran must invalidate the compiled
    barriers, or the new pass would replay with the old schedule."""
    pipeline, buf, dset, pool = add_one(ctx)
    g = ctx.graph()
    counting_pass(g, pipeline, dset)
    ctx.submit(g)
    assert np.allclose(buf.read(np.float32), 1.0)

    counting_pass(g, pipeline, dset)
    ctx.submit(g)
    assert np.allclose(buf.read(np.float32), 3.0), "the added pass did not run"


def test_a_disabled_pass_is_not_a_hazard_for_the_ones_after_it(ctx, fullscreen_vert):
    """The fold skips a disabled pass entirely, so the barriers are computed
    for the passes that actually run. The referee is the validation fixture:
    a barrier from a use that never happens is a wrong oldLayout away from an
    error."""
    comp = ctx.compile_shader(str(SHADER_DIR / "double.comp"), bz.ShaderStage.COMPUTE)
    doubler = ctx.compute_pipeline().shader(comp).storage_buffer(0).build()
    frag = ctx.compile_shader(str(SHADER_DIR / "ssbo.frag"), bz.ShaderStage.FRAGMENT)
    target = ctx.create_render_target(32, 32)
    gfx = (ctx.graphics_pipeline()
           .vertex_shader(fullscreen_vert)
           .fragment_shader(frag)
           .storage_buffer(0, bz.ShaderStage.FRAGMENT, set=0)
           .build(target))

    sbuf = ctx.create_buffer(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
                             bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    pool = ctx.create_descriptor_pool(max_sets=8, storage_buffers=8)
    comp_set = pool.allocate_set(doubler, set=0)
    comp_set.set_buffer(0, sbuf)
    gfx_set = pool.allocate_set(gfx, set=0)
    gfx_set.set_buffer(0, sbuf)

    g = ctx.graph()
    producer = g.add_pass(name="double")
    producer.bind_pipeline(doubler).bind_descriptor_set(comp_set, doubler, set=0).dispatch(1)
    with g.add_pass(target, name="present") as p:
        p.bind_pipeline(gfx).bind_descriptor_set(gfx_set, gfx, set=0).draw(3)

    producer.enabled = False
    ctx.submit(g)

    # The dispatch never ran, so the buffer still holds its blue.
    assert np.allclose(target.color[0].read()[16, 16, :3], [0, 0, 255], atol=2)


# ── pass kinds ────────────────────────────────────────────────────────────


def test_a_pass_without_a_target_refuses_the_draw_verbs(ctx):
    g = ctx.graph()
    p = g.add_pass()
    for call in (lambda: p.draw(3),
                 lambda: p.draw_indexed(3),
                 lambda: p.set_viewport(0, 0, 16, 16),
                 lambda: p.occlusion_query()):
        with pytest.raises(bz.StateError, match="no render target"):
            call()


def test_a_render_pass_refuses_the_compute_and_transfer_verbs(ctx):
    target = ctx.create_render_target(16, 16)
    buf = ctx.create_buffer(np.zeros(4, dtype=np.float32),
                            bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    g = ctx.graph()
    p = g.add_pass(target)
    for call in (lambda: p.dispatch(1),
                 lambda: p.fill_buffer(buf, 0),
                 lambda: p.copy_buffer(buf, buf),
                 lambda: p.barrier(buf, bz.Access.SHADER_WRITE, bz.Access.SHADER_READ)):
        with pytest.raises(bz.StateError, match="rendering scope"):
            call()


def test_the_verbs_both_kinds_share(ctx):
    """bind_pipeline, bind_descriptor_set, push_constants, timer and label
    belong to both: they describe state, not work."""
    pipeline, buf, dset, pool = add_one(ctx)
    target = ctx.create_render_target(16, 16)
    g = ctx.graph()
    general = g.add_pass()
    render = g.add_pass(target)
    for p in (general, render):
        assert p.bind_pipeline(pipeline) is p
        assert p.bind_descriptor_set(dset, pipeline, set=0) is p
        assert p.timer() is not None
        assert p.begin_label("x") is p
        assert p.end_label() is p


def test_clear_arguments_need_a_target(ctx):
    """The clear kwargs answer what happens to the attachments, and a pass
    without a target has none — so the overload without a target does not
    take them at all."""
    g = ctx.graph()
    with pytest.raises(TypeError):
        g.add_pass(clear_color=[0, 0, 0, 1])


def test_the_queue_enum_gained_compute_as_a_value(ctx):
    """0.28 shipped the parameter and 0.29 the second value, which is what made
    async compute additive: a program written against 0.28 schedules exactly as
    it did, and the new queue is one more thing to pass rather than a new way
    to say anything."""
    assert list(bz.Queue.__members__) == ["GRAPHICS", "COMPUTE"]
    assert int(bz.Queue.COMPUTE) == 1
    assert ctx.graph().add_pass(queue=bz.Queue.COMPUTE) is not None


def test_a_render_pass_refuses_the_compute_queue(ctx):
    """A compute queue has no rasterizer, so a pass with a target cannot run
    there. Refused where the pass is made rather than at submit, because the
    target is what says it draws."""
    target = ctx.create_render_target(16, 16)
    g = ctx.graph()
    with pytest.raises(bz.StateError, match="cannot draw"):
        g.add_pass(target, queue=bz.Queue.COMPUTE)


def test_a_compute_queue_pass_refuses_the_blit_verbs(ctx):
    """vkCmdBlitImage needs a graphics queue, and generate_mipmaps is a chain
    of blits. Both are legal on a target-less pass, so the QUEUE is what
    refuses them — and it refuses by Queue.COMPUTE rather than by the family
    the device happens to have, so the contract reads the same everywhere."""
    # From an array rather than empty: a never-written image is still in
    # UNDEFINED, and copying out of one trips validation on any queue.
    src = ctx.create_image(np.zeros((32, 32, 4), np.uint8))
    dst = ctx.create_image(16, 16, bz.Format.RGBA8)
    mipped = ctx.create_image(32, 32, bz.Format.RGBA8, mip_levels=4)
    g = ctx.graph()
    with g.add_pass(name="compute", queue=bz.Queue.COMPUTE) as p:
        with pytest.raises(bz.StateError, match="Queue.GRAPHICS"):
            p.blit_image(src, dst)
        with pytest.raises(bz.StateError, match="Queue.GRAPHICS"):
            p.generate_mipmaps(mipped)
        # An occlusion query needs no new guard: it is a render verb, and a
        # pass on the compute queue can never have a target.
        with pytest.raises(bz.StateError, match="no render target"):
            p.occlusion_query()
        # What a compute pass CAN do: copies, clears and fills are legal on a
        # compute family, and the suite's referee is the validation layers.
        p.copy_image(src, ctx.create_image(32, 32, bz.Format.RGBA8))
        p.fill_buffer(ctx.create_buffer(16, bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC), 0)
        p.clear_image(dst, [0, 0, 0, 1])
    ctx.submit(g)


def test_passes_on_two_queues_run_in_add_order(ctx):
    """Two queues change nothing about the order the caller wrote. The three
    passes are not commutative, so a reordered or unsynchronized run gives a
    different number: (3*2)+1 then *2 is 14, any other order is not."""
    pipeline, _, _, pool = add_one(ctx)
    buf = ctx.create_buffer(np.full(4, 3.0, np.float32), bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    dset = pool.allocate_set(pipeline)
    dset.set_buffer(0, buf)
    double = (ctx.compute_pipeline()
              .shader(ctx.compile_shader(str(SHADER_DIR / "double.comp"), bz.ShaderStage.COMPUTE))
              .storage_buffer(0)
              .build())
    double_set = pool.allocate_set(double)
    double_set.set_buffer(0, buf)

    g = ctx.graph()
    (g.add_pass(name="double on graphics")
        .bind_pipeline(double).bind_descriptor_set(double_set, double).dispatch(1))
    (g.add_pass(name="add one on compute", queue=bz.Queue.COMPUTE)
        .bind_pipeline(pipeline).bind_descriptor_set(dset, pipeline).dispatch(1))
    (g.add_pass(name="double again on graphics")
        .bind_pipeline(double).bind_descriptor_set(double_set, double).dispatch(1))
    ctx.submit(g)

    assert np.allclose(buf.read(np.float32), [14.0] * 4)


def test_toggling_a_pass_recomputes_the_batches(ctx):
    """A disabled pass is absent from the compile, and with two queues that
    changes which batches exist at all — not only which barriers they carry."""
    pipeline, _, _, pool = add_one(ctx)
    double = (ctx.compute_pipeline()
              .shader(ctx.compile_shader(str(SHADER_DIR / "double.comp"), bz.ShaderStage.COMPUTE))
              .storage_buffer(0)
              .build())

    def fresh_buffer():
        buf = ctx.create_buffer(np.full(4, 3.0, np.float32), bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
        add_set = pool.allocate_set(pipeline)
        add_set.set_buffer(0, buf)
        double_set = pool.allocate_set(double)
        double_set.set_buffer(0, buf)
        return buf, add_set, double_set

    buf, dset, double_set = fresh_buffer()
    g = ctx.graph()
    first = (g.add_pass(name="double", queue=bz.Queue.GRAPHICS)
             .bind_pipeline(double).bind_descriptor_set(double_set, double).dispatch(1))
    middle = (g.add_pass(name="add one", queue=bz.Queue.COMPUTE)
              .bind_pipeline(pipeline).bind_descriptor_set(dset, pipeline).dispatch(1))
    (g.add_pass(name="double again", queue=bz.Queue.GRAPHICS)
        .bind_pipeline(double).bind_descriptor_set(double_set, double).dispatch(1))

    ctx.submit(g)
    assert np.allclose(buf.read(np.float32), [14.0] * 4)

    # One batch instead of three: the two graphics passes become consecutive.
    middle.enabled = False
    ctx.submit(g)
    assert np.allclose(buf.read(np.float32), [56.0] * 4), "the disabled pass ran"

    middle.enabled = True
    ctx.submit(g)
    assert np.allclose(buf.read(np.float32), [226.0] * 4)
    assert first.enabled


def test_a_compute_queue_pass_measures_a_timer(ctx):
    """A timer asks the family that REPLAYS the pass whether its timestamps are
    usable, not the graphics family. Both answers are legitimate; what would be
    wrong is reading the graphics family's answer for a compute queue."""
    pipeline, buf, dset, pool = add_one(ctx)
    g = ctx.graph()
    with g.add_pass(name="timed", queue=bz.Queue.COMPUTE) as p:
        with p.timer() as t:
            p.bind_pipeline(pipeline).bind_descriptor_set(dset, pipeline).dispatch(1)
    ctx.submit(g)
    assert t.ms is None or t.ms >= 0.0


# The per-frame claim ("Each window needs its own Graph") is NOT tested here,
# and that is a property of the headless path rather than an omission: a
# headless submit advances the ring itself, so two of them never share a frame
# serial and the claim can never fire. It takes a present and a submit inside
# one frame, which needs a window — see tests/test_multi_window.py.


# ── query handles across a reset ──────────────────────────────────────────


def test_a_timer_from_before_a_reset_is_stale(ctx):
    """The handle's slots belong to a recording that no longer exists. Reading
    it must say so rather than return a number from a different timer."""
    pipeline, buf, dset, pool = add_one(ctx)
    g = ctx.graph()
    p = counting_pass(g, pipeline, dset)
    t = p.timer()
    t.stop()
    ctx.submit(g)
    assert t.ms is not None

    g.reset()
    with pytest.raises(bz.StateError):
        _ = t.ms
