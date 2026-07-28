"""Two small 0.8 diagnostics additions: GPU frame timing (renderer.gpu_time_ms)
and debug object names (name= / .name())."""

import gc
import pathlib

import pytest

import bazalt as bz

SHADER_DIR = pathlib.Path(__file__).parent / "shaders"


def test_debug_names_are_accepted_and_render_cleanly(ctx):
    """name= / .name() attach a debug name and never break rendering. The real
    assertion is the ctx fixture's zero-validation-errors check — a bad object
    name (or a handle/type mismatch) would surface there. Asserting the name
    text inside a validation message would need a provoked error and is
    layer-version specific, so it's a manual check only."""
    buf = ctx.create_buffer([0.0, 0.0, 0.0], bz.BufferType.VERTEX,
                            bz.MemoryUsage.STATIC, bz.DataType.FLOAT, name="verts")
    img = ctx.create_image(8, 8, name="scratch")
    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, 8, 8)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .name("solid_red_pipeline")
                .build(target))

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(target, clear_color=[0, 0, 0, 1])
    cmd.bind_pipeline(pipeline)
    cmd.draw(3)
    cmd.end_rendering(target)
    ctx.submit(cmd)

    assert target.color[0].read().shape == (8, 8, 4)
    assert buf is not None and img is not None


def test_empty_name_is_a_no_op(ctx):
    """The default (no name=) must behave exactly as before."""
    img = ctx.create_image(4, 4)
    assert img.width == 4


def test_sampler_names_accumulate_on_the_shared_object(ctx):
    """The sampler cache keys on the description, so two differently named
    requests for the same filtering are ONE VkSampler.

    Naming only the first caller would drop a name silently, and putting the
    name in the cache key would make a debug label change what gets allocated.
    So the shared object lists its users, and .name says so.
    """
    first = ctx.create_sampler(filter=bz.Filter.NEAREST, anisotropy=False, name="shadow_pcf")
    second = ctx.create_sampler(filter=bz.Filter.NEAREST, anisotropy=False, name="terrain")

    assert first is second, "identical descriptions must still share one sampler"
    assert first.name == "shadow_pcf + terrain"

    # Naming it the same thing twice adds nothing.
    third = ctx.create_sampler(filter=bz.Filter.NEAREST, anisotropy=False, name="terrain")
    assert third.name == "shadow_pcf + terrain"

    # A different description is a different sampler, and starts unnamed.
    other = ctx.create_sampler(filter=bz.Filter.LINEAR, anisotropy=False)
    assert other is not first
    assert other.name == ""


def _double_pipeline(ctx):
    comp = ctx.compile_shader(str(SHADER_DIR / "double.comp"), bz.ShaderStage.COMPUTE)
    return ctx.compute_pipeline().shader(comp).storage_buffer(0).build()


def test_timer_handle_reports_positive_time_headless(ctx):
    """A timer handle measures a slice of the recording and is readable right
    after a blocking headless submit — no window, no frame loop. Both the
    `with` form and the explicit stop() are exercised, plus two overlapping
    timers. On a device without timestamp support .ms is None (documented
    best-effort), not a failure."""
    import numpy as np
    pipeline = _double_pipeline(ctx)
    sbuf = ctx.create_buffer(np.arange(4096, dtype=np.float32),
                             bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    pool = ctx.create_descriptor_pool(max_sets=4, storage_buffers=4)
    dset = pool.allocate_set(pipeline, set=0)
    dset.set_buffer(0, sbuf)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    whole = cmd.timer()  # explicit stop, spans everything
    with cmd.timer() as inner:  # with-form, nested
        cmd.bind_pipeline(pipeline)
        cmd.bind_descriptor_set(dset, pipeline, set=0)
        cmd.dispatch(64)
    whole.stop()
    ctx.submit(cmd)

    for t in (whole.ms, inner.ms):
        if t is not None:  # None only without timestamp support
            assert t >= 0.0
    if whole.ms is not None and inner.ms is not None:
        assert whole.ms >= inner.ms  # the outer scope contains the inner one


def test_stale_timer_handle_reads_none(ctx):
    """A handle from a superseded recording reports None instead of a stale
    number: begin() bumps the recording generation."""
    pipeline = _double_pipeline(ctx)
    import numpy as np
    sbuf = ctx.create_buffer(np.arange(64, dtype=np.float32),
                             bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    pool = ctx.create_descriptor_pool(max_sets=4, storage_buffers=4)
    dset = pool.allocate_set(pipeline, set=0)
    dset.set_buffer(0, sbuf)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.timer() as old:
        cmd.bind_pipeline(pipeline).bind_descriptor_set(dset, pipeline, set=0).dispatch(1)
    ctx.submit(cmd)

    cmd.begin()  # re-record: `old` now belongs to a superseded recording
    with cmd.timer():
        cmd.bind_pipeline(pipeline).bind_descriptor_set(dset, pipeline, set=0).dispatch(1)
    ctx.submit(cmd)

    assert old.ms is None


def test_gpu_time_ms_is_reported_after_the_ring_cycles(ctx):
    """renderer.gpu_time_ms is None until the frame ring has cycled once, then a
    positive float. Windowed only (headless submit is a blocking wait-idle),
    so this skips without a swapchain/display — e.g. on CI's lavapipe."""
    if ctx.headless:
        pytest.skip("no swapchain support (headless Context)")
    try:
        window = bz.Window(64, 64, "bazalt gpu_time test")
    except bz.WindowError:
        pytest.skip("no display available")

    try:
        renderer = bz.SwapchainRenderer(window, ctx)
        vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
        frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
        pipeline = (ctx.graphics_pipeline()
                    .vertex_shader(vert)
                    .fragment_shader(frag)
                    .build(renderer))

        times = []
        for _ in range(ctx.frames_in_flight + 5):
            bz.poll_events()
            ctx.begin_frame()
            if not renderer.acquire():
                continue
            times.append(renderer.gpu_time_ms)
            cmd = ctx.create_command_buffer()
            cmd.begin()
            cmd.begin_rendering(renderer, clear_color=[0, 0, 0, 1])
            cmd.bind_pipeline(pipeline)
            cmd.draw(3)
            cmd.end_rendering(renderer)
            renderer.present(cmd)

        assert times, "expected at least one acquired frame"
        assert times[0] is None, "the first frame has no prior submission to time"
        measured = [t for t in times if t is not None]
        assert measured, "gpu_time_ms should become available once the ring cycles"
        assert all(t > 0 for t in measured), f"GPU times must be positive: {measured}"
    finally:
        # Drop the renderer before the next test: it holds the shared session
        # Context alive, and its surface belongs to a window going away here.
        renderer = None
        window = None
        gc.collect()


# ── 0.18: labels, occlusion queries, memory and subgroup introspection ──


def solid_pipeline(ctx, target, depth_test=False):
    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    builder = ctx.graphics_pipeline().vertex_shader(vert).fragment_shader(frag)
    if depth_test:
        builder = builder.depth_test(True)
    return builder.build(target)


def test_debug_labels_render_cleanly(ctx):
    """As with debug names, the referee is the ctx fixture: an unbalanced
    begin/end pair is a validation error, and that is what could actually break.
    Whether RenderDoc groups the draws is not observable from here."""
    target = bz.RenderTarget(ctx, 8, 8)
    pipeline = solid_pipeline(ctx, target)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.label("outer"):
        with cmd.rendering(target, clear_color=[0, 0, 0, 1]):
            with cmd.label("inner"):
                cmd.bind_pipeline(pipeline).draw(3)
    ctx.submit(cmd)

    assert target.color[0].read()[0, 0, 0] > 200


def test_labels_nest(ctx):
    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.label("outer"):
        with cmd.label("inner"):
            pass
    ctx.submit(cmd)


def test_end_label_without_a_begin_is_ignored(ctx):
    """Recording the unbalanced end is undefined behaviour in Vulkan, so the
    verb drops it. The `with` form cannot produce one; this covers the explicit
    pair, which is there for a recording split across functions and which
    someone will eventually mismatch."""
    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.end_label()
    cmd.begin_label("one").end_label().end_label()
    ctx.submit(cmd)


def test_occlusion_query_counts_fragments(ctx):
    """A fullscreen triangle over an 8x8 target covers every pixel, so the
    query has a known floor. Not an exact equality: without occlusionQueryPrecise
    the spec only promises a non-zero value, and helper invocations can push a
    precise count above the pixel count."""
    target = bz.RenderTarget(ctx, 8, 8)
    pipeline = solid_pipeline(ctx, target)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(target, clear_color=[0, 0, 0, 1]):
        with cmd.occlusion_query() as q:
            cmd.bind_pipeline(pipeline).draw(3)
    ctx.submit(cmd)

    assert q.samples is not None
    assert q.samples > 0


def test_occlusion_query_reports_zero_when_nothing_is_drawn(ctx):
    """The two-sided half: a query that always answered "lots" would pass the
    test above while measuring nothing."""
    target = bz.RenderTarget(ctx, 8, 8)
    solid_pipeline(ctx, target)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(target, clear_color=[0, 0, 0, 1]):
        with cmd.occlusion_query() as q:
            pass
    ctx.submit(cmd)

    assert q.samples == 0


def test_occlusion_query_outside_a_rendering_scope_raises(ctx):
    """Vulkan requires the query to begin and end in one render pass. Refusing
    at the call site beats a validation message at submit that names neither."""
    cmd = ctx.create_command_buffer()
    cmd.begin()
    with pytest.raises(bz.ResourceError):
        cmd.occlusion_query()


def test_stale_occlusion_handle_reads_none(ctx):
    """Same stale-handle contract as a Timer: re-recording gives the slots to a
    different query, so the old handle reports None instead of a wrong number."""
    target = bz.RenderTarget(ctx, 8, 8)
    pipeline = solid_pipeline(ctx, target)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(target, clear_color=[0, 0, 0, 1]):
        with cmd.occlusion_query() as q:
            cmd.bind_pipeline(pipeline).draw(3)
    ctx.submit(cmd)
    assert q.samples is not None

    cmd.begin()
    ctx.submit(cmd)
    assert q.samples is None


def test_memory_stats_grow_with_an_allocation(ctx):
    """A number that never moves is indistinguishable from a stub."""
    before = ctx.memory_stats()
    assert before.budget > 0

    big = ctx.create_image(512, 512, bz.Format.RGBA32F)
    after = ctx.memory_stats()

    assert after.used > before.used
    assert after.reserved >= after.used
    assert big.width == 512


def test_memory_stats_shrink_again(ctx):
    """Deferred destruction means the drop is not instant, so this flushes the
    queue the way a real frame loop would."""
    big = ctx.create_image(512, 512, bz.Format.RGBA32F)
    peak = ctx.memory_stats().used

    del big
    gc.collect()
    ctx.wait()
    for _ in range(ctx.frames_in_flight + 1):
        ctx.begin_frame()

    assert ctx.memory_stats().used < peak


def test_subgroup_size_is_a_power_of_two(ctx):
    """Zero is legal (a driver need not report one), so the assertion covers
    both: either unreported, or a plausible width."""
    size = ctx.subgroup_size
    assert size == 0 or (size & (size - 1)) == 0
    assert size <= 128


def test_renderer_read_pixels_captures_the_frame(ctx):
    """A screenshot of a window. Only offscreen targets could be read back
    before 0.18, so a windowed prototype could not save the picture it drew —
    the one thing a prototype exists to do.

    Two calls on purpose. A presentable image may only be touched between
    acquire and present, so "read the last frame" is illegal by the spec and the
    validation layer says so. present(capture=True) records the copy into the
    frame's own submit; read_pixels() collects it.

    Needs a display, so CI skips it. The channel order is asserted because it is
    the part that varies per machine: most compositors hand out BGRA, and
    out[y, x, 0] has to mean red anyway.
    """
    if ctx.headless:
        pytest.skip("no swapchain support (headless Context)")
    try:
        window = bz.Window(64, 64, "bazalt read_pixels")
    except bz.WindowError:
        pytest.skip("no display available")

    renderer = None
    try:
        renderer = bz.SwapchainRenderer(window, ctx)
        vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
        frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
        pipeline = (ctx.graphics_pipeline()
                    .vertex_shader(vert)
                    .fragment_shader(frag)
                    .build(renderer))

        # Nothing captured yet, and saying so beats handing back an empty buffer.
        with pytest.raises(bz.ResourceError):
            renderer.read_pixels()

        drawn = False
        for _ in range(3):
            bz.poll_events()
            ctx.begin_frame()
            if not renderer.acquire():
                continue
            cmd = ctx.create_command_buffer()
            cmd.begin()
            with cmd.rendering(renderer, clear_color=[0, 0, 0, 1]):
                cmd.bind_pipeline(pipeline).draw(3)
            renderer.present(cmd, capture=True)
            drawn = True

        if not drawn:
            pytest.skip("the window never produced a frame")

        shot = renderer.read_pixels()
        assert shot.shape == (renderer.height, renderer.width, 4)
        assert shot[shot.shape[0] // 2, shot.shape[1] // 2, 0] > 200, "expected red"
        assert shot[shot.shape[0] // 2, shot.shape[1] // 2, 1] < 60
    finally:
        renderer = None
        window = None
        gc.collect()
