"""Two small 0.8 diagnostics additions: GPU frame timing (frame.gpu_time_ms)
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

    assert target.read_pixels().shape == (8, 8, 4)
    assert buf is not None and img is not None


def test_empty_name_is_a_no_op(ctx):
    """The default (no name=) must behave exactly as before."""
    img = ctx.create_image(4, 4)
    assert img.width == 4


def test_scope_timer_reports_positive_time_headless(ctx):
    """`with cmd.timer(name):` measures a slice of the recording and is readable
    right after a blocking headless submit — no window, no begin_frame. On a
    device without timestamp support (some CI rasterizers) timer_ms returns None,
    which is the documented best-effort contract, not a failure."""
    comp = ctx.compile_shader(str(SHADER_DIR / "double.comp"), bz.ShaderStage.COMPUTE)
    pipeline = ctx.compute_pipeline().shader(comp).storage_buffer(0).build()

    import numpy as np
    sbuf = ctx.create_buffer(np.arange(4096, dtype=np.float32),
                             bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    pool = ctx.create_descriptor_pool(max_sets=4, storage_buffers=4)
    dset = pool.allocate_set(pipeline, set=0)
    dset.set_buffer(0, sbuf)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.timer("double"):
        cmd.bind_pipeline(pipeline)
        cmd.bind_descriptor_set(dset, pipeline, set=0)
        cmd.dispatch(64)
    ctx.submit(cmd)

    assert cmd.timer_ms("never recorded") is None
    t = cmd.timer_ms("double")
    if t is not None:  # None only on a device without timestamp support
        assert t >= 0.0


def test_gpu_time_ms_is_reported_after_the_ring_cycles(ctx):
    """frame.gpu_time_ms is None until the frame ring has cycled once, then a
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
            window.poll_events()
            frame = renderer.begin_frame()
            if frame is None:
                continue
            times.append(frame.gpu_time_ms)
            cmd = ctx.create_command_buffer()
            cmd.begin()
            cmd.begin_rendering(renderer, clear_color=[0, 0, 0, 1])
            cmd.bind_pipeline(pipeline)
            cmd.draw(3)
            cmd.end_rendering(renderer)
            frame.submit(cmd)

        assert times, "expected at least one acquired frame"
        assert times[0] is None, "the first frame has no prior submission to time"
        measured = [t for t in times if t is not None]
        assert measured, "gpu_time_ms should become available once the ring cycles"
        assert all(t > 0 for t in measured), f"GPU times must be positive: {measured}"
    finally:
        # Drop the renderer before the next test: only one SwapchainRenderer may
        # exist per Context, and it holds the shared session Context alive.
        renderer = None
        window = None
        gc.collect()
