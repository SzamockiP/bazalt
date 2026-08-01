"""0.14: N windows on one Context, and the frame/window split that enables it.

Two swapchains mean two surfaces, so most of this cannot run on CI's lavapipe —
those tests skip without a display, exactly like the gpu_time_ms test in
test_diagnostics. What *is* headless-testable is the half that moved onto the
Context: begin_frame owning the ring.
"""

import pathlib

import pytest

import bazalt as bz

SHADER_DIR = pathlib.Path(__file__).parent / "shaders"


def _two_windows():
    """Two windows and their renderers, or a skip. Returned rather than
    fixtured: the caller must drop them in a finally, since each holds the
    session Context (and a surface) alive past the test."""
    try:
        a = bz.Window(64, 64, "bazalt multi-window A")
        b = bz.Window(64, 64, "bazalt multi-window B")
    except bz.WindowError:
        pytest.skip("no display available")
    return a, b


def _solid_pipeline(ctx, target):
    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    return ctx.graphics_pipeline().vertex_shader(vert).fragment_shader(frag).build(target)


# ── headless: the frame is the Context's ──────────────────────────────────────


def test_begin_frame_advances_the_ring_exactly_once(ctx):
    """The whole reason a second window is possible: one call, one slot."""
    start = ctx.frame_index
    ctx.begin_frame()
    assert ctx.frame_index == (start + 1) % ctx.frames_in_flight
    ctx.begin_frame()
    assert ctx.frame_index == (start + 2) % ctx.frames_in_flight


def test_frame_index_stays_inside_the_ring(ctx):
    for _ in range(2 * ctx.frames_in_flight + 1):
        ctx.begin_frame()
        assert 0 <= ctx.frame_index < ctx.frames_in_flight


def test_poll_events_without_a_window_is_an_error():
    """Pumping an event queue that no window feeds is a bug, not a no-op.

    GLFW is only initialized while a Window exists, so glfwPollEvents would set
    GLFW_NOT_INITIALIZED and return — and before the first window has ever
    existed bazalt has not installed its error callback yet, so the call would
    vanish without a trace. Runs headless: it asserts the absence of windows."""
    with pytest.raises(bz.WindowError):
        bz.poll_events()


def test_poll_events_works_while_a_window_is_open(ctx):
    """The other half of the guard: it must not fire on the normal path."""
    if ctx.headless:
        pytest.skip("no swapchain support (headless Context)")
    try:
        window = bz.Window(64, 64, "bazalt poll guard")
    except bz.WindowError:
        pytest.skip("no display available")
    try:
        bz.poll_events()
    finally:
        window = None


# ── windowed: two swapchains, one Context ─────────────────────────────────────


def test_two_windows_share_one_context(ctx):
    """The 0.13 guard refused this outright. Both windows draw for several
    frames; the validation-as-assert fixture is the referee for the semaphores,
    fences and layout transitions each swapchain does on its own."""
    if ctx.headless:
        pytest.skip("no swapchain support (headless Context)")
    window_a, window_b = _two_windows()
    renderer_a = renderer_b = None
    try:
        renderer_a = ctx.create_renderer(window_a)
        renderer_b = ctx.create_renderer(window_b)
        pipeline = _solid_pipeline(ctx, renderer_a)

        # One CommandBuffer per window: they own one command buffer per ring
        # slot, and both windows render on the same slot now.
        cmds = []
        for renderer in (renderer_a, renderer_b):
            cmd = ctx.create_command_buffer()
            cmd.begin()
            cmd.begin_rendering(renderer, clear_color=[0, 0, 0, 1])
            cmd.bind_pipeline(pipeline)
            cmd.draw(3)
            cmd.end_rendering(renderer)
            cmds.append(cmd)

        presented = 0
        for _ in range(ctx.frames_in_flight + 4):
            bz.poll_events()
            ctx.begin_frame()
            for renderer, cmd in zip((renderer_a, renderer_b), cmds):
                if renderer.acquire():
                    renderer.present(cmd)
                    presented += 1

        assert presented > 0, "neither window ever acquired an image"
    finally:
        renderer_a = renderer_b = None
        window_a = window_b = None


def test_both_windows_render_on_the_same_ring_slot(ctx):
    """The bug the split exists to prevent: if each window advanced the ring,
    the second would render one slot ahead of every update() the caller made."""
    if ctx.headless:
        pytest.skip("no swapchain support (headless Context)")
    window_a, window_b = _two_windows()
    renderer_a = renderer_b = None
    try:
        renderer_a = ctx.create_renderer(window_a)
        renderer_b = ctx.create_renderer(window_b)

        bz.poll_events()
        ctx.begin_frame()
        slot = ctx.frame_index
        renderer_a.acquire()
        renderer_b.acquire()
        assert ctx.frame_index == slot, "a window advanced the frame ring"
    finally:
        renderer_a = renderer_b = None
        window_a = window_b = None


def test_one_command_buffer_cannot_serve_two_windows(ctx):
    """A CommandBuffer holds one VkCommandBuffer per ring slot, so replaying it
    in a second window would reset a buffer the first still has in flight.
    Validation would report a pending-state VUID; this says it in a sentence,
    and says it in builds with no validation layers at all."""
    if ctx.headless:
        pytest.skip("no swapchain support (headless Context)")
    window_a, window_b = _two_windows()
    renderer_a = renderer_b = None
    try:
        renderer_a = ctx.create_renderer(window_a)
        renderer_b = ctx.create_renderer(window_b)
        pipeline = _solid_pipeline(ctx, renderer_a)

        cmd = ctx.create_command_buffer()
        cmd.begin()
        cmd.begin_rendering(renderer_a, clear_color=[0, 0, 0, 1])
        cmd.bind_pipeline(pipeline)
        cmd.draw(3)
        cmd.end_rendering(renderer_a)

        bz.poll_events()
        ctx.begin_frame()
        if not (renderer_a.acquire() and renderer_b.acquire()):
            pytest.skip("windows did not both acquire (minimized?)")

        renderer_a.present(cmd)
        with pytest.raises(bz.StateError):
            renderer_b.present(cmd)
    finally:
        renderer_a = renderer_b = None
        window_a = window_b = None


def test_acquire_twice_without_begin_frame_is_an_error(ctx):
    """One check catches both the double acquire and the loop that forgot
    ctx.begin_frame() — they are the same mistake seen from two sides."""
    if ctx.headless:
        pytest.skip("no swapchain support (headless Context)")
    try:
        window = bz.Window(64, 64, "bazalt acquire guard")
    except bz.WindowError:
        pytest.skip("no display available")
    renderer = None
    try:
        renderer = ctx.create_renderer(window)
        bz.poll_events()
        ctx.begin_frame()
        renderer.acquire()
        with pytest.raises(bz.StateError):
            renderer.acquire()
    finally:
        renderer = None
        window = None


def test_present_without_an_acquired_image_is_an_error(ctx):
    if ctx.headless:
        pytest.skip("no swapchain support (headless Context)")
    try:
        window = bz.Window(64, 64, "bazalt present guard")
    except bz.WindowError:
        pytest.skip("no display available")
    renderer = None
    try:
        renderer = ctx.create_renderer(window)
        pipeline = _solid_pipeline(ctx, renderer)
        cmd = ctx.create_command_buffer()
        cmd.begin()
        cmd.begin_rendering(renderer, clear_color=[0, 0, 0, 1])
        cmd.bind_pipeline(pipeline)
        cmd.draw(3)
        cmd.end_rendering(renderer)

        ctx.begin_frame()
        with pytest.raises(bz.StateError):
            renderer.present(cmd)
    finally:
        renderer = None
        window = None
