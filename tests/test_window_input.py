"""0.16: window modes, window attributes and the per-poll input state.

Both halves need a real window, so both skip without a display — the same
position test_multi_window is in, and for the same reason. CI's lavapipe has no
surface to show, so the coverage that matters there is test_load_op and
test_pipeline_state.

The mode tests render a frame in each mode on purpose: switching mode resizes
the framebuffer, which recreates the swapchain, and the `ctx` fixture is what
says whether the semaphores and layout transitions survived it.

The input tests can only assert the shape of the surface. OS input cannot be
synthesized here, so what they pin down is the part that is ours: a query reads
the same value twice inside one frame, and it starts at rest.
"""

import pathlib
import time

import pytest

import bazalt as bz

SHADER_DIR = pathlib.Path(__file__).parent / "shaders"

ALL_MODES = (bz.WindowMode.WINDOWED, bz.WindowMode.FRAMELESS,
             bz.WindowMode.FULLSCREEN, bz.WindowMode.FULLSCREEN_WINDOWED)


def a_window(*, mode=bz.WindowMode.WINDOWED, width=128, height=96):
    """One window, or a skip. Returned rather than fixtured: it holds the
    session Context alive, so the caller drops it in a finally."""
    try:
        return bz.Window(width, height, "bazalt window modes", mode=mode)
    except bz.WindowError:
        pytest.skip("no display available")


def solid_pipeline(ctx, target):
    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    return ctx.graphics_pipeline().vertex_shader(vert).fragment_shader(frag).build(target)


def pump(ctx, renderer, cmd, frames=3):
    """Run a few frames, returning how many actually presented."""
    presented = 0
    for _ in range(frames):
        bz.poll_events()
        ctx.begin_frame()
        if renderer.acquire():
            renderer.present(cmd)
            presented += 1
    return presented


# ── window modes ──────────────────────────────────────────────────────────


def test_every_mode_round_trips(ctx):
    """The property reports what was asked for, through all four values and back."""
    if ctx.headless:
        pytest.skip("no swapchain support (headless Context)")
    window = a_window()
    try:
        assert window.mode == bz.WindowMode.WINDOWED
        for mode in ALL_MODES:
            window.set_mode(mode)
            bz.poll_events()
            assert window.mode == mode
        window.set_mode(bz.WindowMode.WINDOWED)
        assert window.mode == bz.WindowMode.WINDOWED
    finally:
        window = None


def test_windowed_restores_the_size_it_left(ctx):
    """Leaving WINDOWED saves the rectangle, so coming back is not a guess."""
    if ctx.headless:
        pytest.skip("no swapchain support (headless Context)")
    window = a_window(width=200, height=150)
    try:
        bz.poll_events()
        before = (window.width, window.height)

        window.set_mode(bz.WindowMode.FULLSCREEN)
        bz.poll_events()
        window.set_mode(bz.WindowMode.WINDOWED)
        bz.poll_events()

        assert (window.width, window.height) == before
    finally:
        window = None


def test_a_window_can_open_in_a_mode(ctx):
    """mode= at construction goes through the same set_mode, so the geometry it
    returns to is the width and height that were asked for."""
    if ctx.headless:
        pytest.skip("no swapchain support (headless Context)")
    window = a_window(mode=bz.WindowMode.FRAMELESS, width=180, height=120)
    try:
        assert window.mode == bz.WindowMode.FRAMELESS
        bz.poll_events()
        window.set_mode(bz.WindowMode.WINDOWED)
        bz.poll_events()
        assert (window.width, window.height) == (180, 120)
    finally:
        window = None


def test_the_swapchain_follows_a_mode_change(ctx):
    """The point of the feature: a mode change is a resize, and the resize path
    already recreates the swapchain. The referee here is the validation layer,
    not the pixel."""
    if ctx.headless:
        pytest.skip("no swapchain support (headless Context)")
    window = a_window()
    renderer = None
    try:
        renderer = ctx.create_renderer(window)
        pipeline = solid_pipeline(ctx, renderer)
        cmd = ctx.create_command_buffer()
        cmd.begin()
        cmd.begin_rendering(renderer, clear_color=[0, 0, 0, 1])
        cmd.bind_pipeline(pipeline)
        cmd.draw(3)
        cmd.end_rendering(renderer)

        presented = pump(ctx, renderer, cmd)
        for mode in ALL_MODES:
            window.set_mode(mode)
            presented += pump(ctx, renderer, cmd)

        assert presented > 0, "the window never acquired an image in any mode"
    finally:
        renderer = None
        window = None


# ── window attributes ─────────────────────────────────────────────────────


def test_attributes_read_back_what_was_set(ctx):
    if ctx.headless:
        pytest.skip("no swapchain support (headless Context)")
    window = a_window()
    try:
        window.set_resizable(False)
        assert window.resizable is False
        window.set_resizable(True)
        assert window.resizable is True

        window.set_always_on_top(True)
        assert window.always_on_top is True
        window.set_always_on_top(False)
        assert window.always_on_top is False

        window.set_size(160, 120)
        bz.poll_events()
        assert (window.width, window.height) == (160, 120)

        window.set_position(80, 60)
        bz.poll_events()
        assert isinstance(window.position, tuple) and len(window.position) == 2

        scale_x, scale_y = window.content_scale
        assert scale_x > 0.0 and scale_y > 0.0
    finally:
        window = None


def test_opacity_is_reported_back(ctx):
    """Some platforms have no compositor, so the value is allowed to stay 1.0 —
    what must not happen is an error or a nonsense number."""
    if ctx.headless:
        pytest.skip("no swapchain support (headless Context)")
    window = a_window()
    try:
        window.set_opacity(0.5)
        assert 0.0 <= window.opacity <= 1.0
    finally:
        window = None


# ── present mode at runtime ───────────────────────────────────────────────


def test_present_mode_switches_at_runtime(ctx):
    """FIFO is the one mode every driver must support, so it is the one a test
    can insist on getting."""
    if ctx.headless:
        pytest.skip("no swapchain support (headless Context)")
    window = a_window()
    renderer = None
    try:
        renderer = ctx.create_renderer(window, present_mode=bz.PresentMode.FIFO)
        pipeline = solid_pipeline(ctx, renderer)
        cmd = ctx.create_command_buffer()
        cmd.begin()
        cmd.begin_rendering(renderer, clear_color=[0, 0, 0, 1])
        cmd.bind_pipeline(pipeline)
        cmd.draw(3)
        cmd.end_rendering(renderer)

        pump(ctx, renderer, cmd)
        renderer.set_present_mode(bz.PresentMode.IMMEDIATE)
        pump(ctx, renderer, cmd)
        renderer.set_present_mode(bz.PresentMode.FIFO)
        pump(ctx, renderer, cmd)

        assert renderer.present_mode == bz.PresentMode.FIFO
    finally:
        renderer = None
        window = None


def test_present_mode_cannot_change_mid_frame(ctx):
    """Recreating the swapchain would free the image the frame is holding."""
    if ctx.headless:
        pytest.skip("no swapchain support (headless Context)")
    window = a_window()
    renderer = None
    try:
        renderer = ctx.create_renderer(window, present_mode=bz.PresentMode.FIFO)
        pipeline = solid_pipeline(ctx, renderer)
        cmd = ctx.create_command_buffer()
        cmd.begin()
        cmd.begin_rendering(renderer, clear_color=[0, 0, 0, 1])
        cmd.bind_pipeline(pipeline)
        cmd.draw(3)
        cmd.end_rendering(renderer)

        bz.poll_events()
        ctx.begin_frame()
        if not renderer.acquire():
            pytest.skip("the window never acquired an image")
        with pytest.raises(bz.StateError, match="acquire"):
            renderer.set_present_mode(bz.PresentMode.IMMEDIATE)
        renderer.present(cmd)
    finally:
        renderer = None
        window = None


# ── input ─────────────────────────────────────────────────────────────────


def test_the_mouse_starts_at_rest(ctx):
    """Every field exists and reads zero before anything has moved. dx/dy are a
    per-cycle delta now, so at rest they are 0.0 and stay there."""
    if ctx.headless:
        pytest.skip("no swapchain support (headless Context)")
    window = a_window()
    try:
        bz.poll_events()
        mouse = window.get_mouse_state()
        assert mouse.dx == 0.0 and mouse.dy == 0.0
        assert mouse.scroll_dx == 0.0 and mouse.scroll_dy == 0.0
        # A position exists whether or not the cursor has entered the window.
        assert isinstance(mouse.x, float) and isinstance(mouse.y, float)
    finally:
        window = None


def test_a_per_cycle_query_is_repeatable(ctx):
    """The rotation is driven by the reader, so asking twice inside one frame
    gives the same answer. Consuming on read would answer True then False."""
    if ctx.headless:
        pytest.skip("no swapchain support (headless Context)")
    window = a_window()
    try:
        bz.poll_events()
        first = window.get_mouse_state()
        second = window.get_mouse_state()
        assert (first.dx, first.dy) == (second.dx, second.dy)
        assert window.was_key_pressed(bz.KEY_F11) == window.was_key_pressed(bz.KEY_F11)
        assert window.was_mouse_button_pressed(bz.MOUSE_BUTTON_LEFT) == \
            window.was_mouse_button_pressed(bz.MOUSE_BUTTON_LEFT)
    finally:
        window = None


def test_an_untouched_key_is_not_an_edge(ctx):
    """The edge query must not report a key nobody pressed, and it must survive
    several poll cycles without inventing one."""
    if ctx.headless:
        pytest.skip("no swapchain support (headless Context)")
    window = a_window()
    try:
        for _ in range(3):
            bz.poll_events()
            assert not window.was_key_pressed(bz.KEY_F11)
            assert not window.was_mouse_button_pressed(bz.MOUSE_BUTTON_RIGHT)
    finally:
        window = None


def test_the_input_enums_agree_with_the_bare_ints():
    """0.23: Key, MouseButton and CursorMode are renames of the GLFW values,
    exactly as GamepadButton is, so each member equals its old module int.
    Needs no window — the values are the whole claim."""
    assert int(bz.Key.W) == bz.KEY_W == 87
    assert int(bz.Key.ESCAPE) == bz.KEY_ESCAPE
    assert int(bz.Key.D0) == bz.KEY_0
    assert int(bz.Key.KP_0) == bz.KEY_KP_0
    assert int(bz.MouseButton.LEFT) == bz.MOUSE_BUTTON_LEFT
    assert int(bz.MouseButton.MIDDLE) == bz.MOUSE_BUTTON_MIDDLE
    assert int(bz.CursorMode.NORMAL) == bz.CURSOR_NORMAL
    assert int(bz.CursorMode.HIDDEN) == bz.CURSOR_HIDDEN
    assert int(bz.CursorMode.DISABLED) == bz.CURSOR_DISABLED


def test_an_enum_key_queries_like_its_int(ctx):
    """The queries keep their int signatures and the enum converts through its
    value, so both spellings must answer the same — this is the conversion
    smoke test the 0.23 design named."""
    if ctx.headless:
        pytest.skip("no swapchain support (headless Context)")
    window = a_window()
    try:
        bz.poll_events()
        assert window.is_key_pressed(bz.Key.W) == window.is_key_pressed(bz.KEY_W)
        assert window.was_key_pressed(bz.Key.F11) == window.was_key_pressed(bz.KEY_F11)
        assert (window.is_mouse_button_pressed(bz.MouseButton.LEFT)
                == window.is_mouse_button_pressed(bz.MOUSE_BUTTON_LEFT))
        window.set_cursor_mode(bz.CursorMode.HIDDEN)
        window.set_cursor_mode(bz.CursorMode.NORMAL)
    finally:
        window = None


# ── window extras (0.19) ──────────────────────────────────────────────────
#
# A drop and an icon cannot be provoked from here: no OS input, and no way to ask
# the compositor what icon it is showing. So these pin the half that is ours —
# a drop list is per-cycle state and behaves like every other per-cycle query,
# and set_icon accepts what it says it accepts. The clipboard is the exception and
# gets a real end-to-end test, because a set/get round trip needs no OS input at
# all. `examples/27_drop_and_icon` is the referee for the rest.

def test_dropped_files_is_per_cycle_state(ctx):
    """A drop expires with the poll cycle, so dropped_files() must read the same
    twice inside one frame and start empty — the same contract as the key edges,
    because it uses the same rotation."""
    if ctx.headless:
        pytest.skip("no swapchain support (headless Context)")
    window = a_window()
    try:
        for _ in range(3):
            bz.poll_events()
            first = window.dropped_files()
            second = window.dropped_files()
            assert first == []
            # Reading must not consume: two callers in one frame both see a drop.
            assert first == second
            assert isinstance(first, list)
    finally:
        window = None


def test_set_cursor_position_does_not_fabricate_a_mouse_delta(ctx):
    """Warping the cursor must not read as the user moving it.

    This is the whole reason set_cursor_position touches pos_x_/first_mouse_:
    mouse_callback accumulates `new position - last position`, so a warp would
    otherwise inject a delta the size of the jump — and a camera would swing exactly
    as far as the code moved the cursor to avoid.

    Asserted as a bound rather than an exact zero, and the bound is the point. The
    real cursor belongs to whoever is at the machine: it may be over this window and
    moving, and that movement is a genuine delta the test has no business rejecting.
    The bug injects a delta the size of the WARP, so a jump of several hundred pixels
    that produces a delta of a few dozen is the discriminating signal. An exact zero
    passes on a quiet machine and fails on a busy one, which is a flake rather than a
    test — it failed exactly that way once before this bound existed.
    """
    if ctx.headless:
        pytest.skip("no swapchain support (headless Context)")
    window = a_window(width=640, height=480)
    try:
        bz.poll_events()
        window.get_mouse_state()          # settle: adopt wherever the cursor is

        for target_x, target_y in ((600, 440), (20, 20)):
            bz.poll_events()
            window.set_cursor_position(target_x, target_y)
            bz.poll_events()
            mouse = window.get_mouse_state()
            # The warps above are >= 400 px apart, so the unfixed behaviour reports
            # hundreds. Incidental hand movement between two polls is nothing like it.
            assert abs(mouse.dx) < 100.0, f"warp leaked into dx: {mouse.dx}"
            assert abs(mouse.dy) < 100.0, f"warp leaked into dy: {mouse.dy}"
    finally:
        window = None


def test_set_icon_takes_rgba8_and_none(ctx):
    """The platform may ignore the request (macOS reads the bundle, Wayland the
    desktop file), so what is testable is that bazalt accepts a valid array and
    refuses the three ways an array is wrong — validated in the binding, on the
    main thread, like every other numpy argument."""
    if ctx.headless:
        pytest.skip("no swapchain support (headless Context)")
    np = pytest.importorskip("numpy")
    window = a_window()
    try:
        icon = np.zeros((16, 16, 4), dtype=np.uint8)
        icon[..., 0] = 255
        icon[..., 3] = 255
        window.set_icon(icon)
        # None restores the system default, so "clear it" needs no second verb.
        window.set_icon(None)

        with pytest.raises(bz.WindowError, match="uint8"):
            window.set_icon(np.zeros((16, 16, 4), dtype=np.float32))
        with pytest.raises(bz.WindowError, match="4 channels"):
            window.set_icon(np.zeros((16, 16, 3), dtype=np.uint8))
        with pytest.raises(bz.WindowError, match="dimensions"):
            window.set_icon(np.zeros((16, 16), dtype=np.uint8))
        # memcpy ignores strides, so a view would copy other bytes.
        with pytest.raises(bz.WindowError, match="C-contiguous"):
            window.set_icon(np.zeros((16, 32, 4), dtype=np.uint8)[:, ::2])
    finally:
        window = None


def test_the_clipboard_round_trips(ctx):
    """The one extra that is testable end to end: set then get is a process round
    trip with no OS input in it.

    Free functions rather than Window methods, for the same reason poll_events is
    one: the GLFW calls take no window and the clipboard belongs to the process, so
    a method would invent a per-window distinction that does not exist.
    """
    if ctx.headless:
        pytest.skip("no swapchain support (headless Context)")
    window = a_window()
    try:
        bz.set_clipboard("bazalt clipboard round trip")
        assert bz.get_clipboard() == "bazalt clipboard round trip"
        # Empty is a value, not a failure.
        bz.set_clipboard("")
        assert bz.get_clipboard() == ""
    finally:
        window = None


def test_a_fresh_window_is_open_and_retitles(ctx):
    """The two verbs an application loop starts and ends with. `is_open` is what
    `while window.is_open()` reads, and a title change is the cheapest thing a
    program does to a live window (a frame counter, a file name).

    Trivial, and untested until 0.23 for exactly that reason — the api-coverage
    report is what found them."""
    if ctx.headless:
        pytest.skip("no swapchain support (headless Context)")
    window = a_window()
    try:
        assert window.is_open()
        window.set_title("bazalt retitled")
        bz.poll_events()
        assert window.is_open(), "a title change must not close the window"
    finally:
        window = None


def test_the_clipboard_needs_a_window():
    """GLFW is initialized with the first Window and shut down with the last, so
    the clipboard says so instead of returning nothing. No `ctx` fixture: this must
    run with no window alive, which is the state the guard is about."""
    try:
        bz.get_clipboard()
    except bz.WindowError:
        return
    # A window from another test may still be alive in this process. That is not a
    # failure of the guard, so skip rather than assert on something this test did
    # not control.
    pytest.skip("a window is still alive in this process")


# ── wait_events: the pump that sleeps (0.24) ────────────────────────────


def test_wait_events_with_a_timeout_returns(ctx):
    """The one thing a test can assert without a user at the keyboard: a timeout
    wakes it. Without the timeout this would block until somebody moved a mouse,
    which is the whole point of the call and also untestable."""
    window = a_window()
    start = time.perf_counter()
    bz.wait_events(timeout=0.05)
    elapsed = time.perf_counter() - start

    assert window.is_open()
    # Generous: the OS may deliver an event early and wake it sooner, and a
    # loaded machine may take longer. The assertion is "it returned", not "it
    # slept precisely".
    assert elapsed < 5.0


def test_wait_events_rotates_the_per_cycle_state(ctx):
    """It is a pump, so the edge queries have to expire on it exactly as they do
    on poll_events — otherwise a program that waits sees a key press twice."""
    window = a_window()
    bz.poll_events()
    bz.wait_events(timeout=0.01)
    assert window.was_key_pressed(bz.Key.W) is False


def test_wait_events_rejects_a_negative_timeout(ctx):
    window = a_window()
    with pytest.raises(ValueError, match="negative"):
        bz.wait_events(timeout=-1.0)


def test_wait_events_needs_a_window():
    """Same precondition and the same message shape as poll_events: with no
    window GLFW is not initialized, so the call would vanish without a trace."""
    with pytest.raises(bz.WindowError, match="No windows exist"):
        bz.wait_events(timeout=0.0)
