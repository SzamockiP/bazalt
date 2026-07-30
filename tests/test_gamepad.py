"""Gamepads (0.21).

`bz.get_gamepad(index)` returns a snapshot of one pad, or None when that slot is
empty. A free function, not a Window method, for the reason poll_events() is one:
GLFW's gamepad state is per process and takes a joystick id rather than a window.

No pad can be plugged in from a test, so these pin the half that is ours — the
same position test_window_input.py is in. What is left over is:

  * the surface: get_gamepad answers None or a Gamepad and never raises for an
    empty slot, and both enums cover GLFW's whole layout,
  * the refusals: an index or a deadzone outside its fixed range is a ValueError,
    and reading a pad with no window open is a WindowError.

The deadzone arithmetic is checked where it lives, by static_assert in
window.hpp, because it is the one piece of logic here and nothing in Python can
reach it without hardware. The rest is covered by driving a real pad in
examples/30_gamepad.
"""

import pytest

import bazalt as bz


def a_window():
    try:
        return bz.Window(128, 96, "bazalt gamepad")
    except bz.WindowError:
        pytest.skip("no display available")


def test_an_empty_slot_answers_none_rather_than_raising():
    """Nothing is plugged in on CI, and that is an answer, not a failure. Every
    slot is polled, so an index GLFW rejects cannot be mistaken for a pad."""
    window = a_window()
    try:
        bz.poll_events()
        for index in range(16):
            pad = bz.get_gamepad(index)
            assert pad is None or isinstance(pad, bz.Gamepad)
    finally:
        del window


def test_a_connected_pad_reads_every_control():
    """Skipped without hardware, and here so that a machine WITH a pad checks the
    shape of a real reading: every axis in range, every button a bool, and the
    triggers normalized to 0..1 rather than GLFW's -1..1.

    What it cannot check is DIRECTION — that a stick pushed up reads positive.
    Nothing can push a stick from here, so that one is verified by running
    examples/30_gamepad, which prints both stick values in the title bar."""
    window = a_window()
    try:
        bz.poll_events()
        pad = bz.get_gamepad(0)
        if pad is None:
            pytest.skip("no gamepad connected")
        assert pad.index == 0
        assert pad.name
        for axis in bz.GamepadAxis.__members__.values():
            value = pad.axis(axis)
            assert isinstance(value, float)
            if axis in (bz.GamepadAxis.LEFT_TRIGGER, bz.GamepadAxis.RIGHT_TRIGGER):
                assert 0.0 <= value <= 1.0, f"{axis} is not normalized: {value}"
            else:
                assert -1.0 <= value <= 1.0, f"{axis} out of range: {value}"
        for button in bz.GamepadButton.__members__.values():
            assert isinstance(pad.button(button), bool)
    finally:
        del window


def test_the_enums_cover_the_whole_layout():
    """GLFW maps every pad it knows onto 15 buttons and 6 axes, and bazalt renames
    those rather than translating them — so a missing entry would be a control no
    caller can name."""
    assert len(bz.GamepadButton.__members__) == 15
    assert len(bz.GamepadAxis.__members__) == 6
    # The values are GLFW's own, so the first and last of each pin the mapping.
    assert int(bz.GamepadButton.A) == 0
    assert int(bz.GamepadButton.DPAD_LEFT) == 14
    assert int(bz.GamepadAxis.LEFT_X) == 0
    assert int(bz.GamepadAxis.RIGHT_TRIGGER) == 5


@pytest.mark.parametrize("index", [-1, 16, 999])
def test_an_index_outside_the_slots_is_a_value_error(index):
    """ValueError, not a BazaltError: the number is wrong on its own, and
    `except bz.WindowError` around a windowing fallback must not swallow a typo."""
    window = a_window()
    try:
        with pytest.raises(ValueError):
            bz.get_gamepad(index)
    finally:
        del window


@pytest.mark.parametrize("deadzone", [-0.1, 1.0, 2.0])
def test_a_deadzone_outside_its_range_is_a_value_error(deadzone):
    window = a_window()
    try:
        with pytest.raises(ValueError):
            bz.get_gamepad(0, deadzone=deadzone)
    finally:
        del window


def test_reading_a_pad_with_no_window_is_refused():
    """GLFW is initialized with the first Window and shut down with the last, so
    the pads exist only while one is open. The same precondition poll_events() and
    the clipboard have, and it says so rather than answering None — an empty slot
    and a dead GLFW are different facts."""
    # poll_events() has the same precondition, so it is the way to ask whether
    # some other test in this run still holds a window open.
    try:
        bz.poll_events()
    except bz.WindowError:
        pass
    else:
        pytest.skip("another window is still open, so GLFW is still initialized")

    with pytest.raises(bz.WindowError):
        bz.get_gamepad(0)
