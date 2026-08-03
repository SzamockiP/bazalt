"""Gamepad — the window colour follows the sticks.

`bz.get_gamepad(0)` is a free function, like `bz.poll_events()`: a pad belongs to
the process, not to a window, so there is no such thing as one window's gamepad.
It answers None when nothing is plugged into that slot, which is why this example
still runs with no hardware at all.

The reading carries both the level and the edge: where the sticks are and which
buttons are down as of the last `poll_events()`, plus which of them went down
since the poll before that (`pad.was_button_pressed`, 0.25). The edge is measured
between two readings rather than recorded as it happens, because GLFW has no
gamepad callback — so read the pad every frame.

Nothing here compiles a shader. The whole picture is the clear colour, which is
the smallest thing that can show a live analogue value.

  * left stick   -> red and green
  * right stick  -> blue
  * triggers     -> overall brightness, 0..1 each (bazalt normalizes them; GLFW
                    reports a trigger as -1 released)
  * any button   -> printed once, on the frame it goes down
"""

import time

import bazalt as bz

DEADZONE = 0.15

logger = bz.Logger()
logger.on_message(lambda msg: print(f"[{msg.severity}] {msg.text}"))

window = bz.Window(800, 600, "Bazalt Demo - Gamepad", logger=logger)
ctx = bz.Context(logger)
renderer = ctx.create_renderer(window)

print("plug in a gamepad and move the sticks. ESC or close the window to quit.")
announced = False

while window.is_open():
    bz.poll_events()
    if window.is_key_pressed(bz.Key.ESCAPE):
        break

    pad = bz.get_gamepad(0, deadzone=DEADZONE)
    if pad is None:
        color = [0.1, 0.1, 0.12, 1.0]
        announced = False
        window.set_title("Bazalt Demo - Gamepad | no pad in slot 0")
    else:
        if not announced:
            print(f"slot {pad.index}: {pad.name}")
            announced = True

        # -1..1 from the sticks, moved into 0..1 for a colour.
        red = (pad.axis(bz.GamepadAxis.LEFT_X) + 1.0) * 0.5
        green = (pad.axis(bz.GamepadAxis.LEFT_Y) + 1.0) * 0.5
        blue = (pad.axis(bz.GamepadAxis.RIGHT_X) + 1.0) * 0.5
        # Already 0..1, so they multiply straight in.
        brightness = 0.25 + 0.75 * max(pad.axis(bz.GamepadAxis.LEFT_TRIGGER),
                                       pad.axis(bz.GamepadAxis.RIGHT_TRIGGER))
        color = [red * brightness, green * brightness, blue * brightness, 1.0]

        # The edge, not the level, so the caller keeps no "was it down last
        # frame" set of its own — which is what these three lines used to be.
        for button in bz.GamepadButton.__members__.values():
            if pad.was_button_pressed(button):
                print(f"  {button.name} down")

        window.set_title(
            f"Bazalt Demo - Gamepad | {pad.name} | "
            f"L({pad.axis(bz.GamepadAxis.LEFT_X):+.2f}, {pad.axis(bz.GamepadAxis.LEFT_Y):+.2f}) "
            f"R({pad.axis(bz.GamepadAxis.RIGHT_X):+.2f}, {pad.axis(bz.GamepadAxis.RIGHT_Y):+.2f})")

    # One recording per frame: the clear colour is baked into it, and the clear
    # colour is the whole picture here.
    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(renderer, clear_color=color):
        pass

    ctx.begin_frame()
    if renderer.acquire():
        renderer.present(cmd)
    time.sleep(0.004)
