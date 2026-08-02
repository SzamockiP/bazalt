"""Gamepad — the window colour follows the sticks.

`bz.get_gamepad(0)` is a free function, like `bz.poll_events()`: a pad belongs to
the process, not to a window, so there is no such thing as one window's gamepad.
It answers None when nothing is plugged into that slot, which is why this example
still runs with no hardware at all.

The reading is level state — where the sticks are and which buttons are down as
of the last `poll_events()`. There are no edge queries (`was_button_pressed`):
those rotate on a per-window counter, and a pad has no window to hang one on.

Nothing here compiles a shader. The whole picture is the clear colour, which is
the smallest thing that can show a live analogue value.

  * left stick   -> red and green
  * right stick  -> blue
  * triggers     -> overall brightness, 0..1 each (bazalt normalizes them; GLFW
                    reports a trigger as -1 released)
  * any button   -> printed once while it is held
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
held = set()
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

        down = {b for b in bz.GamepadButton.__members__.values() if pad.button(b)}
        for button in sorted(down - held, key=lambda b: b.name):
            print(f"  {button.name} down")
        held = down

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
