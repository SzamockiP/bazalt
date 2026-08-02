import bazalt as bz

# Create window, logger, and renderer
logger = bz.Logger()
@logger.on_message
def on_message(msg):
    print(f"[{msg.severity}] {msg.text}")

window = bz.Window(1024, 720, "Bazalt Demo - Empty", logger=logger)
ctx = bz.Context(logger)
renderer = ctx.create_renderer(window)

# Record command buffer once
cmd = ctx.create_command_buffer()
cmd.begin()
with cmd.rendering(renderer, clear_color=[0.1, 0.1, 0.1, 1.0]):
    pass  # nothing to draw — the pass just clears the swapchain

# Main loop. acquire() answers False while the window is minimized or resizing,
# and the frame is skipped — that is the whole windowed contract.
while window.is_open():
    bz.poll_events()
    ctx.begin_frame()
    if renderer.acquire():
        renderer.present(cmd)
