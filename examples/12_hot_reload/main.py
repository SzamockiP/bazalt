"""Hot reload (0.8): edit shaders or the texture while this runs.

Start the demo, then WHILE IT IS RUNNING try any of these and watch the window
update on save — no restart:

  * edit ``shader.frag``        — change the fragment math
  * edit ``palette.glsl``       — the #included tint (warm vs. cool)
  * replace ``../assets/wall.png`` with another image of the SAME size

Make a deliberate typo in a shader: the app keeps rendering the last good
version and prints a ShaderError to the console instead of crashing. Swap in a
differently sized image and it warns and keeps the old one. That resilience is
the whole point — hot_reload=True is the only new line versus example 03.
"""

import time

import bazalt as bz

logger = bz.Logger()


@logger.on_message
def on_message(msg):
    print(f"[{msg.severity}] {msg.source}: {msg.text}")


window = bz.Window(800, 600, "Bazalt - Hot Reload (edit the shaders while running)")
ctx = bz.Context(logger, hot_reload=True)   # <-- the only new line
renderer = bz.SwapchainRenderer(window, ctx)

vert = ctx.compile_shader("shader.vert", bz.ShaderStage.VERTEX)
frag = ctx.compile_shader("shader.frag", bz.ShaderStage.FRAGMENT)
texture = ctx.load_image("../assets/wall.png", name="wall")

pipeline = (ctx.graphics_pipeline()
            .vertex_shader(vert)
            .fragment_shader(frag)
            .texture(0, bz.ShaderStage.FRAGMENT, set=0)
            .name("hot_reload_pipeline")
            .build(renderer))

pool = ctx.create_descriptor_pool(max_sets=1, samplers=1)
desc_set = pool.allocate_set(pipeline, set=0)
# The descriptor set keeps pointing at the same image/view across reloads —
# an image reload re-uploads in place, so this never needs rewriting.
desc_set.set_image(0, texture)

frames = 0
last_time = time.time()
frame_count = 0
fps_timer = 0.0
while window.is_open():
    bz.poll_events()
    ctx.begin_frame()
    if renderer.acquire():
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        frame_count += 1
        fps_timer += dt
        if fps_timer >= 1.0:
            avg_fps = frame_count / fps_timer
            window.set_title(f"Bazalt - Hot Reload (edit while running) | {1000.0 / avg_fps:.2f} ms/frame | {avg_fps:.1f} FPS")
            frame_count = 0
            fps_timer = 0.0

        cmd = ctx.create_command_buffer()
        cmd.begin()
        with cmd.rendering(renderer, clear_color=[0.02, 0.02, 0.03, 1.0]) as c:
            (c.bind_pipeline(pipeline)
              .bind_descriptor_set(desc_set, pipeline, set=0)
              .draw(3))
        renderer.present(cmd)

        frames += 1
        if frames % 120 == 0 and renderer.gpu_time_ms is not None:
            print(f"GPU frame time: {renderer.gpu_time_ms:.3f} ms")
