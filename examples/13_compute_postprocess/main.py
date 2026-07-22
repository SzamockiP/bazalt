"""Compute post-processing: generate an image in compute, then sample it.

The 0.9 headline, end to end:

  * a compute shader writes a procedural image into a STORAGE image with
    imageStore (no vertices, no fragment shader) — ctx.create_image gives it
    STORAGE usage automatically, DescriptorSet.set_storage_image binds it;
  * a fullscreen pass samples that image to the screen. The barrier that takes
    the image from GENERAL (storage) to SHADER_READ_ONLY (sampled) is recorded
    for you and hoisted before the render pass — no cmd.barrier() by hand;
  * `with cmd.timer("generate"):` measures the dispatch. Read it back with
    cmd.timer_ms("generate") after a blocking submit — no window needed, which
    is why the measurement below runs before the render loop.

The command buffer is re-recorded each frame only because the animation time is
a push constant; the storage-image barriers are recomputed identically each
time.
"""

import struct
import time

import bazalt as bz

W, H = 512, 512

logger = bz.Logger()


@logger.on_message
def on_message(msg):
    print(f"[{msg.severity}] {msg.text}")


window = bz.Window(W, H, "Bazalt Demo - Compute Post-processing", logger=logger)
ctx = bz.Context(logger)
renderer = bz.SwapchainRenderer(window, ctx)

# Compute: writes the storage image. No stage arguments, no target.
gen_shader = ctx.compile_shader("pattern.comp", bz.ShaderStage.COMPUTE)
generate = (ctx.compute_pipeline()
            .shader(gen_shader)
            .storage_image(0)
            .push_constant(4)
            .build())

# Presentation: a fullscreen triangle sampling the generated image.
vert = ctx.compile_shader("fullscreen.vert", bz.ShaderStage.VERTEX)
frag = ctx.compile_shader("present.frag", bz.ShaderStage.FRAGMENT)
present = (ctx.graphics_pipeline()
           .vertex_shader(vert)
           .fragment_shader(frag)
           .texture(0, bz.ShaderStage.FRAGMENT, set=0)
           .build(renderer))

# One image, bound two ways: as a storage image to the compute set (written),
# and as a sampled texture to the present set (read). The tracker keys on the
# image, so it sees both uses and transitions the layout between them.
image = ctx.create_image(W, H, bz.Format.RGBA8)

pool = ctx.create_descriptor_pool(max_sets=4, storage_images=4, samplers=4)
gen_set = pool.allocate_set(generate, set=0)
gen_set.set_storage_image(0, image)
present_set = pool.allocate_set(present, set=0)
present_set.set_image(0, image)


def record(cmd, t):
    cmd.begin()
    with cmd.timer("generate"):
        (cmd.bind_pipeline(generate)
            .bind_descriptor_set(gen_set, generate, set=0)
            .push_constants(generate, 0, struct.pack("<f", t))
            .dispatch((W + 7) // 8, (H + 7) // 8))
    with cmd.rendering(renderer):
        cmd.bind_pipeline(present).bind_descriptor_set(present_set, present, set=0).draw(3)


# Measure the compute cost with one blocking headless submit — timer_ms reads
# back reliably here (no swapchain, no frames-in-flight race).
measure = ctx.create_command_buffer()
measure.begin()
with measure.timer("generate"):
    (measure.bind_pipeline(generate)
        .bind_descriptor_set(gen_set, generate, set=0)
        .push_constants(generate, 0, struct.pack("<f", 0.0))
        .dispatch((W + 7) // 8, (H + 7) // 8))
ctx.submit(measure)
ms = measure.timer_ms("generate")
print(f"compute generate ({W}x{H}): {ms:.4f} ms" if ms is not None
      else "compute generate: timestamps unsupported on this device")

cmd = ctx.create_command_buffer()
start = time.time()
while window.is_open():
    window.poll_events()
    frame = renderer.begin_frame()
    if frame is None:
        continue
    record(cmd, time.time() - start)
    frame.submit(cmd)
