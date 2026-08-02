"""Volumetric raymarching through a 3D texture (0.23).

The 0.23 headline, end to end:

  * ctx.create_image(depth=N) makes a real VK_IMAGE_TYPE_3D volume — one more
    kwarg on the function that already made 2D images, arrays and cubemaps;
  * a compute shader fills it with fBm noise through an image3D storage
    binding, which works on every driver (no render-to-slice needed);
  * cmd.generate_mipmaps() builds the volume's mip chain — depth halves along
    with width and height, so a 128^3 field gets a full LOD pyramid;
  * a fullscreen fragment shader raymarches it as a sampler3D: one filtered
    lookup anywhere in the field is the thing a stack of 2D slices cannot do.

Also the 0.23 ergonomics in their natural habitat: the descriptor pool takes
no sizes (it grows blocks from the layouts it serves), and Key.ESCAPE is an
enum, not a bare int.
"""

import struct
import time

import bazalt as bz

W, H = 800, 600
VOLUME_SIZE = 128

logger = bz.Logger()


@logger.on_message
def on_message(msg):
    print(f"[{msg.severity}] {msg.text}")


window = bz.Window(W, H, "Bazalt Demo - Volume Raymarch", logger=logger)
ctx = bz.Context(logger)
renderer = ctx.create_renderer(window)

# The volume: 128^3 single-channel density with a full mip chain. The levels
# start empty — compute writes mip 0, generate_mipmaps fills the rest.
volume = ctx.create_image(VOLUME_SIZE, VOLUME_SIZE, bz.Format.R8,
                          depth=VOLUME_SIZE, mip_levels=8, name="density")

fill = (ctx.compute_pipeline()
        .shader(ctx.compile_shader("fill_noise.comp", bz.ShaderStage.COMPUTE))
        .storage_image(0)
        .build())

march = (ctx.graphics_pipeline()
         .vertex_shader(ctx.compile_shader("fullscreen.vert", bz.ShaderStage.VERTEX))
         .fragment_shader(ctx.compile_shader("raymarch.frag", bz.ShaderStage.FRAGMENT))
         .texture(0, bz.ShaderStage.FRAGMENT)
         .push_constant(4, bz.ShaderStage.FRAGMENT)
         .build(renderer))

# No sizes: the pool grows blocks from the layouts it serves (0.23).
pool = ctx.create_descriptor_pool()
fill_set = pool.allocate_set(fill)
fill_set.set_storage_image(0, volume)
march_set = pool.allocate_set(march)
march_set.set_image(0, volume)

# Fill once at startup: dispatch over the whole volume, then build the mip
# chain. The GENERAL -> TRANSFER -> SHADER_READ_ONLY transitions are recorded
# for you; the timer says what a 128^3 noise field costs.
groups = VOLUME_SIZE // 4
setup = ctx.create_command_buffer()
setup.begin()
with setup.timer() as t:
    (setup.bind_pipeline(fill)
          .bind_descriptor_set(fill_set, fill)
          .dispatch(groups, groups, groups))
    setup.generate_mipmaps(volume, src=bz.Access.SHADER_WRITE)
ctx.submit(setup)
# Since 0.24 a device that can never measure says so with UnsupportedError.
# The submit above was blocking, so t.ms is never None here.
try:
    print(f"fill + mips ({VOLUME_SIZE}^3): {t.ms:.2f} ms")
except bz.UnsupportedError:
    print("fill + mips: timestamps unsupported on this device")

cmd = ctx.create_command_buffer()
start = time.time()
last_time = start
frame_count = 0
fps_timer = 0.0
while window.is_open():
    bz.poll_events()
    if window.was_key_pressed(bz.Key.ESCAPE):
        break
    ctx.begin_frame()
    if not renderer.acquire():
        continue
    now = time.time()
    frame_count += 1
    fps_timer += now - last_time
    last_time = now
    if fps_timer >= 1.0:
        window.set_title(f"Bazalt Demo - Volume Raymarch | {frame_count / fps_timer:.1f} FPS")
        frame_count = 0
        fps_timer = 0.0

    cmd.begin()
    with cmd.rendering(renderer) as c:
        c.bind_pipeline(march).bind_descriptor_set(march_set, march)
        c.push_constants(march, 0, struct.pack("<f", now - start))
        c.draw(3)
    renderer.present(cmd)
