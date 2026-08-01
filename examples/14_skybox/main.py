"""Skybox from a procedural cubemap — the 0.10 headline, end to end.

  * an EMPTY cubemap: ctx.create_image(size, size, cube=True) makes a 6-layer
    image with a CUBE view (for sampling) and a 2D_ARRAY view (for storage);
  * a compute shader fills all six faces with imageStore, writing through the
    2D_ARRAY storage view — no vertices, no fragment shader;
  * the cubemap is baked ONCE, up front: a compute dispatch fills the faces,
    then cmd.barrier(cubemap, SHADER_WRITE, SHADER_READ) transitions all six
    layers from GENERAL (storage) to SHADER_READ_ONLY (sampled). After that the
    render loop just samples it every frame — no regeneration;
  * a fullscreen pass turns each pixel into a world-space ray and samples the
    cubemap as a samplerCube.

Move the mouse to look around — the cubemap surrounds you.
"""

import math
import struct
import time

import bazalt as bz

W, H = 900, 600
SKY = 512  # cubemap face resolution

logger = bz.Logger()


@logger.on_message
def on_message(msg):
    print(f"[{msg.severity}] {msg.text}")


window = bz.Window(W, H, "Bazalt Demo - Skybox (procedural cubemap)", logger=logger)
ctx = bz.Context(logger)
renderer = ctx.create_renderer(window)
window.set_cursor_mode(bz.CURSOR_DISABLED)  # mouse-look

# Compute writes the six faces of an empty cubemap.
sky_comp = ctx.compile_shader("sky.comp", bz.ShaderStage.COMPUTE)
generate = ctx.compute_pipeline().shader(sky_comp).storage_image(0).build()
cubemap = ctx.create_image(SKY, SKY, bz.Format.RGBA8, cube=True)

# Skybox pass: fullscreen triangle, per-pixel ray, samplerCube.
vert = ctx.compile_shader("skybox.vert", bz.ShaderStage.VERTEX)
frag = ctx.compile_shader("skybox.frag", bz.ShaderStage.FRAGMENT)
skybox = (ctx.graphics_pipeline()
          .vertex_shader(vert)
          .fragment_shader(frag)
          .texture(0, bz.ShaderStage.FRAGMENT, set=0)
          .push_constant(64, bz.ShaderStage.FRAGMENT)
          .build(renderer))

# One image, bound two ways: storage (written) and sampled (read).
pool = ctx.create_descriptor_pool(max_sets=4, storage_images=4, textures=4)
gen_set = pool.allocate_set(generate, set=0)
gen_set.set_storage_image(0, cubemap)
sky_set = pool.allocate_set(skybox, set=0)
sky_set.set_image(0, cubemap)


def normalize(v):
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return (v[0] / length, v[1] / length, v[2] / length)


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


TAN_HALF_FOV = math.tan(math.radians(60.0) / 2.0)
ASPECT = W / H


def camera_push(yaw, pitch):
    # Look direction straight from the mouse-driven yaw/pitch — same convention
    # as the Camera in examples 04-07.
    forward = normalize((
        math.cos(yaw) * math.cos(pitch),
        math.sin(pitch),
        math.sin(yaw) * math.cos(pitch),
    ))
    right = normalize(cross(forward, (0.0, 1.0, 0.0)))
    up = cross(right, forward)
    return struct.pack(
        "16f",
        right[0], right[1], right[2], 0.0,
        up[0], up[1], up[2], 0.0,
        forward[0], forward[1], forward[2], 0.0,
        TAN_HALF_FOV, ASPECT, 0.0, 0.0)


# Bake the cubemap once: fill every face in compute, then transition all six
# layers from GENERAL (storage) to SHADER_READ_ONLY (sampled) by hand. A
# blocking headless submit, so it's done before the loop starts.
setup = ctx.create_command_buffer()
setup.begin()
(setup.bind_pipeline(generate)
      .bind_descriptor_set(gen_set, generate, set=0)
      .dispatch((SKY + 7) // 8, (SKY + 7) // 8, 6))
setup.barrier(cubemap, bz.Access.SHADER_WRITE, bz.Access.SHADER_READ)
ctx.submit(setup)


def record(cmd, yaw, pitch):
    cmd.begin()
    with cmd.rendering(renderer) as c:
        (c.bind_pipeline(skybox)
          .bind_descriptor_set(sky_set, skybox, set=0)
          .push_constants(skybox, 0, camera_push(yaw, pitch))
          .draw(3))


cmd = ctx.create_command_buffer()
yaw, pitch = 0.0, 0.0
last_time = time.time()
frame_count = 0
fps_timer = 0.0
while window.is_open():
    bz.poll_events()
    ctx.begin_frame()
    if not renderer.acquire():
        continue

    current_time = time.time()
    dt = current_time - last_time
    last_time = current_time
    frame_count += 1
    fps_timer += dt
    if fps_timer >= 1.0:
        avg_fps = frame_count / fps_timer
        window.set_title(f"Bazalt Demo - Skybox (procedural cubemap) | {1000.0 / avg_fps:.2f} ms/frame | {avg_fps:.1f} FPS")
        frame_count = 0
        fps_timer = 0.0

    mouse = window.get_mouse_state()
    yaw += mouse.dx * 0.002
    pitch = max(-math.pi / 2 + 0.01, min(math.pi / 2 - 0.01, pitch + mouse.dy * 0.002))

    record(cmd, yaw, pitch)
    renderer.present(cmd)
