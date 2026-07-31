"""Colour grading through a 3D LUT, built by render-to-slice (0.23).

A grading LUT is a 32^3 texture whose texel at (r, g, b) holds where the grade
sends that colour, so applying ANY colour transform becomes one sampler3D
lookup in the post pass. Two ways to build it, picked at startup:

  * render-to-slice, where the driver allows it: a RenderTarget over the
    volume, then one tiny fullscreen pass per Z slice through target.layer(z)
    — ctx.supports(Feature.IMAGE_VIEW_2D_ON_3D) is the question to ask;
  * an image3D compute fill everywhere else (MoltenVK) — the escape hatch the
    feature's own error message names.

The window shows the animated scene split down the middle: original on the
left, graded on the right. Drag the split with the arrow keys (Key enums,
0.23); ESC quits.
"""

import struct
import time

import bazalt as bz

W, H = 800, 600
LUT_SIZE = 32

logger = bz.Logger()


@logger.on_message
def on_message(msg):
    print(f"[{msg.severity}] {msg.text}")


window = bz.Window(W, H, "Bazalt Demo - LUT Grading", logger=logger)
ctx = bz.Context(logger)
renderer = bz.SwapchainRenderer(window, ctx)

fullscreen = ctx.compile_shader("fullscreen.vert", bz.ShaderStage.VERTEX)
pool = ctx.create_descriptor_pool()  # automatic (0.23)

# ── build the LUT, once ───────────────────────────────────────────────────

lut = ctx.create_image(LUT_SIZE, LUT_SIZE, bz.Format.RGBA8, depth=LUT_SIZE, name="grade lut")

if ctx.supports(bz.Feature.IMAGE_VIEW_2D_ON_3D):
    print("building the LUT by render-to-slice")
    lut_target = ctx.create_render_target(color=[lut])
    slice_pipe = (ctx.graphics_pipeline()
                  .vertex_shader(fullscreen)
                  .fragment_shader(ctx.compile_shader("lut_slice.frag", bz.ShaderStage.FRAGMENT))
                  .push_constant(4, bz.ShaderStage.FRAGMENT)
                  .build(lut_target))
    bake = ctx.create_command_buffer()
    bake.begin()
    for z in range(LUT_SIZE):
        # Texel centres, so both build paths produce the same LUT.
        slice_target = lut_target.layer(z)
        with bake.rendering(slice_target) as c:
            c.bind_pipeline(slice_pipe)
            c.push_constants(slice_pipe, 0, struct.pack("<f", (z + 0.5) / LUT_SIZE))
            c.draw(3)
    ctx.submit(bake)
else:
    print("no IMAGE_VIEW_2D_ON_3D on this driver - building the LUT in compute")
    fill_pipe = (ctx.compute_pipeline()
                 .shader(ctx.compile_shader("lut_fill.comp", bz.ShaderStage.COMPUTE))
                 .storage_image(0)
                 .build())
    fill_set = pool.allocate_set(fill_pipe, set=0)
    fill_set.set_storage_image(0, lut)
    bake = ctx.create_command_buffer()
    bake.begin()
    groups = LUT_SIZE // 4
    bake.bind_pipeline(fill_pipe).bind_descriptor_set(fill_set, fill_pipe, set=0)
    bake.dispatch(groups, groups, groups)
    ctx.submit(bake)

# ── the scene and the post pass ───────────────────────────────────────────

scene_target = ctx.create_render_target(W, H)
scene_pipe = (ctx.graphics_pipeline()
              .vertex_shader(fullscreen)
              .fragment_shader(ctx.compile_shader("scene.frag", bz.ShaderStage.FRAGMENT))
              .push_constant(4, bz.ShaderStage.FRAGMENT)
              .build(scene_target))

apply_pipe = (ctx.graphics_pipeline()
              .vertex_shader(fullscreen)
              .fragment_shader(ctx.compile_shader("apply_lut.frag", bz.ShaderStage.FRAGMENT))
              .texture(0, bz.ShaderStage.FRAGMENT, set=0)
              .texture(1, bz.ShaderStage.FRAGMENT, set=0)
              .push_constant(4, bz.ShaderStage.FRAGMENT)
              .build(renderer))

apply_set = pool.allocate_set(apply_pipe, set=0)
apply_set.set_image(0, scene_target.color[0])
# CLAMP, not the default REPEAT: a colour of 0.999 must not wrap to 0.
apply_set.set_image(1, lut, ctx.create_sampler(address_mode=bz.AddressMode.CLAMP))

cmd = ctx.create_command_buffer()
split = 0.5
start = time.time()
while window.is_open():
    bz.poll_events()
    if window.was_key_pressed(bz.Key.ESCAPE):
        break
    if window.is_key_pressed(bz.Key.LEFT):
        split = max(0.0, split - 0.01)
    if window.is_key_pressed(bz.Key.RIGHT):
        split = min(1.0, split + 0.01)
    ctx.begin_frame()
    if not renderer.acquire():
        continue

    cmd.begin()
    with cmd.rendering(scene_target) as c:
        c.bind_pipeline(scene_pipe)
        c.push_constants(scene_pipe, 0, struct.pack("<f", time.time() - start))
        c.draw(3)
    with cmd.rendering(renderer) as c:
        c.bind_pipeline(apply_pipe).bind_descriptor_set(apply_set, apply_pipe, set=0)
        c.push_constants(apply_pipe, 0, struct.pack("<f", split))
        c.draw(3)
    renderer.present(cmd)
