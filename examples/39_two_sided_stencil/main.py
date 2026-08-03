"""Two-sided stencil — counting front faces against back faces (0.25).

The camera flies in and out of a cube. While it is INSIDE, the screen glows
orange. Nothing computes that on the CPU: the stencil buffer does, by counting.

The trick is the oldest one in the stencil book, and it is what shadow volumes
and stencil CSG are both built on:

  * draw the volume with culling OFF, writing no colour,
  * FRONT faces increment the stencil, BACK faces decrement it,
  * wherever the counter ends up non-zero, the near plane is inside the volume.

From outside, every pixel of the cube shows one front face and one back face, so
the two cancel and the counter is 0. From inside, the front faces are behind the
camera and never drawn, so only the decrements land and the counter is not 0.

**That needs a different stencil state per face, which is what `face=` is for.**
Press F to make both faces increment instead, and the effect breaks in exactly
the way that shows why: the whole silhouette of the cube lights up, inside or
out, because nothing is cancelling any more.

`enable` is deliberately not per face — Vulkan has ONE stencil-test bit and two
op-states, so any call sets the bit and the last one wins. INCREMENT_WRAP rather
than INCREMENT_CLAMP for the same "do what the hardware does" reason: a clamped
decrement from 0 stays 0, and this count has to go negative to mean anything.

    F    two-sided on/off
    ESC  quit
"""

import math
import time

import glm
import numpy as np

import bazalt as bz

logger = bz.Logger()
logger.on_message(lambda msg: print(f"[{msg.severity}] {msg.text}"))

window = bz.Window(900, 600, "Bazalt Demo - Two-sided stencil", logger=logger)
ctx = bz.Context(logger)
# The stencil lives on the depth attachment, so the renderer has to be asked for
# one. Without it the pipeline build says so rather than drawing nothing.
renderer = ctx.create_renderer(window, stencil=True)

cube_vert = ctx.compile_shader("cube.vert", bz.ShaderStage.VERTEX)
cube_frag = ctx.compile_shader("cube.frag", bz.ShaderStage.FRAGMENT)
fullscreen_vert = ctx.compile_shader("fullscreen.vert", bz.ShaderStage.VERTEX)
inside_frag = ctx.compile_shader("inside.frag", bz.ShaderStage.FRAGMENT)

# A unit cube, positions only.
positions = np.array([
    -0.5, -0.5, -0.5,   0.5, -0.5, -0.5,   0.5,  0.5, -0.5,  -0.5,  0.5, -0.5,
    -0.5, -0.5,  0.5,   0.5, -0.5,  0.5,   0.5,  0.5,  0.5,  -0.5,  0.5,  0.5,
], dtype=np.float32)
indices = np.array([
    0, 2, 1, 0, 3, 2,   4, 5, 6, 4, 6, 7,
    0, 1, 5, 0, 5, 4,   3, 7, 6, 3, 6, 2,
    0, 4, 7, 0, 7, 3,   1, 2, 6, 1, 6, 5,
], dtype=np.uint32)
vbuf = ctx.create_buffer(positions, bz.BufferType.VERTEX, bz.MemoryUsage.STATIC)
ibuf = ctx.create_buffer(indices, bz.BufferType.INDEX, bz.MemoryUsage.STATIC)


def cube_pipeline(**stencil):
    builder = (ctx.graphics_pipeline()
               .vertex_shader(cube_vert)
               .fragment_shader(cube_frag)
               .vertex_format([bz.VertexFormat.FLOAT3])
               .push_constant(64, bz.ShaderStage.VERTEX))
    if not stencil:
        # The reference picture: an ordinary solid cube, seen from outside.
        return builder.depth_test(True).cull_mode(bz.CullMode.BACK).build(renderer)
    # The counting pass writes no colour and no depth — only the stencil. Culling
    # off, because it has to see BOTH faces to have anything to cancel.
    return (builder
            .depth_test(False)
            .cull_mode(bz.CullMode.NONE)
            .color_mask(False, False, False, False)
            .stencil_test(True, compare=bz.CompareOp.ALWAYS,
                          pass_op=stencil["front_op"], face=bz.Face.FRONT)
            .stencil_test(True, compare=bz.CompareOp.ALWAYS,
                          pass_op=stencil["back_op"], face=bz.Face.BACK)
            .build(renderer))


reference = cube_pipeline()
# Two-sided: the faces cancel, so only "inside" survives.
counting = cube_pipeline(front_op=bz.StencilOp.INCREMENT_WRAP,
                         back_op=bz.StencilOp.DECREMENT_WRAP)
# One-sided, for the F key: nothing cancels, so the whole silhouette counts.
broken = cube_pipeline(front_op=bz.StencilOp.INCREMENT_WRAP,
                       back_op=bz.StencilOp.INCREMENT_WRAP)

# Painted wherever the count is not zero. It writes no stencil of its own —
# write_mask=0 — because it is asking a question, not answering one.
fill = (ctx.graphics_pipeline()
        .vertex_shader(fullscreen_vert)
        .fragment_shader(inside_frag)
        .depth_test(False)
        .cull_mode(bz.CullMode.NONE)
        .stencil_test(True, compare=bz.CompareOp.NOT_EQUAL, ref=0, write_mask=0)
        .build(renderer))

print(__doc__)

two_sided = True
start = time.perf_counter()

while window.is_open():
    bz.poll_events()
    if window.is_key_pressed(bz.Key.ESCAPE):
        break
    if window.was_key_pressed(bz.Key.F):
        two_sided = not two_sided
        print("two-sided: the faces cancel" if two_sided
              else "one-sided: nothing cancels, so the silhouette lights up")

    elapsed = time.perf_counter() - start
    # In and out: 1.9 is well outside the cube, 0.15 is well inside it.
    distance = 1.0 + 0.9 * math.sin(elapsed * 0.8)
    angle = elapsed * 0.35
    eye = glm.vec3(math.cos(angle) * distance, 0.35 * distance, math.sin(angle) * distance)
    aspect = max(renderer.width, 1) / max(renderer.height, 1)
    proj = glm.perspectiveRH_ZO(glm.radians(60.0), aspect, 0.05, 50.0)
    proj[1][1] *= -1
    view_proj = bytes(glm.transpose(proj * glm.lookAt(eye, glm.vec3(0), glm.vec3(0, 1, 0))))

    with ctx.record() as cmd:
        # Pass 1 clears colour, depth AND stencil, then draws the cube so there
        # is something to look at from outside.
        with cmd.rendering(renderer, clear_color=[0.05, 0.06, 0.09, 1.0], clear_stencil=0):
            cmd.bind_pipeline(reference)
            cmd.push_constants(0, view_proj)
            cmd.bind_vertex_buffer(vbuf)
            cmd.bind_index_buffer(ibuf)
            cmd.draw_indexed(len(indices))

        # Pass 2 preserves all of it and only counts. clear_color=None is what
        # makes a second pass a second pass rather than a second frame.
        with cmd.rendering(renderer, clear_color=None):
            cmd.bind_pipeline(counting if two_sided else broken)
            cmd.push_constants(0, view_proj)
            cmd.bind_vertex_buffer(vbuf)
            cmd.bind_index_buffer(ibuf)
            cmd.draw_indexed(len(indices))

        # Pass 3 asks the counter the question.
        with cmd.rendering(renderer, clear_color=None):
            cmd.bind_pipeline(fill)
            cmd.draw(3)

    ctx.begin_frame()
    if renderer.acquire():
        renderer.present(cmd)

    window.set_title(
        f"Bazalt Demo - Two-sided stencil | {'two-sided' if two_sided else 'one-sided (broken)'} "
        f"| distance {distance:.2f}")

cmd = None
renderer = None
window = None
