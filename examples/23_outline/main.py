"""Stencil outlines — "draw only where something else was not".

Two passes over one mesh:

  1. the object itself, writing 1 into the stencil wherever it covers a pixel
     (`stencil_test(True, compare=ALWAYS, ref=1, pass_op=REPLACE)`),
  2. the same mesh inflated along its normals, with `compare=NOT_EQUAL, ref=1`
     and the depth test off — so the enlarged silhouette paints only the ring
     that the object did not already cover.

The window's depth buffer carries the stencil aspect because the renderer was
built with `stencil=True`; without it the pipeline build says so instead of
drawing nothing.

Keys: SPACE cycles which object is selected, UP/DOWN change the outline width.
"""

import struct
import time

import glm
import numpy as np

import bazalt as bz

logger = bz.Logger()
logger.on_message(lambda msg: print(f"[{msg.severity}] {msg.text}"))

window = bz.Window(1024, 720, "Bazalt Demo - Stencil outline", logger=logger)
ctx = bz.Context(logger)
# The stencil lives on the depth attachment, so it is a renderer option.
renderer = ctx.create_renderer(window, stencil=True)

vert = ctx.compile_shader("object.vert", bz.ShaderStage.VERTEX)
frag = ctx.compile_shader("object.frag", bz.ShaderStage.FRAGMENT)


def pipeline(**stencil):
    builder = (ctx.graphics_pipeline()
        .vertex_shader(vert)
        .fragment_shader(frag)
        .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
        .push_constant(20, bz.ShaderStage.VERTEX)
        .push_constant(20, bz.ShaderStage.FRAGMENT)
        .uniform_buffer(0, bz.ShaderStage.VERTEX)
        .cull_mode(bz.CullMode.BACK, bz.FrontFace.COUNTER_CLOCKWISE))
    depth = stencil.pop("depth", True)
    return builder.depth_test(depth).stencil_test(True, **stencil).build(renderer)


# Pass 1: paint the object and stamp its silhouette into the stencil.
mark = pipeline(compare=bz.CompareOp.ALWAYS, ref=1, pass_op=bz.StencilOp.REPLACE)
# Pass 2: the inflated copy, kept only where the stamp is absent. The depth test
# is off so the ring is not hidden by the object it surrounds.
outline = pipeline(compare=bz.CompareOp.NOT_EQUAL, ref=1, depth=False)
# The unselected objects write no stencil at all, so nothing outlines them.
plain = pipeline(compare=bz.CompareOp.ALWAYS, ref=0, write_mask=0)

# ── one cube ──────────────────────────────────────────────────────────────
vertices = np.array([
    -0.5, -0.5,  0.5,   0.0, 0.0, 1.0,
     0.5, -0.5,  0.5,   0.0, 0.0, 1.0,
     0.5,  0.5,  0.5,   0.0, 0.0, 1.0,
    -0.5,  0.5,  0.5,   0.0, 0.0, 1.0,
    -0.5, -0.5, -0.5,   0.0, 0.0, -1.0,
     0.5, -0.5, -0.5,   0.0, 0.0, -1.0,
     0.5,  0.5, -0.5,   0.0, 0.0, -1.0,
    -0.5,  0.5, -0.5,   0.0, 0.0, -1.0,
    -0.5, -0.5, -0.5,  -1.0, 0.0, 0.0,
    -0.5, -0.5,  0.5,  -1.0, 0.0, 0.0,
    -0.5,  0.5,  0.5,  -1.0, 0.0, 0.0,
    -0.5,  0.5, -0.5,  -1.0, 0.0, 0.0,
     0.5, -0.5, -0.5,   1.0, 0.0, 0.0,
     0.5, -0.5,  0.5,   1.0, 0.0, 0.0,
     0.5,  0.5,  0.5,   1.0, 0.0, 0.0,
     0.5,  0.5, -0.5,   1.0, 0.0, 0.0,
    -0.5, -0.5, -0.5,   0.0, -1.0, 0.0,
     0.5, -0.5, -0.5,   0.0, -1.0, 0.0,
     0.5, -0.5,  0.5,   0.0, -1.0, 0.0,
    -0.5, -0.5,  0.5,   0.0, -1.0, 0.0,
    -0.5,  0.5, -0.5,   0.0, 1.0, 0.0,
     0.5,  0.5, -0.5,   0.0, 1.0, 0.0,
     0.5,  0.5,  0.5,   0.0, 1.0, 0.0,
    -0.5,  0.5,  0.5,   0.0, 1.0, 0.0,
], dtype=np.float32)
vbuf = ctx.create_buffer(vertices, bz.BufferType.VERTEX, bz.MemoryUsage.STATIC)

indices = np.array([
    0, 1, 2, 2, 3, 0,
    5, 4, 7, 7, 6, 5,
    8, 9, 10, 10, 11, 8,
    13, 12, 15, 15, 14, 13,
    16, 17, 18, 18, 19, 16,
    23, 22, 21, 21, 20, 23,
], dtype=np.uint32)
ibuf = ctx.create_buffer(indices, bz.BufferType.INDEX, bz.MemoryUsage.STATIC)

# view_proj + model
ubuf = ctx.create_buffer(32 * 4, bz.BufferType.UNIFORM, bz.MemoryUsage.DYNAMIC)
pool = ctx.create_descriptor_pool()
desc_set = pool.allocate_frame_set(mark)
desc_set.set_buffer(0, ubuf)

OBJECTS = [
    (glm.vec3(-1.6, 0.0, 0.0), (0.85, 0.35, 0.25, 1.0)),
    (glm.vec3(0.0, 0.0, 0.0), (0.30, 0.65, 0.90, 1.0)),
    (glm.vec3(1.6, 0.0, 0.0), (0.45, 0.80, 0.40, 1.0)),
]
OUTLINE_COLOR = (1.0, 0.85, 0.1, 1.0)

selected = 1
width = 0.06

proj = glm.perspectiveRH_ZO(glm.radians(45.0), 1024.0 / 720.0, 0.1, 100.0)
proj[1][1] *= -1
start = time.time()
frames = 0
fps_timer = time.time()

while window.is_open():
    bz.poll_events()

    if window.was_key_pressed(bz.Key.SPACE):
        selected = (selected + 1) % len(OBJECTS)
    if window.was_key_pressed(bz.Key.UP):
        width = min(width + 0.02, 0.3)
    if window.was_key_pressed(bz.Key.DOWN):
        width = max(width - 0.02, 0.0)

    ctx.begin_frame()
    if not renderer.acquire():
        continue

    t = time.time() - start
    view = glm.lookAt(glm.vec3(0, 1.6, 5.5), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
    view_proj = proj * view

    # One recording per frame: the model matrix rides in the uniform buffer, so
    # each object needs its own pass over the mesh anyway.
    cmd = ctx.create_command_buffer()
    cmd.begin()
    for index, (position, color) in enumerate(OBJECTS):
        model = glm.translate(glm.mat4(1.0), position)
        model = glm.rotate(model, t * (0.6 + 0.2 * index), glm.vec3(0.3, 1.0, 0.15))
        ubuf.update(bytes(glm.transpose(view_proj)) + bytes(glm.transpose(model)))

        first = index == 0
        # The first pass of the frame clears; the rest preserve, or each object
        # would wipe the ones before it.
        clear = [0.03, 0.03, 0.06, 1.0] if first else None
        with cmd.rendering(renderer, clear_color=clear, clear_stencil=0) as c:
            body = mark if index == selected else plain
            c.bind_pipeline(body)
            c.bind_descriptor_set(desc_set, body)
            c.push_constants(body, 0, struct.pack("4ff", *color, 0.0))
            c.bind_vertex_buffer(vbuf).bind_index_buffer(ibuf).draw_indexed(36)

        if index == selected and width > 0.0:
            with cmd.rendering(renderer, clear_color=None) as c:
                c.bind_pipeline(outline)
                c.bind_descriptor_set(desc_set, outline)
                c.push_constants(outline, 0, struct.pack("4ff", *OUTLINE_COLOR, width))
                c.bind_vertex_buffer(vbuf).bind_index_buffer(ibuf).draw_indexed(36)

    renderer.present(cmd)

    frames += 1
    if time.time() - fps_timer >= 1.0:
        window.set_title(
            f"Bazalt Demo - Stencil outline | object {selected} | "
            f"width {width:.2f} | {frames} FPS")
        frames = 0
        fps_timer = time.time()
