"""Bindless — 48 textured quads, eight textures, ONE draw call.

Without descriptor arrays, a quad that uses a different texture is a different
descriptor set, so 48 quads are 48 binds and 48 draws. Here the eight textures
live in ONE binding declared with `texture(0, ..., count=8)`, each written with
`set_image(0, image, index=i)`, and every quad carries its own index as a
per-instance attribute. The whole grid is `draw_indexed(6, instances=48)`.

The index differs between instances of that one draw, so the shader reads the
array through `nonuniformEXT(...)`. That is the part descriptor indexing adds
over a plain array: without it the index has to be the same for every
invocation.

SPACE replaces one texture in place while frames are in flight, which is what
update-after-bind makes legal. Every quad using that slot changes at once.
"""

import struct
import time

import numpy as np

import bazalt as bz

TEXTURE_COUNT = 8
COLUMNS, ROWS = 8, 6

logger = bz.Logger()
logger.on_message(lambda msg: print(f"[{msg.severity}] {msg.text}"))

window = bz.Window(1024, 720, "Bazalt Demo - Bindless")
ctx = bz.Context(logger, optional=[bz.Feature.BINDLESS], gpu_timing=True)
if not ctx.supports(bz.Feature.BINDLESS):
    raise SystemExit("this GPU reports no descriptorIndexing, so there is nothing to show")

renderer = bz.SwapchainRenderer(window, ctx)

vert = ctx.compile_shader("quad.vert", bz.ShaderStage.VERTEX)
frag = ctx.compile_shader("quad.frag", bz.ShaderStage.FRAGMENT)

pipeline = (ctx.graphics_pipeline()
            .vertex_shader(vert)
            .fragment_shader(frag)
            .vertex_format([bz.VertexFormat.FLOAT2, bz.VertexFormat.FLOAT2])
            .instance_format([bz.VertexFormat.FLOAT2,   # offset
                              bz.VertexFormat.FLOAT,    # scale
                              bz.VertexFormat.UINT])    # which texture
            .texture(0, bz.ShaderStage.FRAGMENT, set=0, count=TEXTURE_COUNT)
            .build(renderer))

# ── the mesh: one quad ────────────────────────────────────────────────────
# Vertex order matters: the pipeline default is cull BACK with a
# COUNTER_CLOCKWISE front face, and Vulkan's NDC y points down, so a quad wound
# the "obvious" way (right along the top first) is back-facing and vanishes.
# Corners go bottom-left, top-left, top-right, bottom-right. uv (0,0) is the
# top-left of the quad on screen.
vertices = np.array([
    -1.0, -1.0,  0.0, 0.0,
    -1.0,  1.0,  0.0, 1.0,
     1.0,  1.0,  1.0, 1.0,
     1.0, -1.0,  1.0, 0.0,
], dtype=np.float32)
vbuf = ctx.create_buffer(vertices, bz.BufferType.VERTEX, bz.MemoryUsage.STATIC)
ibuf = ctx.create_buffer(np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32),
                         bz.BufferType.INDEX, bz.MemoryUsage.STATIC)

# ── the instances: a grid, each cell naming a texture ─────────────────────
# 16 bytes each: two floats of offset, one of scale, one uint of texture index.
rows = []
for i in range(COLUMNS * ROWS):
    x = (i % COLUMNS + 0.5) / COLUMNS * 2.0 - 1.0
    y = (i // COLUMNS + 0.5) / ROWS * 2.0 - 1.0
    rows.append(struct.pack("<3fI", x, y, 0.9 / COLUMNS, i % TEXTURE_COUNT))
instances = ctx.create_buffer(np.frombuffer(b"".join(rows), dtype=np.uint8),
                              bz.BufferType.VERTEX, bz.MemoryUsage.STATIC,
                              name="instances")


def make_texture(seed):
    """A 64x64 pattern. Procedural so the example needs no asset files."""
    rng = np.random.default_rng(seed)
    hue = rng.integers(40, 256, size=3)
    yy, xx = np.mgrid[0:64, 0:64]
    pattern = ((xx // 8 + yy // 8) % 2) * 0.55 + 0.45
    pixels = np.zeros((64, 64, 4), dtype=np.uint8)
    pixels[:, :, :3] = (pattern[:, :, None] * hue).astype(np.uint8)
    pixels[:, :, 3] = 255
    return ctx.create_image(pixels, name=f"material {seed}")


# One binding of eight, so the pool needs eight sampler descriptors for one set.
pool = ctx.create_descriptor_pool(max_sets=1, samplers=TEXTURE_COUNT)
desc_set = pool.allocate_set(pipeline, set=0)
textures = [make_texture(i) for i in range(TEXTURE_COUNT)]
for i, image in enumerate(textures):
    desc_set.set_image(0, image, index=i)

cmd = ctx.create_command_buffer()
cmd.begin()
with cmd.rendering(renderer, clear_color=[0.02, 0.02, 0.05, 1.0]) as c:
    (c.bind_pipeline(pipeline)
      .bind_descriptor_set(desc_set, pipeline, set=0)
      .bind_vertex_buffer(vbuf)
      .bind_vertex_buffer(instances, binding=1)
      .bind_index_buffer(ibuf)
      .draw_indexed(6, instances=COLUMNS * ROWS))

print(f"{COLUMNS * ROWS} quads, {TEXTURE_COUNT} textures, 1 draw call. "
      f"SPACE swaps one texture while the loop runs.")

swaps = 0
frames = 0
fps_timer = time.time()

while window.is_open():
    bz.poll_events()

    if window.was_key_pressed(bz.KEY_SPACE):
        # Replaced in place: the recording is not touched and the draw is not
        # re-issued. Legal while earlier frames are still reading the set only
        # because an array binding carries UPDATE_AFTER_BIND.
        swaps += 1
        slot = swaps % TEXTURE_COUNT
        textures[slot] = make_texture(100 + swaps)
        desc_set.set_image(0, textures[slot], index=slot)
        print(f"slot {slot} replaced while {ctx.frames_in_flight} frames are in flight")

    ctx.begin_frame()
    if renderer.acquire():
        renderer.present(cmd)

        frames += 1
        if time.time() - fps_timer >= 1.0:
            gpu = renderer.gpu_time_ms
            gpu_text = f"{gpu:.2f} ms GPU" if gpu else "GPU time n/a"
            window.set_title(
                f"Bazalt Demo - Bindless | {COLUMNS * ROWS} quads, "
                f"{TEXTURE_COUNT} textures, 1 draw | {frames} FPS | {gpu_text}")
            frames = 0
            fps_timer = time.time()
