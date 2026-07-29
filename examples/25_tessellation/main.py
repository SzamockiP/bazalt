"""Tessellation — displacement and adaptive LOD on a flat grid.

The vertex buffer here is 2 floats per vertex and completely flat: a grid of quad
patch corners on the xz plane. Every hill you see, and every triangle that makes
one, is generated on the GPU after the vertex stage.

Three things this shows that no other stage can do:

  1. **Displacement.** `terrain.tese` runs per generated vertex and moves it, so
     the surface detail is a function, not data. The buffer never changes.
  2. **Adaptive LOD.** `terrain.tesc` picks each patch's subdivision level from
     its distance to the camera, so near ground is dense and far ground is
     cheap — decided per frame, on the GPU. Fly in and out and watch the
     wireframe change.
  3. **Crack-free seams.** The level for an edge comes from that edge's midpoint,
     which two neighbouring patches share. Using the patch centre instead is the
     classic way to end up with a terrain full of gaps.

Keys: W toggles wireframe, UP/DOWN change the LOD multiplier, LEFT/RIGHT change
the height scale, SPACE pauses the animation.
"""

import struct
import time

import glm
import numpy as np

import bazalt as bz

logger = bz.Logger()
logger.on_message(lambda msg: print(f"[{msg.severity}] {msg.text}"))

window = bz.Window(1024, 720, "Bazalt Demo - Tessellated terrain")
# TESSELLATION is an optional device feature, like every other capability bazalt
# negotiates. WIREFRAME is optional too, and only the toggle depends on it.
ctx = bz.Context(logger, features=[bz.Feature.TESSELLATION],
                 optional=[bz.Feature.WIREFRAME])
renderer = bz.SwapchainRenderer(window, ctx)

vert = ctx.compile_shader("terrain.vert", bz.ShaderStage.VERTEX)
tesc = ctx.compile_shader("terrain.tesc", bz.ShaderStage.TESS_CONTROL)
tese = ctx.compile_shader("terrain.tese", bz.ShaderStage.TESS_EVALUATION)
frag = ctx.compile_shader("terrain.frag", bz.ShaderStage.FRAGMENT)


def pipeline(polygon):
    return (ctx.graphics_pipeline()
            .vertex_shader(vert)
            .tess_control_shader(tesc)
            .tess_evaluation_shader(tese)
            .fragment_shader(frag)
            .vertex_format([bz.VertexFormat.FLOAT2])
            # A patch is 4 corners, and PATCH_LIST is the only topology a
            # tessellation pipeline can draw. Both are checked at build time.
            .topology(bz.Topology.PATCH_LIST)
            .patch_control_points(4)
            .uniform_buffer(0, bz.ShaderStage.TESS_EVALUATION, set=0)
            .push_constant(32, bz.ShaderStage.TESS_CONTROL)
            .push_constant(32, bz.ShaderStage.TESS_EVALUATION)
            .push_constant(32, bz.ShaderStage.FRAGMENT)
            .depth_test(True)
            .polygon_mode(polygon)
            .cull_mode(bz.CullMode.NONE, bz.FrontFace.COUNTER_CLOCKWISE)
            .build(renderer))


solid = pipeline(bz.PolygonMode.FILL)
wireframe = pipeline(bz.PolygonMode.LINE) if ctx.supports(bz.Feature.WIREFRAME) else None
if wireframe is None:
    print("[info] this GPU has no fillModeNonSolid, so the wireframe toggle is off")

# ── the flat grid ─────────────────────────────────────────────────────────
# 4 corners per quad patch, in the winding terrain.tese's mix() expects:
# 0 = (x, z), 1 = (x+s, z), 2 = (x+s, z+s), 3 = (x, z+s).
GRID = 24
EXTENT = 24.0
step = EXTENT / GRID

corners = []
for iz in range(GRID):
    for ix in range(GRID):
        x = -EXTENT * 0.5 + ix * step
        z = -EXTENT * 0.5 + iz * step
        corners += [
            x, z,
            x + step, z,
            x + step, z + step,
            x, z + step,
        ]

vertices = np.array(corners, dtype=np.float32)
vbuf = ctx.create_buffer(vertices, bz.BufferType.VERTEX, bz.MemoryUsage.STATIC)
patch_vertices = GRID * GRID * 4

ubuf = ctx.create_buffer(16 * 4, bz.BufferType.UNIFORM, bz.MemoryUsage.DYNAMIC)
pool = ctx.create_descriptor_pool(max_sets=2, uniform_buffers=2)
desc_set = pool.allocate_frame_set(solid, set=0)
desc_set.set_buffer(0, ubuf)

show_wireframe = False
lod = 1.0
height_scale = 1.2
paused = False

proj = glm.perspectiveRH_ZO(glm.radians(55.0), 1024.0 / 720.0, 0.1, 200.0)
proj[1][1] *= -1
start = time.time()
animation = 0.0
last = start
frames = 0
fps_timer = start

while window.is_open():
    bz.poll_events()

    if window.was_key_pressed(bz.KEY_W) and wireframe is not None:
        show_wireframe = not show_wireframe
    if window.was_key_pressed(bz.KEY_UP):
        lod = min(lod + 0.25, 4.0)
    if window.was_key_pressed(bz.KEY_DOWN):
        lod = max(lod - 0.25, 0.25)
    if window.was_key_pressed(bz.KEY_RIGHT):
        height_scale = min(height_scale + 0.2, 4.0)
    if window.was_key_pressed(bz.KEY_LEFT):
        height_scale = max(height_scale - 0.2, 0.0)
    if window.was_key_pressed(bz.KEY_SPACE):
        paused = not paused

    now = time.time()
    if not paused:
        animation += now - last
    last = now

    ctx.begin_frame()
    if not renderer.acquire():
        continue

    # The camera flies in and out, which is what makes the adaptive LOD visible:
    # the same patch gets a different level as the distance changes.
    t = now - start
    radius = 14.0 + 9.0 * (0.5 + 0.5 * glm.sin(t * 0.18))
    eye = glm.vec3(glm.cos(t * 0.12) * radius, 5.0 + 3.0 * glm.sin(t * 0.09), glm.sin(t * 0.12) * radius)
    view = glm.lookAt(eye, glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
    ubuf.update(bytes(glm.transpose(proj * view)))

    active = wireframe if show_wireframe else solid
    push = struct.pack("4f4f", eye.x, eye.y, eye.z, lod, height_scale, animation, 0.0, 0.0)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(renderer, clear_color=[0.05, 0.07, 0.10, 1.0]) as c:
        c.bind_pipeline(active)
        c.bind_descriptor_set(desc_set, active, set=0)
        c.push_constants(active, 0, push)
        c.bind_vertex_buffer(vbuf)
        # One draw for the whole terrain. The triangle count it turns into is the
        # tessellator's decision, not this line's.
        c.draw(patch_vertices)
    renderer.present(cmd)

    frames += 1
    if time.time() - fps_timer >= 1.0:
        window.set_title(
            f"Bazalt Demo - Tessellated terrain | "
            f"{'wireframe' if show_wireframe else 'solid'} | "
            f"LOD {lod:.2f} | height {height_scale:.1f} | {frames} FPS")
        frames = 0
        fps_timer = time.time()
