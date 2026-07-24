"""Cascade shadow maps — render-to-layer for depth.

The shadow map is a depth-only RenderTarget with layers=3: a texture ARRAY, one
layer per cascade. Each frame, three depth-only passes render the scene from the
light into shadow.layer(0..2), each cascade an orthographic box of increasing
size around the scene centre (concentric CSM). The scene pass then samples
shadow.depth as a sampler2DArrayShadow, choosing the cascade per fragment.

The new API is exactly:
  * shadow = bz.RenderTarget(ctx, S, S, color=None, depth=D32F, layers=3)
  * cmd.rendering(shadow.layer(c))  → render this cascade's depth

Each cascade tints the surface (red / green / blue near→far) so the split lines
are visible.

NOTE: windowed demo, verified by eye on a real GPU. Concentric-box cascades with
a constant depth bias — no tight frustum fit, no texel snapping, no slope-scaled
bias (a real renderer would add those; they're orthogonal to render-to-layer).
"""

import math
import struct
import time

import bazalt as bz
import glm
import numpy as np

W, H = 1024, 720
SHADOW = 1024
CASCADE_EXTENT = [4.0, 9.0, 18.0]  # half-size of each concentric cascade box
LIGHT_DIR = glm.normalize(glm.vec3(-0.5, -1.0, -0.35))  # surface -> light is -LIGHT_DIR

logger = bz.Logger()
logger.on_message(lambda msg: print(f"[{msg.severity}] {msg.text}"))

window = bz.Window(W, H, "Bazalt Demo - Cascade Shadow Maps (render-to-layer)", logger=logger)
ctx = bz.Context(logger)
renderer = bz.SwapchainRenderer(window, ctx)

# The shadow array: depth-only, one layer per cascade.
shadow = bz.RenderTarget(ctx, SHADOW, SHADOW, color=None, depth=bz.Format.D32F, layers=3)

depth_vert = ctx.compile_shader("depth.vert", bz.ShaderStage.VERTEX)
scene_vert = ctx.compile_shader("scene.vert", bz.ShaderStage.VERTEX)
scene_frag = ctx.compile_shader("scene.frag", bz.ShaderStage.FRAGMENT)

# Depth-only pipeline (no fragment shader — legal on a depth-only target). Built
# against one cascade layer; all three share formats, so one pipeline serves all.
depth_pipe = (ctx.graphics_pipeline()
              .vertex_shader(depth_vert)
              .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
              .depth_test(True)
              .push_constant(64, bz.ShaderStage.VERTEX)
              .build(shadow.layer(0)))

scene_pipe = (ctx.graphics_pipeline()
              .vertex_shader(scene_vert)
              .fragment_shader(scene_frag)
              .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
              .depth_test(True)
              .push_constant(64, bz.ShaderStage.VERTEX)
              .uniform_buffer(0, bz.ShaderStage.FRAGMENT, set=0)
              .texture(1, bz.ShaderStage.FRAGMENT, set=0)
              .build(renderer))

# Concentric ortho cascades around the origin: each is a light-space box of
# increasing size, the same light view backed off along its direction.
def cascade_vp(half):
    eye = glm.vec3(0.0) - LIGHT_DIR * 25.0  # back the light off along its direction
    view = glm.lookAt(eye, glm.vec3(0.0), glm.vec3(0.0, 1.0, 0.0))
    proj = glm.orthoRH_ZO(-half, half, -half, half, 0.1, 60.0)
    proj[1][1] *= -1  # Vulkan Y-flip
    return proj * view


LIGHT_VP = [cascade_vp(e) for e in CASCADE_EXTENT]

# The cascade UBO: 3 light matrices + extents + light dir. The light is static, so
# this is a STATIC uniform buffer + a static descriptor set — no per-frame ring to
# keep fed. (A DYNAMIC buffer written once would leave the OTHER frame-in-flight's
# slot stale and the image would flicker every other frame.) LINEAR + compare =
# hardware PCF, as in the shadow-map example.
cascade_blob = (b"".join(bytes(glm.transpose(vp)) for vp in LIGHT_VP)
                + struct.pack("4f", *CASCADE_EXTENT, 0.0)
                + struct.pack("4f", LIGHT_DIR.x, LIGHT_DIR.y, LIGHT_DIR.z, 0.0))
cascade_ubo = ctx.create_buffer(np.frombuffer(cascade_blob, np.float32).copy(),
                                bz.BufferType.UNIFORM, bz.MemoryUsage.STATIC)

pool = ctx.create_descriptor_pool(max_sets=2, uniform_buffers=2, samplers=2)
scene_set = pool.allocate_set(scene_pipe, set=0)
scene_set.set_buffer(0, cascade_ubo)
scene_set.set_image(1, shadow.depth, sampler=ctx.create_sampler(
    filter=bz.Filter.LINEAR, compare=bz.CompareOp.LESS))


# ── geometry: a ground plane and a few boxes to cast shadows ─────────────────

def box(cx, cy, cz, s):
    faces = [
        ((0, 0, 1), [(-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s)]),
        ((0, 0, -1), [(s, -s, -s), (-s, -s, -s), (-s, s, -s), (s, s, -s)]),
        ((-1, 0, 0), [(-s, -s, -s), (-s, -s, s), (-s, s, s), (-s, s, -s)]),
        ((1, 0, 0), [(s, -s, s), (s, -s, -s), (s, s, -s), (s, s, s)]),
        ((0, 1, 0), [(-s, s, s), (s, s, s), (s, s, -s), (-s, s, -s)]),
        ((0, -1, 0), [(-s, -s, -s), (s, -s, -s), (s, -s, s), (-s, -s, s)]),
    ]
    verts, idx = [], []
    for normal, corners in faces:
        base = len(verts) // 6
        for (x, y, z) in corners:
            verts += [x + cx, y + cy, z + cz, *normal]
        idx += [base, base + 1, base + 2, base + 2, base + 3, base]
    return verts, idx


verts, idx = [], []
# Ground.
gb = len(verts) // 6
verts += [-18, 0, -18, 0, 1, 0,  -18, 0, 18, 0, 1, 0,  18, 0, 18, 0, 1, 0,  18, 0, -18, 0, 1, 0]
idx += [gb, gb + 1, gb + 2, gb + 2, gb + 3, gb]
# A scatter of boxes across the cascades' range.
for (bx, bz_, s) in [(0, 1, 1.0), (5, 1.5, 1.5), (-4, 1, 1.0), (10, 2, 2.0), (-11, 1.5, 1.5), (2, 1, 0.8)]:
    bv, bi = box(bx, s, bz_, s)
    base = len(verts) // 6
    verts += bv
    idx += [base + i for i in bi]

vbuf = ctx.create_buffer(np.array(verts, np.float32), bz.BufferType.VERTEX, bz.MemoryUsage.STATIC)
ibuf = ctx.create_buffer(np.array(idx, np.uint32), bz.BufferType.INDEX, bz.MemoryUsage.STATIC)
index_count = len(idx)


def record(cmd, camera_vp):
    cmd.begin()
    # Three depth-only cascade passes, each into its own shadow layer.
    for c in range(3):
        with cmd.rendering(shadow.layer(c)) as sc:
            (sc.bind_pipeline(depth_pipe)
               .push_constants(depth_pipe, 0, bytes(glm.transpose(LIGHT_VP[c])))
               .bind_vertex_buffer(vbuf)
               .bind_index_buffer(ibuf)
               .draw_indexed(index_count))
    # Scene pass: sample the cascade array.
    with cmd.rendering(renderer, clear_color=[0.05, 0.07, 0.1, 1.0]) as sc:
        (sc.bind_pipeline(scene_pipe)
           .bind_descriptor_set(scene_set, scene_pipe, set=0)
           .push_constants(scene_pipe, 0, bytes(glm.transpose(camera_vp)))
           .bind_vertex_buffer(vbuf)
           .bind_index_buffer(ibuf)
           .draw_indexed(index_count))


cmd = ctx.create_command_buffer()
proj = glm.perspectiveRH_ZO(glm.radians(50.0), W / H, 0.1, 100.0)
proj[1][1] *= -1

TITLE = "Bazalt Demo - Cascade Shadow Maps (render-to-layer)"
start = time.time()
last_time = start
frame_count = 0
fps_timer = 0.0
while window.is_open():
    window.poll_events()
    frame = renderer.begin_frame()
    if frame is None:
        continue

    now = time.time()
    frame_count += 1
    fps_timer += now - last_time
    last_time = now
    if fps_timer >= 1.0:
        fps = frame_count / fps_timer
        window.set_title(f"{TITLE} | {1000.0 / fps:.2f} ms/frame | {fps:.1f} FPS")
        frame_count = 0
        fps_timer = 0.0

    t = (now - start) * 0.25
    eye = glm.vec3(16.0 * math.cos(t), 9.0, 16.0 * math.sin(t))
    view = glm.lookAt(eye, glm.vec3(0.0, 1.0, 0.0), glm.vec3(0.0, 1.0, 0.0))
    record(cmd, proj * view)
    frame.submit(cmd)
