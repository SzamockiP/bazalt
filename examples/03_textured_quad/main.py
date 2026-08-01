import time

import bazalt as bz

# Create window, logger, and renderer
logger = bz.Logger()
@logger.on_message
def on_message(msg):
    print(msg)

window = bz.Window(800, 600, "Bazalt Demo - Textured Quad")
ctx = bz.Context(logger)
renderer = ctx.create_renderer(window)

# Compile shaders
vert_spv = ctx.compile_shader("quad_tex.vert", bz.ShaderStage.VERTEX)
frag_spv = ctx.compile_shader("quad_tex.frag", bz.ShaderStage.FRAGMENT)

# Load texture
texture = ctx.load_image("../assets/wall.png")

# Build pipeline: Position (FLOAT2) + UV (FLOAT2)
pipeline = (ctx.graphics_pipeline()
    .vertex_shader(vert_spv)
    .fragment_shader(frag_spv)
    .vertex_format([bz.VertexFormat.FLOAT2, bz.VertexFormat.FLOAT2])
    .texture(0, bz.ShaderStage.FRAGMENT, set=0)
    .build(renderer))

# Geometry with interleaved Position (x,y) and UV (u,v)
vertices = [
    -0.5, -0.5,  0.0, 0.0,
     0.5, -0.5,  1.0, 0.0,
     0.5,  0.5,  1.0, 1.0,
    -0.5,  0.5,  0.0, 1.0,
]
vbuf = ctx.create_buffer(vertices, bz.BufferType.VERTEX, bz.MemoryUsage.STATIC, bz.DataType.FLOAT)

# Counter-clockwise on screen (Vulkan's y points down), so the quad is
# front-facing under the pipeline's default back-face culling.
indices = [0, 3, 2, 2, 1, 0]
ibuf = ctx.create_buffer(indices, bz.BufferType.INDEX, bz.MemoryUsage.STATIC, bz.DataType.UINT32)

# Descriptors
pool = ctx.create_descriptor_pool(max_sets=1, textures=1)
desc_set = pool.allocate_set(pipeline, set=0)
desc_set.set_image(0, texture)

# Record commands
cmd = ctx.create_command_buffer()
cmd.begin()
with cmd.rendering(renderer, clear_color=[0.1, 0.2, 0.3, 1.0]) as c:
    (c.bind_pipeline(pipeline)
      .bind_descriptor_set(desc_set, pipeline, set=0)
      .bind_vertex_buffer(vbuf)
      .bind_index_buffer(ibuf)
      .draw_indexed(6))

# Main loop
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
            window.set_title(f"Bazalt Demo - Textured Quad | {1000.0 / avg_fps:.2f} ms/frame | {avg_fps:.1f} FPS")
            frame_count = 0
            fps_timer = 0.0

        renderer.present(cmd)
