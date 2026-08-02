import bazalt as bz

# Create window, logger, and renderer
logger = bz.Logger()
@logger.on_message
def on_message(msg):
    print(msg)

window = bz.Window(800, 600, "Bazalt Demo - Textured Quad", logger=logger)
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
    .texture(0, bz.ShaderStage.FRAGMENT)
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

# Descriptors. No sizes on the pool: it reads the layouts it serves and grows a
# new block whenever one fills. Pass explicit counts only to hand-budget it.
pool = ctx.create_descriptor_pool()
desc_set = pool.allocate_set(pipeline)
desc_set.set_image(0, texture)

# Record commands
cmd = ctx.create_command_buffer()
cmd.begin()
with cmd.rendering(renderer, clear_color=[0.1, 0.2, 0.3, 1.0]) as c:
    (c.bind_pipeline(pipeline)
      .bind_descriptor_set(desc_set, pipeline)
      .bind_vertex_buffer(vbuf)
      .bind_index_buffer(ibuf)
      .draw_indexed(6))

# Main loop
while window.is_open():
    bz.poll_events()
    ctx.begin_frame()
    if renderer.acquire():
        renderer.present(cmd)
