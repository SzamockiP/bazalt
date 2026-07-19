import bazalt as bz

# Create window, logger, and renderer
logger = bz.Logger()
@logger.on_message
def on_message(msg):
    print(msg)

window = bz.Window(800, 600, "Bazalt Demo - Textured Quad")
ctx = bz.Context(logger)
renderer = bz.SwapchainRenderer(window, ctx)

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
pool = ctx.create_descriptor_pool(max_sets=1, samplers=1)
desc_set = pool.allocate_set(pipeline, set=0)
desc_set.set_image(0, texture)

# Record commands
cmd = ctx.create_command_buffer()
cmd.begin()
cmd.begin_rendering(renderer, clear_color=[0.1, 0.2, 0.3, 1.0])
cmd.bind_pipeline(pipeline)
cmd.bind_descriptor_set(desc_set, pipeline, set=0)
cmd.bind_vertex_buffer(vbuf)
cmd.bind_index_buffer(ibuf)
cmd.draw_indexed(6)
cmd.end_rendering(renderer)

# Main loop
while window.is_open():
    window.poll_events()
    if frame := renderer.begin_frame():
        frame.submit(cmd)
