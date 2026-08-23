import bazalt as bz
import numpy as np

# Create window, logger, and renderer
logger = bz.Logger()
@logger.on_message
def on_message(msg):
    print(f"[{msg.severity}] {msg.text}")

window = bz.Window(1024, 720, "Bazalt Demo - Triangle", logger=logger)
ctx = bz.Context(logger)
renderer = ctx.create_renderer(window)

# Load and compile shaders
vert_spv = ctx.compile_shader("triangle.vert", bz.ShaderStage.VERTEX)
frag_spv = ctx.compile_shader("triangle.frag", bz.ShaderStage.FRAGMENT)

# Build pipeline: Position (FLOAT3) + Color (FLOAT3)
pipeline = (ctx.graphics_pipeline()
    .vertex_shader(vert_spv)
    .fragment_shader(frag_spv)
    .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
    .build(renderer))

# Interleaved Position (x,y,z) and Color (r,g,b)
vertices = [
     0.0, -0.5, 0.0,   1.0, 0.0, 0.0, # Top / Red
    -0.5,  0.5, 0.0,   0.0, 1.0, 0.0, # Bottom-Left / Green
     0.5,  0.5, 0.0,   0.0, 0.0, 1.0, # Bottom-Right / Blue
]
vbuf = ctx.create_buffer(vertices, bz.BufferUsage.VERTEX, bz.MemoryUsage.STATIC)

indices = [0, 1, 2]
ibuf = ctx.create_buffer(indices, bz.BufferUsage.INDEX, bz.MemoryUsage.STATIC, dtype=np.uint32)

# Build the graph once
g = ctx.graph()
with g.add_pass(renderer, clear_color=[0.1, 0.2, 0.3, 1.0]) as p:
    (p.bind_pipeline(pipeline)
      .bind_vertex_buffer(vbuf)
      .bind_index_buffer(ibuf)
      .draw_indexed(3))

# Main loop. One graph, sent every frame: a pass holds lambdas, not a frame's
# worth of state, so nothing here needs a rebuild.
while window.is_open():
    bz.poll_events()
    ctx.begin_frame()
    if renderer.acquire():
        renderer.present(g)