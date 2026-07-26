import time

import bazalt as bz

# Create window, logger, and renderer
logger = bz.Logger()
@logger.on_message
def on_message(msg):
    print(f"[{msg.severity}] {msg.text}")

window = bz.Window(1024, 720, "Bazalt Demo - Triangle", logger=logger)
ctx = bz.Context(logger)
renderer = bz.SwapchainRenderer(window, ctx)

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
vbuf = ctx.create_buffer(vertices, bz.BufferType.VERTEX, bz.MemoryUsage.STATIC, bz.DataType.FLOAT)

indices = [0, 1, 2]
ibuf = ctx.create_buffer(indices, bz.BufferType.INDEX, bz.MemoryUsage.STATIC, bz.DataType.UINT32)

# Record command buffer once
cmd = ctx.create_command_buffer()
cmd.begin()
with cmd.rendering(renderer, clear_color=[0.1, 0.2, 0.3, 1.0]) as c:
    (c.bind_pipeline(pipeline)
      .bind_vertex_buffer(vbuf)
      .bind_index_buffer(ibuf)
      .draw_indexed(3))

# Main loop
last_time = time.time()
frame_count = 0
fps_timer = 0.0

while window.is_open():
    window.poll_events()
    ctx.begin_frame()
    if renderer.acquire():
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        frame_count += 1
        fps_timer += dt
        if fps_timer >= 1.0:
            avg_fps = frame_count / fps_timer
            window.set_title(f"Bazalt Demo - Triangle | {1000.0 / avg_fps:.2f} ms/frame | {avg_fps:.1f} FPS")
            frame_count = 0
            fps_timer = 0.0

        renderer.present(cmd)