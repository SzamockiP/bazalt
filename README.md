# Bazalt

**Bazalt** is a modern Python library for rapid prototyping and building graphical applications using the Vulkan API. It provides a clean, intuitive interface over a high-performance C++ core, allowing developers to create rendering applications quickly without the typical boilerplate.

## Installation

You can install `bazalt` easily via `pip`:

```bash
pip install bazalt
```

Prebuilt wheels are provided for Windows and Linux. Building from source
requires a C++23 compiler — GCC 14+ or MSVC 19.36+ (Visual Studio 17.6) —
and the Vulkan SDK.

## Key Features

- **Modern Graphics API:** Built on top of Vulkan for optimal hardware utilization.
- **Easy to Use Interface:** Write clear and concise code with an intuitive API.
- **Automatic Shader Compilation:** Compile GLSL shaders (Vertex/Fragment) directly from your code.
- **Pipeline & Buffer Management:** Easy builder pattern for graphics pipelines and unified buffer creation.
- **Command Buffers:** Explicit, yet simple command recording for drawing operations.
- **Headless Rendering:** Draw into an offscreen `RenderTarget` and read the pixels back as a NumPy array — no window, no display required.
- **Runs Widely:** Vulkan 1.2 baseline with 1.3 used where available, so bazalt runs on older integrated GPUs too. Capabilities are requested by name, never by version or extension.
- **Decoupled Architecture:** Clean separation of concerns between Windowing (GLFW), Vulkan Context (GPU initialization), and render targets — a window is one target among others.

## Quick Start: Rendering Without a Window

Everything below works with no display attached, which makes it usable from CI
and tests:

```python
import bazalt as bz

ctx = bz.Context()
target = bz.RenderTarget(ctx, 800, 600, depth=True)

pipeline = (ctx.pipeline_builder()
    .vertex_shader(ctx.compile_shader("triangle.vert", bz.ShaderStage.VERTEX))
    .fragment_shader(ctx.compile_shader("triangle.frag", bz.ShaderStage.FRAGMENT))
    .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
    .build(target))

# Interleaved position (x,y,z) and color (r,g,b)
vbuf = ctx.create_buffer([
     0.0, -0.5, 0.0,   1.0, 0.0, 0.0,
    -0.5,  0.5, 0.0,   0.0, 1.0, 0.0,
     0.5,  0.5, 0.0,   0.0, 0.0, 1.0,
], bz.BufferType.VERTEX, bz.MemoryUsage.STATIC, bz.DataType.FLOAT)

cmd = ctx.create_command_buffer()
cmd.begin()
cmd.begin_rendering(target, clear_color=[0.1, 0.2, 0.3, 1.0])
cmd.bind_pipeline(pipeline)
cmd.bind_vertex_buffer(vbuf)
cmd.draw(3)
cmd.end_rendering(target)

ctx.submit(cmd)

pixels = target.read_pixels()   # numpy (600, 800, 4) uint8
```

## Quick Start: Drawing a Triangle

Here is a minimal example demonstrating how to initialize the window, Vulkan Context, and SwapchainRenderer, compile shaders, create a pipeline, and draw a colorful triangle.

```python
import bazalt as bz

# 1. Initialize Logger and register a callback.
#    Each message carries its severity and source as data, so you can filter
#    without parsing strings. This is optional: a Context created without one
#    reports warnings on stderr by default.
logger = bz.Logger(min_severity=bz.Severity.WARNING)
@logger.on_message
def on_message(msg):
    print(f"[{msg.severity}] {msg.text}")

# 2. Create the window, Vulkan Context, and SwapchainRenderer
window = bz.Window(1024, 720, "Bazalt Demo - Triangle", logger=logger)
ctx = bz.Context(logger)
renderer = bz.SwapchainRenderer(window, ctx)

if __name__ == "__main__":
    # Load and compile shaders. The vertex shader processes our geometry,
    # and the fragment shader determines the final color of the pixels.
    # Shaders are compiled through the Context.
    vert_spv = ctx.compile_shader("triangle.vert", bz.ShaderStage.VERTEX)
    frag_spv = ctx.compile_shader("triangle.frag", bz.ShaderStage.FRAGMENT)

    # The pipeline is a baked state object that tells the GPU how to interpret our data.
    # It is built against a render target, which supplies the color/depth formats.
    # A SwapchainRenderer is one, so is an offscreen RenderTarget — same call.
    pipeline = (ctx.pipeline_builder()
        .vertex_shader(vert_spv)
        .fragment_shader(frag_spv)
        .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3]) # Position + Color
        .build(renderer))

    # We interleave Position (x,y,z) and Color (r,g,b) in a single flat array.
    # Vulkan's Normalized Device Coordinates (NDC) range from -1 to 1,
    # where Y points downwards and X points to the right.
    vertices = [
         0.0, -0.5, 0.0,   1.0, 0.0, 0.0, # Top / Red
        -0.5,  0.5, 0.0,   0.0, 1.0, 0.0, # Bottom-Left / Green
         0.5,  0.5, 0.0,   0.0, 0.0, 1.0, # Bottom-Right / Blue
    ]
    
    # Create Vertex Buffer through the Context
    vbuf = ctx.create_buffer(vertices, bz.BufferType.VERTEX, bz.MemoryUsage.STATIC, bz.DataType.FLOAT)
    
    # Create Index Buffer through the Context
    ibuf = ctx.create_buffer([0, 1, 2], bz.BufferType.INDEX, bz.MemoryUsage.STATIC, bz.DataType.UINT32)

    # Command buffers store a sequence of commands for the GPU.
    # For a static triangle, we can record this buffer once during initialization 
    # and submit the same pre-recorded buffer every frame to save CPU time.
    cmd = ctx.create_command_buffer()

    cmd.begin()

    # begin_rendering names the target you are drawing into, clears it, and sets
    # a viewport and scissor covering it. Naming the target is what lets the same
    # code render to a window or to an offscreen image.
    cmd.begin_rendering(renderer, clear_color=[0.1, 0.2, 0.3, 1.0])

    # Bind the baked pipeline and the geometry buffers
    cmd.bind_pipeline(pipeline)
    cmd.bind_vertex_buffer(vbuf)
    cmd.bind_index_buffer(ibuf)

    # Draw 3 indices (1 triangle)
    cmd.draw_indexed(3)
    cmd.end_rendering(renderer)

    # Main game/rendering loop
    while window.is_open():
        window.poll_events()
        
        # begin_frame returns a Frame, or None when this frame should be
        # skipped (window minimized, mid-resize)
        if frame := renderer.begin_frame():
            frame.submit(cmd)
```
