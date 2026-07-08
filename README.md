# Bazalt

**Bazalt** is a modern Python library for rapid prototyping and building graphical applications using the Vulkan API. It provides a clean, intuitive interface over a high-performance C++ core, allowing developers to create rendering applications quickly without the typical boilerplate.

## Installation

You can install `bazalt` easily via `pip`:

```bash
pip install bazalt
```

## Key Features

- **Modern Graphics API:** Built on top of Vulkan for optimal hardware utilization.
- **Easy to Use Interface:** Write clear and concise code with an intuitive API.
- **Automatic Shader Compilation:** Compile GLSL shaders (Vertex/Fragment) directly from your code.
- **Pipeline & Buffer Management:** Easy builder pattern for graphics pipelines and unified buffer creation.
- **Command Buffers:** Explicit, yet simple command recording for drawing operations.
- **Event-Driven Architecture:** Register callbacks easily using decorators (e.g., `@engine.onFrame`, `@engine.onError`).

## Quick Start: Drawing a Triangle

Here is a minimal example demonstrating how to initialize the engine, compile shaders, create a pipeline, and draw a colorful triangle.

```python
import bazalt as bz
import time

engine = bz.Engine()

# Optional: Register an error callback
@engine.onError
def error(msg):
    print(f"Error: {msg}")

# Register a per-frame callback
@engine.onFrame
def on_update():
    # Submit our pre-recorded command buffer to the GPU each frame
    engine.submit(cmd)

if __name__ == "__main__":
    engine.init(1024, 720, "Bazalt Demo - Triangle")

    # Load and compile shaders. The vertex shader processes our geometry,
    # and the fragment shader determines the final color of the pixels.
    vert_spv = engine.compileShader("triangle.vert", bz.ShaderStage.VERTEX)
    frag_spv = engine.compileShader("triangle.frag", bz.ShaderStage.FRAGMENT)

    # The pipeline is a baked state object that tells the GPU how to interpret our data.
    # vertexFormat specifies that each vertex consists of 6 floats:
    # 3 for Position (layout location=0) and 3 for Color (layout location=1).
    pipeline = (engine.createPipeline()
        .vertexShader(vert_spv)
        .fragmentShader(frag_spv)
        .vertexFormat([bz.Format.FLOAT3, bz.Format.FLOAT3]) # Position + Color
        .build())

    # We interleave Position (x,y,z) and Color (r,g,b) in a single flat array.
    # Vulkan's Normalized Device Coordinates (NDC) range from -1 to 1,
    # where Y points downwards and X points to the right.
    vertices = [
         0.0, -0.5, 0.0,   1.0, 0.0, 0.0, # Top / Red
        -0.5,  0.5, 0.0,   0.0, 1.0, 0.0, # Bottom-Left / Green
         0.5,  0.5, 0.0,   0.0, 0.0, 1.0, # Bottom-Right / Blue
    ]
    
    # Create Vertex Buffer
    vbuf = engine.createBuffer(vertices, bz.BufferType.VERTEX, bz.DataType.FLOAT)
    
    # Create Index Buffer
    indices = [0, 1, 2]
    ibuf = engine.createBuffer(indices, bz.BufferType.INDEX, bz.DataType.UINT32)

    # Command buffers store a sequence of commands for the GPU.
    # For a static triangle, we can record this buffer once during initialization 
    # and submit the same pre-recorded buffer every frame to save CPU time.
    cmd = engine.createCommandBuffer()
    
    cmd.begin()
    
    # beginRendering starts a render pass, automatically handling framebuffers
    # and clearing the screen to the specified color before drawing.
    cmd.beginRendering(clear_color=[0.1, 0.2, 0.3, 1.0])
    cmd.setViewport()
    cmd.setScissor()
    
    # Bind the baked pipeline and the geometry buffers
    cmd.bindPipeline(pipeline)
    cmd.bindVertexBuffer(vbuf)
    cmd.bindIndexBuffer(ibuf)
    
    # Draw 3 indices (1 triangle)
    cmd.drawIndexed(3)
    cmd.endRendering()

    engine.run()
```
