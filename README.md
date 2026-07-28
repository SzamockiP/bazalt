# Bazalt

**Bazalt** is a Python library for the Vulkan API. Use it to prototype GPU and shader
work, and to build graphical applications. A C++23 core does the Vulkan work. The Python
layer keeps the setup code short.

## Installation

Install `bazalt` with `pip`:

```bash
pip install bazalt
```

Bazalt supplies prebuilt wheels for Windows and Linux. To build from source, you need a
C++23 compiler (GCC 14+ or MSVC 19.36+, Visual Studio 17.6) and the Vulkan SDK.

## Key Features

- **Vulkan Core:** Bazalt calls Vulkan for all GPU work. You keep explicit control of the hardware.
- **Shader Compilation:** `ctx.compile_shader()` compiles GLSL vertex, fragment and compute shaders from your code.
- **Compute Pipelines:** `ctx.compute_pipeline()` and `cmd.dispatch()` run GPU compute. The results go back into NumPy. You need no images.
- **Storage Images:** Compute shaders also write images. `.storage_image(binding)` and `set_storage_image()` bind a read/write image for `imageStore` and `imageLoad`. Use this for post-processing and procedural textures. Bazalt records the barriers and the layout transitions between the dispatch and a later sample.
- **Cubemaps and Texture Arrays:** Pass a list of layers to `ctx.create_image([...], cube=True)` or to `ctx.load_image([...6 paths...], cube=True)` for a cubemap. Omit `cube=` for a texture array. One mechanism gives you `samplerCube` and `sampler2DArray`. Compute can fill empty layered images, for example a procedural skybox. See `examples/14_skybox`.
- **Mipmaps:** `load_image()` makes mips for files by default. Pass `mipmaps=False` to stop this. `create_image(array, mipmaps=True)` makes mips for a NumPy texture. `create_image(w, h, mip_levels=N)` allocates an empty chain, and `cmd.generate_mipmaps(image)` fills it from mip 0. This works across every array layer and every cube face.
- **Automatic Barriers:** Bazalt computes the barriers for hazards between resources at record time. Examples: dispatch to dispatch, a compute-written buffer to a vertex fetch, a compute-written image to a sample. `Context(auto_barriers=False)` gives you manual control through `cmd.barrier()`.
- **Pipeline and Buffer Management:** A builder makes graphics pipelines. One function makes every buffer type.
- **Command Buffers:** Command recording is explicit. Calls chain, as in `cmd.bind_pipeline(p).draw(3)`. The block `with cmd.rendering(target):` ends the pass for you.
- **Asynchronous Texture Streaming:** `ctx.load_image()` returns immediately. The decode and the GPU copy run in the background. Anything that samples the image waits for it. `ctx.upload_progress` gives you the data for a loading bar. Pass `bytes` instead of a path for an image with no file behind it, for example one you downloaded. `ctx.submit(cmd, wait=False)` does the same for a headless submit, and `ctx.wait()` collects the results.
- **One Rule for What Blocks:** Every write is asynchronous and every read blocks. `create_buffer`, `create_image` and `load_image` give you the handle before the GPU has the data, and the handle is its own future. Anything that uses it waits for it, so you write no wait at all in normal code. `buf.read()`, `img.read()` and `target.read_pixels()` block, because they give you an array and have nothing to give you until the bytes arrive. Three verbs wait when you want to be explicit: `res.wait()` for one resource, `ctx.wait_for_uploads()` for every upload, and `ctx.wait()` for every submit.
- **Hot Reload:** `Context(hot_reload=True)` watches the shaders you loaded, their `#include` files, and your images. Bazalt applies edits live: shaders recompile and rebuild their pipelines in place, and images upload again into the same handle. Bazalt logs a typo or a bad file and keeps the last good version on screen. A mistake does not stop the application. See `examples/12_hot_reload`.
- **Frame Timing and Debug Names:** `Context(gpu_timing=True)` makes `renderer.gpu_time_ms` report the GPU time of a recent frame. This is opt-in and costs nothing when off. Use `t = cmd.timer()` and `t.stop()` to time a slice of a recording, then read `t.ms`. A timer is a handle, so you need no `with` block, and it reads back headless. The `name=` argument and the `.name()` method label buffers, images and pipelines, so validation messages name the object.
- **Headless Rendering:** Draw into an offscreen `RenderTarget` and read the pixels back as a NumPy array. You need no window and no display.
- **Render-to-Texture, MRT and Shadow Maps:** Target attachments are ordinary `Image` objects in any supported `Format`. Sample `target.color[0]`, or `target.depth` on a depth-only target, like any other texture.
- **Window Modes:** `window.set_mode()` switches between `WINDOWED`, `FRAMELESS`, `FULLSCREEN` and `FULLSCREEN_WINDOWED` while the application runs. `bz.Window(..., mode=)` opens in a mode. Fullscreen takes the monitor that the window covers most of, so a second display gets the window it holds. The swapchain follows on its own. You can also set the size, the position, the resizable flag, the always-on-top flag and the opacity, and read `window.content_scale` for a HiDPI display. See `examples/21_window_modes`.
- **Per-Frame Input:** `window.was_key_pressed(key)` reports the key going down during the last `bz.poll_events()`. Use it for a toggle, where `is_key_pressed` gives you the level. `get_mouse_state()` reports `dx`, `dy` and the scroll wheel for that one cycle, so a camera needs no bookkeeping. Every one of these reads the same twice inside one frame.
- **Multi-Pass Targets:** `cmd.rendering(target, clear_color=None)` keeps the colour and the depth the last pass left. Use it to draw opaque geometry, then transparent geometry, then a UI into one image. `renderer.set_present_mode()` switches vsync at runtime.
- **Blending and Depth Control:** `blend(True, mode=bz.BlendMode.ADDITIVE)` gives you glow and particles. `PREMULTIPLIED` composites a texture that carries its alpha. `depth_test(True, write=False)` tests without writing, which a transparency pass needs, and `compare=` replaces the default `LESS_OR_EQUAL`. Use `clear_depth=0.0` with `compare=bz.CompareOp.GREATER` for a reversed-depth buffer. `depth_bias()` fixes shadow acne. `polygon_mode(bz.PolygonMode.LINE)` draws a wireframe, and `line_width()` makes it visible on a HiDPI display.
- **Instancing:** `instance_format([...])` declares a second vertex buffer. That buffer advances once per instance, not once per vertex. Bind it with `cmd.bind_vertex_buffer(buf, binding=1)` and draw with `draw_indexed(36, instances=20000)`. The mesh is stored one time, and each object supplies its own row of data. `bz.VertexFormat.UBYTE4_NORM` puts a colour in four bytes. See `examples/22_instancing`.
- **Stencil:** `RenderTarget(..., depth=bz.Format.DEPTH_STENCIL)` or `SwapchainRenderer(..., stencil=True)` adds a stencil buffer. `stencil_test()` on the pipeline masks a pass by what an earlier pass drew. Use it for outlines, portals, masks and decals. `clear_stencil=` sets the start value. See `examples/23_outline`.
- **Specialization Constants:** `.constant(id, value, stage)` supplies a value when the pipeline is built. One compiled shader serves several pipelines that differ by a number, for example a quality level or a kernel radius. The driver folds the value, so a constant `False` removes the branch behind it. Bazalt also caches pipeline compilation for each Context, which makes a hot reload faster.
- **Image Transfers:** `cmd.copy_image(src, dst)` copies one image into another of the same size and format, across every mip level the two share. Use it for a history buffer, for a temporal effect, or for a compute ping-pong. `cmd.blit_image(src, dst)` copies between two different sizes and scales the pixels, which is what a bloom downsample or a thumbnail needs. `cmd.clear_image(image, color)` fills an image with no pipeline and no pass. `cmd.copy_buffer()` and `cmd.fill_buffer()` do the same for buffers, so you can zero a counter without a dispatch.
- **CPU Image Streaming:** `img.update(array)` writes new pixels into an image that already exists. Use it for a video frame, a camera feed, a plot, or a texture you compute in NumPy. Give it `region=(x, y, w, h)` to write one rectangle and keep the rest, which is what painting needs. The update runs on the upload worker, so the call returns at once and the frame that samples the image waits for it. Two updates of one image arrive in the order you made them. `img.read(layer=, mip=)` reads any face or level back. See `examples/24_video_texture`.
- **Screenshots:** `renderer.present(cmd, capture=True)` copies the frame you present, and `renderer.read_pixels()` gives it to you as a NumPy array. It takes two calls because Vulkan lets you touch a window image only between `acquire()` and `present()`, so the copy travels with that frame.
- **Shader printf:** `Context(shader_printf=True)` sends `debugPrintfEXT()` output from your shaders to the logger. Write `#extension GL_EXT_debug_printf : enable` in the shader and print the value you want to see. The validation layers do this work, so you need them installed. Bazalt also compiles that Context's shaders without the optimizer, because the optimizer is allowed to delete a print.
- **Capture Labels and Memory:** `with cmd.label("shadow pass"):` names a scope, so a RenderDoc capture reads as a frame and not as a list of draws. `ctx.memory_stats()` reports the memory you hold and the memory you have left. `cmd.occlusion_query()` counts the fragments a group of draws passed.
- **Shader Sources:** `compile_shader(path, stage)` reads the file. `source=` takes the content instead: a `str` is compiled, and `bytes` is loaded as ready SPIR-V with the same checks a `.spv` file gets. `include_dirs=` adds fallback directories for `#include`, tried only when the name is not beside the including file. `entry_point=` picks one HLSL function out of a file that holds several, for example `VSMain` beside `PSMain`. A hot reload keeps both settings.
- **Multi-Window:** One Context drives any number of windows. `ctx.begin_frame()` opens the frame. Each window calls `renderer.acquire()` and `renderer.present(cmd)`, and owns its own swapchain. If you resize or close one window, the other windows continue to render. See `examples/19_multi_window`.
- **GPU Selection:** `bz.list_devices()` lists every card with its name, type, VRAM, API version and per-feature support, before a Context exists. `Context(device=...)` runs on the card you chose. Omit the argument and bazalt chooses for you.
- **Multi-Context and Multi-GPU:** Any number of Contexts can be alive at the same time, on one card or on different cards. You can bake with compute on one GPU and render on the other. `other_ctx.create_image(image)` moves the pixels across with their format, their layers and their cube property. Every other resource stays with its Context. If you give a resource to the wrong command buffer, you get a `ResourceError` that names the mistake, not a driver crash. See `examples/20_multi_context`.
- **Wide Hardware Support:** Bazalt needs Vulkan 1.2 and uses 1.3 where the driver has it, so it also runs on older integrated GPUs. You request capabilities by name, never by version or by extension.
- **Separated Layers:** Windowing (GLFW), the Vulkan Context (GPU initialization) and render targets stay separate. A window is one render target among others.

## Quick Start: Rendering Without a Window

The code below runs with no display attached. You can use it from CI and from tests:

```python
import bazalt as bz

ctx = bz.Context()
target = bz.RenderTarget(ctx, 800, 600, depth=bz.Format.D32F)

pipeline = (ctx.graphics_pipeline()
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
with cmd.rendering(target, clear_color=[0.1, 0.2, 0.3, 1.0]) as c:
    c.bind_pipeline(pipeline).bind_vertex_buffer(vbuf).draw(3)

ctx.submit(cmd)

pixels = target.read_pixels()   # numpy (600, 800, 4) uint8
```

## Quick Start: GPU Compute

Compute needs no window and no images. Dispatch, then read the storage buffer back as a
NumPy array:

```python
import numpy as np
import bazalt as bz

ctx = bz.Context()

# double.comp: values[i] *= 2.0 over a std430 float array, local_size_x = 64
sim = (ctx.compute_pipeline()
    .shader(ctx.compile_shader("double.comp", bz.ShaderStage.COMPUTE))
    .storage_buffer(0)      # no stage argument — compute has exactly one stage
    .build())               # no target — compute has no attachments

data = np.arange(128, dtype=np.float32)
sbuf = ctx.create_buffer(data, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)

pool = ctx.create_descriptor_pool(max_sets=4, storage_buffers=4)
dset = pool.allocate_set(sim, set=0)
dset.set_buffer(0, sbuf)

cmd = ctx.create_command_buffer()
cmd.begin()
cmd.bind_pipeline(sim).bind_descriptor_set(dset, sim, set=0).dispatch(128 // 64)
ctx.submit(cmd)

assert np.allclose(sbuf.read(np.float32), data * 2)
```

You can mix compute and rendering in one command buffer. A dispatch writes the vertices
and a draw reads them, with no extra code. Bazalt records the barrier between them. See
`examples/11_particles`.

## Quick Start: Hot Reload

Add one keyword argument. Bazalt then watches the files it loaded: the shaders, their
`#include` files, and the images. On each save, bazalt recompiles and uploads again:

```python
ctx = bz.Context(logger, hot_reload=True)

vert = ctx.compile_shader("shader.vert", bz.ShaderStage.VERTEX)
frag = ctx.compile_shader("shader.frag", bz.ShaderStage.FRAGMENT)
tex  = ctx.load_image("wall.png")
pipeline = ctx.graphics_pipeline().vertex_shader(vert).fragment_shader(frag)...build(renderer)

while window.is_open():
    bz.poll_events()
    ctx.begin_frame()                  # edits are applied here (and at ctx.submit)
    if renderer.acquire():
        renderer.present(cmd)          # renderer.gpu_time_ms (with gpu_timing=True) gives GPU frame timing
```

When you edit `shader.frag`, bazalt rebuilds the pipeline in place. When you save
`wall.png` again with the same size and the same format, bazalt uploads it into the same
handle, so the descriptor sets stay correct. A shader typo logs a `ShaderError`, and the
last good pipeline continues to render. A mistake does not stop the application. For the
full demo, see `examples/12_hot_reload`.

## Quick Start: Drawing a Triangle

This example starts the window, the Vulkan Context and the SwapchainRenderer. It then
compiles the shaders, creates a pipeline and draws a triangle with three colors.

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
    pipeline = (ctx.graphics_pipeline()
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
        bz.poll_events()
        
        # begin_frame opens one logical frame for every window on this
        # Context; acquire() takes this window's swapchain image and returns
        # False when it should sit the frame out (minimized, mid-resize)
        ctx.begin_frame()
        if renderer.acquire():
            renderer.present(cmd)
```
