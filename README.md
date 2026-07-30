# Bazalt

Bazalt is a Python library for the Vulkan API. Use it to prototype GPU and shader work, and
to build graphical applications. A C++23 core does the Vulkan work. The Python layer keeps
the setup code short.

```bash
pip install bazalt
```

Bazalt supplies wheels for Windows, Linux and macOS, for CPython 3.10 to 3.14. You need a
driver with Vulkan 1.2.

On macOS you must also install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home). The
system supplies no Vulkan loader, and the SDK gives you the loader and MoltenVK, which runs
Vulkan on Metal. The macOS wheels are for Apple Silicon and need macOS 14 or later.

To build from source, you need a C++23 compiler (GCC 14+, MSVC 19.36+ from Visual Studio
17.6, or Apple Clang 15+) and the Vulkan SDK.

## A triangle

Three files: the program, a vertex shader and a fragment shader. Put them in one directory
and run the program.

```python
import bazalt as bz

window = bz.Window(1024, 720, "Bazalt - triangle")
ctx = bz.Context()
renderer = bz.SwapchainRenderer(window, ctx)

# A pipeline is built against a render target, which supplies the formats.
# The window is one target. An offscreen image is another. Same call.
pipeline = (ctx.graphics_pipeline()
    .vertex_shader(ctx.compile_shader("triangle.vert", bz.ShaderStage.VERTEX))
    .fragment_shader(ctx.compile_shader("triangle.frag", bz.ShaderStage.FRAGMENT))
    .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])  # position + colour
    .build(renderer))

# Interleaved position (x, y, z) and colour (r, g, b). Vulkan clip space runs
# from -1 to 1, and y points down.
vbuf = ctx.create_buffer([
     0.0, -0.5, 0.0,   1.0, 0.0, 0.0,
    -0.5,  0.5, 0.0,   0.0, 1.0, 0.0,
     0.5,  0.5, 0.0,   0.0, 0.0, 1.0,
], bz.BufferType.VERTEX, bz.MemoryUsage.STATIC, bz.DataType.FLOAT)

# The triangle does not change, so record the commands one time and send the
# same recording every frame.
cmd = ctx.create_command_buffer()
cmd.begin()
with cmd.rendering(renderer, clear_color=[0.1, 0.2, 0.3, 1.0]) as c:
    c.bind_pipeline(pipeline).bind_vertex_buffer(vbuf).draw(3)

while window.is_open():
    bz.poll_events()
    ctx.begin_frame()          # opens one frame for every window on this Context
    if renderer.acquire():     # False when this window sits the frame out
        renderer.present(cmd)
```

```glsl
// triangle.vert
#version 450
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = vec4(inPosition, 1.0);
    fragColor = inColor;
}
```

```glsl
// triangle.frag
#version 450
layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(fragColor, 1.0);
}
```

## Compute writes an image, and you edit it while it runs

A compute shader fills an image texel by texel. A fullscreen triangle then samples that
image. Two things happen here that you do not write:

- Bazalt sees that the dispatch writes the image and the draw reads it. It records the
  barrier and the layout change between them.
- `hot_reload=True` watches every file bazalt loaded. Edit `pattern.comp` with the window
  open and the picture changes. A typo logs a `ShaderError` and the last good version stays
  on screen.

```python
import struct
import time

import bazalt as bz

W = H = 512

window = bz.Window(W, H, "Bazalt - compute")
ctx = bz.Context(hot_reload=True)
renderer = bz.SwapchainRenderer(window, ctx)

# Compute: no vertices and no fragment shader. One dispatch per frame.
generate = (ctx.compute_pipeline()
    .shader(ctx.compile_shader("pattern.comp", bz.ShaderStage.COMPUTE))
    .storage_image(0)
    .push_constant(4)          # 4 bytes, one float: the time
    .build())                  # no target, because compute has no attachments

# Graphics: a fullscreen triangle that samples the result.
present = (ctx.graphics_pipeline()
    .vertex_shader(ctx.compile_shader("fullscreen.vert", bz.ShaderStage.VERTEX))
    .fragment_shader(ctx.compile_shader("present.frag", bz.ShaderStage.FRAGMENT))
    .texture(0, bz.ShaderStage.FRAGMENT, set=0)
    .build(renderer))

# One image, bound two ways: written as a storage image, read as a texture.
image = ctx.create_image(W, H, bz.Format.RGBA8)

pool = ctx.create_descriptor_pool(max_sets=2, samplers=1, storage_images=1)
write_set = pool.allocate_set(generate, set=0)
write_set.set_storage_image(0, image)
read_set = pool.allocate_set(present, set=0)
read_set.set_image(0, image)

cmd = ctx.create_command_buffer()
start = time.time()

while window.is_open():
    bz.poll_events()
    ctx.begin_frame()          # bazalt applies the file edits here
    if not renderer.acquire():
        continue

    cmd.begin()
    (cmd.bind_pipeline(generate)
        .bind_descriptor_set(write_set, generate, set=0)
        .push_constants(generate, 0, struct.pack("<f", time.time() - start))
        .dispatch((W + 7) // 8, (H + 7) // 8))
    with cmd.rendering(renderer) as c:
        c.bind_pipeline(present).bind_descriptor_set(read_set, present, set=0).draw(3)

    renderer.present(cmd)
```

```glsl
// pattern.comp
#version 450
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;
layout(push_constant) uniform Push { float time; } pc;

void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    ivec2 sz = imageSize(img);
    if (p.x >= sz.x || p.y >= sz.y) return;

    vec2 uv = vec2(p) / vec2(sz);
    float v = 0.5 + 0.5 * sin(uv.x * 12.0 + pc.time) * cos(uv.y * 12.0 - pc.time * 0.7);
    imageStore(img, p, vec4(v, uv.x, uv.y, 1.0));
}
```

```glsl
// fullscreen.vert — a triangle from gl_VertexIndex, with no vertex buffer
#version 450
layout(location = 0) out vec2 uv;

void main() {
    uv = vec2(gl_VertexIndex & 2, (gl_VertexIndex << 1) & 2);
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
```

```glsl
// present.frag
#version 450
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;
layout(set = 0, binding = 0) uniform sampler2D tex;

void main() {
    outColor = texture(tex, uv);
}
```

## The same code, with no window

A `RenderTarget` is an offscreen image with the same interface as the window. Build the
pipeline against it, submit, and read the pixels into NumPy. This runs in a test and in CI,
where no display exists:

```python
ctx = bz.Context()
target = bz.RenderTarget(ctx, 800, 600, depth=bz.Format.D32F)

pipeline = ...   # the triangle pipeline above, built with .build(target)

cmd = ctx.create_command_buffer()
cmd.begin()
with cmd.rendering(target, clear_color=[0.1, 0.2, 0.3, 1.0]) as c:
    c.bind_pipeline(pipeline).bind_vertex_buffer(vbuf).draw(3)
ctx.submit(cmd)

pixels = target.color[0].read()      # numpy (600, 800, 4) uint8
```

## What you get

- **Vulkan, and not an engine.** You keep the pipelines, the command buffers and the memory.
  Bazalt removes the setup code and the object lifetimes.
- **Barriers on their own.** Bazalt tracks the hazards while you record, then inserts the
  barriers and the layout changes. It reads the SPIR-V to find out which resources a shader
  writes, so a storage image written by a fragment shader is ordered for you.
  `Context(auto_barriers=False)` gives that job back to you through `cmd.barrier()`.
- **Compute beside graphics.** One command buffer holds a dispatch and a draw. A dispatch
  writes the vertices and the draw reads them. Results come back as NumPy arrays.
- **Every pipeline stage.** Vertex, fragment and compute, plus tessellation for displacement
  and adaptive detail, and geometry for a primitive that becomes a different one. A dispatch
  or a draw can also read its own arguments from a buffer the GPU filled, so the count never
  travels back to the CPU.
- **One rule for what blocks.** Every write is asynchronous and every read blocks. A handle
  is its own future, so normal code waits nowhere. `buf.read()` blocks, because it has
  nothing to give you until the bytes arrive.
- **Hot reload.** Bazalt watches the shaders you loaded, their `#include` files and your
  images, then applies the edits in place. A mistake does not stop the application.
- **Headless.** Draw into an offscreen target and read the pixels as a NumPy array. You need
  no window and no display.
- **The window, and what arrives through it.** Keys, the mouse and the scroll wheel, plus
  files dropped on the window, the clipboard through `bz.get_clipboard()` and
  `bz.set_clipboard()`, an icon through `window.set_icon()` and the cursor through
  `window.set_cursor_position()`.
- **Tools for a picture that looks wrong.** The validation layers report through a Python
  logger. `Context(shader_printf=True)` sends `debugPrintfEXT()` from a shader to that
  logger. `Context(gpu_timing=True)` and `cmd.timer()` measure a frame or one slice of a
  recording. `cmd.label()` makes a RenderDoc capture readable. A compiled shader also tells
  you what bazalt read out of it: `shader.writes`, `shader.writes_unknown` and
  `shader.prints`.
- **Wide reach.** Vulkan 1.2 is the baseline and bazalt uses 1.3 where the driver has it. You
  ask for a capability by what it does, never by a version or an extension name.
- **No ceiling.** `cmd.barrier()`, `raw_extensions` and the Vulkan handles stay open for the
  work bazalt does not cover.

## Design rules

Four rules decide what goes into the API. `DESIGN.md` holds them with their reasons.

- One obvious way to do one thing. A resource that differs by one parameter is a keyword
  argument, not a new function.
- Never cap the ceiling. Each layer keeps an escape hatch to the Vulkan below it.
- It must run on more than 90% of machines.
- Prototyping comes first. A feature that makes pictures ranks before plumbing.

## Examples

Every directory in `examples/` runs on its own.

| Subject | Examples |
| --- | --- |
| Basics | [01_empty_window](examples/01_empty_window), [02_triangle](examples/02_triangle), [03_textured_quad](examples/03_textured_quad), [04_colored_cube](examples/04_colored_cube), [05_textured_cube](examples/05_textured_cube), [06_multiple_cubes](examples/06_multiple_cubes), [07_model_loading](examples/07_model_loading) |
| Compute | [11_particles](examples/11_particles) (compute writes the vertices), [13_compute_postprocess](examples/13_compute_postprocess) |
| Shadows and deferred | [09_shadow_map](examples/09_shadow_map), [17_cascade_shadows](examples/17_cascade_shadows), [10_gbuffer_mrt](examples/10_gbuffer_mrt) |
| Cubemaps and layers | [14_skybox](examples/14_skybox), [16_env_capture](examples/16_env_capture) (six faces), [18_multiview](examples/18_multiview) |
| Image quality | [15_msaa](examples/15_msaa), [23_outline](examples/23_outline) (stencil) |
| Pipeline stages | [25_tessellation](examples/25_tessellation) (displacement and adaptive detail), [26_geometry_normals](examples/26_geometry_normals) (triangles become lines) |
| GPU-driven work | [28_gpu_culling](examples/28_gpu_culling) (indirect draw, two windows) |
| Data in and out | [24_video_texture](examples/24_video_texture) (per-frame updates), [22_instancing](examples/22_instancing) (20000 instances) |
| Windows and devices | [19_multi_window](examples/19_multi_window), [20_multi_context](examples/20_multi_context) (two GPUs), [21_window_modes](examples/21_window_modes), [08_pyqt_integration](examples/08_pyqt_integration) |
| Tools | [12_hot_reload](examples/12_hot_reload), [27_drop_and_icon](examples/27_drop_and_icon) (drag a picture onto the window) |

[CHANGELOG.md](CHANGELOG.md) lists what each release added. [DESIGN.md](DESIGN.md) gives the
reasons behind the API.

## License

MIT. See `LICENSE.txt`.
