# Changelog

All notable changes to **bazalt** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/) (pre-1.0: minor versions may break the API,
patch versions never do).

## [0.21.0] — 2026-07-30

"One binding, many textures". A quad that used a different texture from the one
beside it needed a different descriptor set, so 48 quads with 8 materials were
48 binds and 48 draws. A descriptor array puts all eight in ONE binding, each
quad carries the index of the one it wants, and the whole grid is one draw.

Getting there needed the feature table to grow. Bazalt named a GPU capability by
one boolean in `VkPhysicalDeviceFeatures`, and descriptor indexing does not live
there — nor do multiview and the GPU-decided draw count. The table now reaches
all three structs, which pays a debt that had been waiting since 0.18: multiview
had its own question, `ctx.supports_multiview()`, because there was no row for it
to sit in.

The rest of the release empties the small additive list before the 1.0 freeze: a
draw count the GPU decides, gamepads, and MSAA into images you already own.

### Added
- **Descriptor arrays: `texture(binding, stage, set, count=N)` and
  `set_image(binding, image, index=i)`.** One binding holding N textures, the
  shader picking one per draw or per fragment. `count=` is on all four binding
  declarators of both pipeline builders.

  An index that differs between invocations of one draw needs `nonuniformEXT(i)`
  and `#extension GL_EXT_nonuniform_qualifier : require` in the shader. Slots you
  never write are legal as long as nothing samples them, and a slot may be
  rewritten while an earlier frame still reads the set — swapping a texture at
  run time is the point.

  Needs `bz.Feature.BINDLESS`.
- **`bz.Feature.MULTIVIEW`, `bz.Feature.BINDLESS` and
  `bz.Feature.DRAW_INDIRECT_COUNT`.** Three capabilities the feature table could
  not name before.
- **A draw count the GPU decides: `count_buffer=` and `count_offset=` on
  `cmd.draw_indirect` and `cmd.draw_indexed_indirect`.** `count` becomes the
  maximum and the 4 bytes at `count_offset` say how many commands to issue, so a
  compute pass chooses the NUMBER of draws and not only their contents — one
  command per mesh, or per level of detail. The count buffer must be
  `BufferType.STORAGE`, like the arguments. Needs
  `bz.Feature.DRAW_INDIRECT_COUNT`.
- **`bz.get_gamepad(index=0, deadzone=0.0)`**, with `bz.GamepadButton` and
  `bz.GamepadAxis`. Returns a reading of that pad, or `None` when the slot is
  empty. A free function, because a pad belongs to the process and not to any one
  window — the same reason `bz.poll_events()` is one.
- **`update_after_bind=` on every binding declarator.** Whether that binding may
  be rewritten while a submit that uses the set is still in flight. The default
  answers per shape — on for an array, off for a single descriptor — because
  that is what each is for. Name it to override either: `True` on one texture you
  swap between frames, `False` on a static array you write once. Needs
  `bz.Feature.BINDLESS`.
- **`texture()` on the compute pipeline builder.** A compute shader could only
  reach an image as a storage image, so it had no filtering, no mip selection and
  no address mode. Nothing downstream was missing — `set_image`, the pool and the
  barrier tracker already handled a sampler binding on a compute set.
- **`samples=` on the `RenderTarget` signature that takes images.** MSAA into
  images you already own: bazalt renders into multisampled attachments and
  resolves into the images you passed, which stay single-sample and readable.
- **Examples `29_bindless`** (48 quads, 8 textures, one draw, and a key that
  swaps a texture while the loop runs) and **`30_gamepad`** (no shaders at all —
  the window colour follows the sticks).

### Changed (breaking)
- **`ctx.supports_multiview()` and `device.supports_multiview()` are gone.** Ask
  `ctx.supports(bz.Feature.MULTIVIEW)` and `device.supports(bz.Feature.MULTIVIEW)`
  instead. They were a second way to ask one question, and they existed only
  because the feature table could not reach the struct multiview lives in.

  Nothing else changes: multiview is still enabled by itself wherever the device
  has it, so `target.all_layers()` needs no opt-in.

### Changed
- **A gamepad reports what the hand did, not what the hardware sent.** A trigger
  reads 0 released and 1 pulled, where GLFW reports -1 to +1. A stick pushed UP or
  RIGHT reads positive, where GLFW's Y is screen space and reads positive downward.
  So `GamepadAxis.LEFT_Y` and `window.get_mouse_state().dy` point opposite ways on
  purpose: a mouse delta is a screen measurement and a stick is not.

### Fixed
- **A descriptor set kept every image ever written to it.** Writing one binding
  twice held the first image alive for the set's whole life, and made the
  record-time barrier pass walk a list that only grew. Invisible with one
  descriptor per binding; an unbounded leak for an array rewritten each frame.
- **A DYNAMIC vertex or index buffer had the wrong usage flags.** Only STORAGE
  was recognised; every other type got uniform-buffer usage, so
  `create_buffer(n, BufferType.VERTEX, MemoryUsage.DYNAMIC)` — geometry rebuilt
  each frame, which is what DYNAMIC is for — could not be bound and the draw read
  undefined data. The two memory usages computed their flags separately and had
  drifted; they now share one function.
- **`image.wait()` returned while updates were still queued.** The upload state
  was one flag, so the worker submitting the FIRST of several updates marked the
  image done — `wait()` returned and `read()` gave whichever frame had landed. With
  a queue deeper than the worker can drain it was measurably far off: update 71 of
  200. The state counts outstanding jobs now.
- **Two updates of one image could land backwards.** `image.update` promises that
  the call order is the GPU order, and one worker thread submitting in sequence
  does not deliver it: two submits on one queue may overlap unless one waits for
  the other. Each upload now waits for the previous one. A video decoder that
  queued two frames could show them in the wrong order, on drivers that take the
  freedom the spec gives them.
- **Example `29_bindless` drew nothing on the first attempt**, and a clean
  20-second run did not say so. The quad was wound the obvious way — right along
  the top first — which is back-facing under the default cull. Only a measured
  read-back caught it. The same trap `stripe.vert` already carried a comment
  about.

### Notes
- **Ask for BINDLESS.** `Context(optional=[bz.Feature.BINDLESS])`, then check
  `ctx.supports(...)`. `count>1` is refused without it. That is stricter than
  Vulkan, which allows a fixed-size array with a dynamically uniform index in
  core: without the feature an unwritten slot and a per-fragment index are both
  undefined, so the unguarded version works on one machine and returns garbage on
  the next.
- **Rewriting a descriptor mid-flight needs `update_after_bind=True` unless it is
  an array.** An array has it by default and a single descriptor does not, because
  a plain binding is written once at setup in nearly every program and the flag
  puts the descriptor in a separate limit budget. Without it, rewriting a
  descriptor a pending submit reads is undefined behaviour and the validation
  layers report it.
- **A gamepad reports level state only.** There is no `was_button_pressed`: the
  edge queries rotate on a per-window counter, and a pad has no window to hang
  one on.
- **`deadzone=` applies to the sticks, not the triggers.** A trigger rests at one
  end of its range, so a dead zone around zero would eat the first part of the
  pull.
- **MSAA on a borrowed target cannot combine with a mipped image.** A
  multisampled image has no mip chain, which is the rule the other signature
  already had.

## [0.20.0] — 2026-07-30

No new feature. This release splits the binding layer into eight translation
units, fixes six bugs and removes one API nobody meant to publish.

The library reached about 19,000 lines of C++ and all of it compiled as one
translation unit: `src/main.cpp` held every binding and included the whole
header-only core. One translation unit means one CPU core, and 0.14 had already
needed `/bigobj` because the object file passed a hard format limit. The
bindings now live in `src/bindings/`, one file per subject, and `main.cpp` is
the module definition plus seven calls. The core headers stay headers.

The bugs were found by reading the binding layer rather than by writing anything
new, and four of them crash or hang rather than misbehave. That is the argument
for the audit: the tests were green the whole time.

### Added
- **`tests/test_bounds.py`** — a near-maximum offset on each of the six verbs
  that take one.
- A test that scans every bound method for a missing parameter name, so the whole
  class of the keyword-argument bug is caught rather than the ten instances of it.
- A test that puts `read()` and `update()` side by side on the same mistake, which
  is what catches an exception type drifting apart again.
- A test that the bare enum names stay gone.

### Changed
- **The bindings are eight files.** `src/bindings/` holds `Enums.cpp`,
  `Resources.cpp`, `Pipelines.cpp`, `Commands.cpp`, `Windowing.cpp`,
  `ContextBind.cpp` and `Targets.cpp`, with `Common.hpp` for what they share and
  `Pch.hpp` as a precompiled header for the third-party includes. A new binding
  goes in the file that owns its subject. The call order in `main.cpp` carries
  the two registration constraints that are real and the one that only looks
  real.
- **A rebuild reuses its build directory.** scikit-build-core deletes its build
  tree by default, so every install reconfigured CMake and rebuilt glfw, volk and
  vk-bootstrap from scratch. This was the largest single change to build time in
  the release, it is two lines of configuration, and it has nothing to do with the
  split.

  The measured numbers, because the split is a trade and not a win everywhere:

  | | 0.19.0 | 0.20.0 |
  |---|---|---|
  | rebuild after editing one binding | 66.1s | **25.8s** |
  | rebuild after editing a core header | 66.1s | **43.9s** |
  | of which pip and CMake overhead | 10.2s | 13.1s |

  Against a same-configuration baseline, that is 36.6s before the split versus
  25.8s after for a binding edit, and 43.9s after for a core-header edit — 30%
  faster on the first and 20% **slower** on the second. Eight translation units
  each parse the header-only core, and one did before. The header parse is what
  dominates a rebuild, and making it cheaper means splitting the core into
  `.hpp` + `.cpp`, which is a rewrite this release does not attempt.
- **Sixty exception messages joined two clauses with a semicolon** and now read
  as two sentences, which is what the writing convention already asked for.
- **Six messages said too little to act on.** The image-upload failures named an
  internal object and discarded the Vulkan result code; `create_buffer` would
  not say which element type it found or how to state one; the render-target
  check named neither the wrong argument nor its value; two shader messages
  explained a binary format instead of naming the bad file; and the window
  failures named a C library the user has never heard of.

### Changed (breaking)
- **`.export_values()` is gone from all 15 enums that had it.** This is the break
  that makes the release 0.20.0 and not 0.19.1.

  The switch binds every enum member a second time as a bare module attribute,
  so `bz.ShaderStage.VERTEX` had a twin `bz.VERTEX`. About sixty such names
  existed: `NONE`, `ERROR`, `INFO`, `LINE`, `POINT`, `FILL`, `ALPHA`, `STATIC`,
  `DEVICE`, `WINDOW` and more. None of them appear in `_core.pyi` or in
  `__all__`, and nothing in the README, the tests or the 28 examples used them.

  Two collided, and the enum that bound last silently won: `bz.VERTEX` was
  `ShaderStage.VERTEX`, so `BufferType.VERTEX` had no bare spelling, and
  `bz.FLOAT` was `VertexFormat.FLOAT`, so `DataType.FLOAT` had none either.

  Use the qualified name. `bz.ShaderStage.VERTEX`, not `bz.VERTEX`.
- **Nine user errors raise `bz.ResourceError` where they raised `ValueError`.**
  The same mistake on the same object used to get two different answers:
  `image.read(layer=9)` raised `bz.ResourceError` and `image.update(pixels,
  layer=9)` raised `ValueError`. The C-contiguous rule reported the same sentence
  as `bz.ResourceError` from `Buffer.update` and as `ValueError` from
  `Image.update`, and the type stub only documented the first.

  The rule is now one line: **`ValueError` when the argument is wrong on its own,
  `bz.ResourceError` when a resource had to be consulted to know it was wrong.**
  So `image.update` raises `bz.ResourceError` for a layer, a mip or a region the
  image does not have, for a dtype or shape its format does not accept, for a
  strided array and for a multisampled image. `frames_in_flight` outside 1..4, an
  unknown `validation=` name and a `region=` that is not four numbers stay
  `ValueError`.

  Catch `(bz.ResourceError, ValueError)` if you were relying on the old type.

### Fixed
- **A `SwapchainRenderer` outlived the `Window` it was built from.** The
  renderer reads the window on every present, through lambdas that captured the
  raw window pointer and a pointer to the window's own resize flag. Nothing tied
  the two lifetimes, so `del window` left both pointers dangling and the next
  `present()` read freed memory.
- **`record_frame` raised a Python exception with the GIL released.** A failed
  `vkBeginCommandBuffer` or `vkEndCommandBuffer` took the interpreter down
  instead of raising `bz.DeviceLostError`. Both submit paths reach it without the
  GIL. It returns the error now, and `SwapchainRenderer.present` does too.
- **A frame that failed to record was never given back, so the next frame hung.**
  `acquire()` resets the frame slot's fence and only a submit signals it. Once
  `present()` reported a failed recording instead of crashing, it left that fence
  unsignalled, and `acquire()` waits on it without a timeout three frames later.
  The exception is now followed by the frame being released: the fence is
  signalled, the acquire semaphore is consumed and the swapchain is recreated.
- **A failed submit presented anyway.** `present()` logged a `vkQueueSubmit`
  failure and then called `vkQueuePresentKHR`, which waits on a render-finished
  semaphore that the failed submit was going to signal. It strands the same frame
  fence as well. The present is now skipped and the frame is given back, so the
  window logs the lost frame and carries on with the next one.
- **Six bounds checks could be bypassed by a large offset.** They were written
  `offset + length > size` on unsigned values, so an offset near the maximum
  wrapped the sum to a small number and passed. `buffer.update(data,
  offset=2**64 - 10)` reached a memory copy through an invalid pointer. The
  others are `image.update(region=)`, `image.read(layer=)`, `cmd.copy_buffer`,
  `cmd.fill_buffer` and the indirect draw verbs.
- **Ten methods refused the keyword arguments their own type stub declared.**
  `gb.cull_mode(mode=..., front_face=...)` raised `TypeError`, because the
  binding registered no parameter names. The affected methods are
  `vertex_shader`, `fragment_shader`, `tess_control_shader`,
  `tess_evaluation_shader`, `geometry_shader`, `vertex_format`, `cull_mode`,
  `topology` and `push_constant` on `GraphicsPipelineBuilder`, plus
  `ComputePipelineBuilder.shader`.

### Notes
- **Nothing about the split is visible from Python.** The module is the same
  module: every binding, every registration order and every annotation came
  across unchanged, and the type stub did not move.
- **A `ValueError` stays outside `BazaltError` on purpose.** `except
  bz.InitializationError` is the fall-back-to-headless handler and `except
  bz.BazaltError` is the catch-all, and a typo in a keyword argument must not be
  caught by either. Python already answers `ValueError` to "this argument is
  nonsense", and pybind raises it next door for a failed cast.
- **A lost frame is skipped, not fatal.** `acquire()` returning `False` and a
  failed submit both log the frame and let the loop carry on, which is the
  contract the whole windowed path follows.
- **The two frame-recovery fixes have no test.** Both need an out-of-memory result
  from a command-buffer call, which no test can provoke without an injected
  allocator. The windowed path was verified by running the examples, as usual.

## [0.19.0] — 2026-07-29

"The stages the pipeline could not run". `ShaderStage` had three values, so a
whole class of effect had no route at all: no displacement on terrain, no
adaptive detail, no normals drawn as lines, no grass grown from a point. This
release adds the two tessellation stages and the geometry stage, each behind its
own feature.

Three of those words need separating, because they sound like one thing.
Tessellation subdivides a patch with fixed-function hardware between two small
shaders. A geometry shader runs per primitive and may emit a **different** kind
of primitive, which is why tessellation does not replace it — "triangles in,
lines out" is outside what tessellation can say. Mesh shaders do replace both,
and they reach about half of the machines bazalt runs on, so they stay a future
feature rather than a baseline.

Second theme: the barriers. Bazalt now reads the SPIR-V to find out which
resources a shader writes. Until now a storage buffer bound to a graphics
pipeline was assumed read-only and a storage image bound to one was not tracked
at all, so a fragment shader that wrote an image needed a hand-written
`cmd.barrier()`. That was not a slow path, it was a wrong one.

Third theme: work the GPU decides for itself. A draw or a dispatch can read its
arguments out of a buffer, so a compute pass chooses what gets drawn and the
count never travels back to the CPU.

Fourth theme: the small things a prototype keeps needing. A render target on
images you already own, files dropped on the window, a window icon, the
clipboard.

### Added
- **Tessellation: `tess_control_shader()` and `tess_evaluation_shader()`.** With
  `topology(bz.Topology.PATCH_LIST)` and `patch_control_points(n)`. The vertex
  buffer holds flat patches and the surface is a function, so detail is chosen
  per frame from where the camera is. Needs `bz.Feature.TESSELLATION`.
- **Geometry: `geometry_shader()`.** One invocation per primitive, and it may
  emit another kind: a triangle becomes the lines of its normals, a point
  becomes a quad. Needs `bz.Feature.GEOMETRY_SHADER`. It is slow on modern
  hardware and absent on MoltenVK, so reach for it for a debug view rather than
  a hot path. Sending one draw into every layer of an array attachment does
  **not** need it — that is `target.all_layers()`.
- **`bz.RenderTarget(ctx, color=[image], depth=image)`.** Render into images from
  `create_image` instead of attachments the target allocates. This is what a
  graphics ping-pong between two textures needs, and drawing over a texture a
  compute pass baked, and drawing into an image carried from another Context.
  There is no size in that call: the images answer that question already.
- **`cmd.draw_indirect(buffer)`, `cmd.draw_indexed_indirect(buffer)` and
  `cmd.dispatch_indirect(buffer)`.** The arguments come out of a storage buffer,
  so a compute pass decides what to draw. Culling and particle compaction stop
  needing a readback between the pass that decides and the draw that obeys.
  bazalt declares no struct type for the arguments: the layout is
  `VkDrawIndirectCommand` and NumPy writes it. `count>1` needs
  `bz.Feature.MULTI_DRAW_INDIRECT`.
- **`Access.INDIRECT_READ`**, for a manual barrier against the command processor
  reading those arguments.
- **`window.dropped_files()`.** The paths dropped on the window during the last
  poll cycle, so dragging a picture onto a prototype is two lines. Like the key
  edges, it reads the same twice inside one frame.
- **`window.set_icon(rgba)`** takes a (height, width, 4) uint8 array, or `None`
  for the system default. **`window.set_cursor_position(x, y)`** moves the
  cursor, and the move never reads as the user moving it.
- **`bz.get_clipboard()` and `bz.set_clipboard(text)`.** Free functions, because
  the clipboard belongs to the process and not to any one window — the same
  reason `poll_events()` is one.
- **`shader.writes`, `shader.writes_unknown` and `shader.prints`.** What the
  SPIR-V says a shader does. `writes` is the list of `(set, binding)` pairs it
  writes, which is what decides the barriers, and it is readable so you can see
  why a barrier is or is not there.
- **`bz.Feature.FRAGMENT_STORES` and `bz.Feature.VERTEX_STAGE_STORES.`** Writing
  a storage buffer or image from a graphics shader. 0.17 added the declarator
  for it without these, so the write worked on the GPU and the pipeline failed
  to build.
- **Examples `25_tessellation`** (displaced terrain with detail chosen per
  patch), **`26_geometry_normals`** (normals as lines, from the mesh's own
  buffer), **`27_drop_and_icon`** and **`28_gpu_culling`** (two windows: one is
  the camera the culling is done for, the other flies around and shows you what
  is left).

### Changed
- **Bazalt reads the SPIR-V to decide barriers.** A storage buffer or image
  written by a graphics shader is now ordered against a later read with no
  `cmd.barrier()`. Compute loses barriers it never needed: two dispatches that
  only read the same buffer used to get one between them. Reads are always
  assumed and a write is only ever ruled out by proof, so an unusual shader is
  treated exactly as it was before.
- **A `Context(shader_printf=True)` compiles only the shaders that print without
  the optimizer.** Every shader in that Context used to pay for it.
- **`compile_shader` refuses a stage whose feature is off.** SPIR-V for
  tessellation and geometry names a capability the driver rejects, so the error
  arrives at the line you wrote instead of at the pipeline.
- **An HLSL `entry_point=` that matches no function is refused.** glslang
  answers a misspelled name with an empty shader that draws nothing, and that
  was an accepted limit until there was a way to see it.
- **`BufferType.STORAGE` carries the indirect usage flag**, so any storage buffer
  can hold draw arguments. A compute shader writing them needs a storage buffer
  anyway.

### Fixed
- **Error messages arrived with a broken character in them.** The sources are
  UTF-8 and the compiler was reading them as the system code page, so an em dash
  reached Python as one invalid byte. Eleven messages were affected, and one
  arrived empty.
- **Example `23_outline` drew six detached quads instead of a ring.** It grew
  the silhouette along the vertex normal, and a cube has one normal per face, so
  the faces moved apart and left the edges open.

### Notes
- **Tessellation and geometry are optional features, so ask for them.**
  `Context(features=[bz.Feature.TESSELLATION])` fails on a device without it;
  `optional=[...]` and `ctx.supports(...)` let a program carry on without.
- **A geometry shader is the wrong tool for a hot path.** Modern hardware runs
  it slowly and Metal has none at all. Its own trick — changing the kind of
  primitive — is what to use it for.
- **The draw count stays on the CPU.** A GPU-decided *count* needs an extension
  bazalt cannot reach yet. One draw command whose instance count a compute
  shader accumulates does the same job with no extension, and
  `examples/28_gpu_culling` is that shape.
- **`set_cursor_position` does not combine with `CURSOR_DISABLED`.** That mode
  already hands out unbounded motion and recentres itself, so warping every
  frame cancels every frame's movement. Pick one.
- **Automatic barriers are decided while you record.** A hot reload that makes a
  shader start writing a resource it only read does not re-barrier a recording
  that is only replayed. Record the frame again, which every example does.

## [0.18.0] — 2026-07-28

"Data, copies and tools". The boundary between your program and the GPU was
one-way and one-shot: pixels went in when an image was created, and came out
only from mip 0 of layer 0 of an offscreen target. A video frame, a camera
feed, a plot, a texture computed in NumPy and painting all meant one thing —
build a new `Image` every frame.

Around that sit two smaller gaps that the same work reaches. Copying was half
done: `copy_image` copied one mip level, a cross-Context transfer regenerated
the others, there was no scaled copy and no buffer copy at all. And an `Image`
held ONE layout for the whole thing, so a pass into `target.layer(0, 1)` claimed
mip 0 had reached the final layout too — the reason the notes used to say "render
every layer and every mip before you sample", which rules out the two things a
mip chain is for.

Third theme: debugging a prototype. A shader was debugged by rendering a value
as a colour and reading the pixel back.

Fourth theme, and the one that closes the release: which calls block.
`load_image(path)` returned at once, `create_image(array)` beside it waited for
the GPU, and no part of either name said so. Now every write is asynchronous
and every read blocks. That is one rule for the whole library.

Fifth theme, from an audit before the release: one path to one effect. Four
releases of additions had left several effects reachable two ways — five verbs
that waited, two names for reading a target back, two spellings of one render
target view. Each duplicate is now one verb. This is where the breaking changes
come from, and they are the last batch before 1.0 that touches these names.

`cmd.begin_rendering`/`end_rendering` and `cmd.begin_label`/`end_label` both
stay. They look like the same duplication and are not: a `with` block cannot
span a function boundary, so a recording assembled from helpers needs the
explicit pair. The block is still the form to reach for.

### Added
- **`img.update(array, layer=, mip=, region=)`.** Writes new pixels into an
  image that already exists. `region=(x, y, w, h)` writes one rectangle and
  keeps the rest, which is what painting and a sprite atlas need. The update
  runs on the same worker as `load_image`, so the call returns at once and the
  frame that samples the image waits for it on the GPU. Two updates of one
  image arrive in the order you made them — one worker, one queue, and that is
  a promise, not an accident. The array must match the format and be
  C-contiguous: a strided array would upload other bytes, so bazalt refuses it
  and names the fix.
- **`img.read(layer=, mip=)`.** Reads any face or level back, shaped to that
  level: mip 2 of a 64x64 texture returns 16x16. This is also the test
  `generate_mipmaps` never had — reading only mip 0 meant a mip generator that
  did nothing at all passed everything.
- **`ctx.load_image(bytes)`.** Decodes an image from memory, for a picture with
  no file behind it: one you downloaded, unpacked from a zip, or made with PIL.
  Everything after the decode matches the file path. There is no hot reload,
  because there is no file to watch.
- **`buffer.update(data, offset=)`.** Writes part of a buffer. Changing one
  matrix of an instance array used to rewrite the whole array.
- **`renderer.present(cmd, capture=True)` and `renderer.read_pixels()`.** A
  screenshot of a window. Two calls, and that is the design: Vulkan lets you
  touch a window image only between `acquire()` and `present()`, so reading the
  last frame is not legal and the copy travels with the frame instead. Bazalt
  asks for the copy capability only where the compositor allows it, and says so
  when it does not.
- **`cmd.blit_image(src, dst, filter=)`.** Copies between two images of
  DIFFERENT sizes and scales the pixels. `copy_image` needs a match, and
  `generate_mipmaps` scales only inside one image, so a bloom downsample, an
  upscale of a compute result and a thumbnail each used to need a full pass
  with a fullscreen shader.
- **`cmd.copy_buffer(src, dst)` and `cmd.fill_buffer(buffer, value)`.** Buffer
  copies on the GPU. Zeroing is the reason: a counter an atomic increments has
  to start each frame at a known value, and saying so used to take a dispatch
  whose whole body was an assignment. DYNAMIC buffers work here too.
- **`Context(shader_printf=True)`.** `debugPrintfEXT()` from your shaders
  arrives at the logger as `Severity.INFO` from `Source.SHADER`. Write
  `#extension GL_EXT_debug_printf : enable` and print the value you want. See
  **Notes** for the two costs.
- **`with cmd.label("shadow pass"):`.** Names a scope, so a RenderDoc capture
  reads as a frame and not as a list of draws. Bazalt has named objects since
  0.8; this names passes. Costs nothing when validation is off.
- **`cmd.occlusion_query()`.** Counts the fragments of the draws inside it that
  passed the depth and stencil tests. The handle is the identity, exactly like
  `cmd.timer()`. It must sit inside a rendering scope, because Vulkan requires
  the query to begin and end in one render pass.
- **`ctx.memory_stats()`.** The memory this Context holds, what it reserved and
  what the driver says is left, in bytes. "Am I leaking" used to need an
  external tool.
- **`ctx.subgroup_size`.** The width this GPU runs shader subgroups at. A
  compute reduction that uses `subgroupAdd` needs it to size its workgroup.
- **`ctx.submit(cmd, wait=False)` and `ctx.wait()`.** A headless submit that
  returns at once. With the wait, a compute prototype that submits in a loop
  leaves the GPU idle between iterations, so the loop runs at the speed of the
  round trip and not of the work. Reusing one command buffer is safe: the frame
  ring paces it.
- **`buffer.ready` and `buffer.wait()`.** The same pair an `Image` carries, for
  the same reason: a STATIC buffer fills itself in the background now. You do
  not have to ask. A submit that binds the buffer waits for it on the GPU, and
  `read()` waits for it on the CPU. These are for a loading screen and for
  timing a setup step.
- **Example `24_video_texture`.** A texture rewritten from NumPy every frame,
  painting into a region with the mouse, a labelled pass, and a screenshot.

### Changed
- **An image tracks its layout for each layer and level, not as a whole.** A
  pass now records only the part it drew, and `end_rendering` brings the rest of
  the image to the same final layout. So you can render into one mip or one cube
  face and then sample or read the image, which used to be a validation error
  some distance from the pass that caused it. Nothing changes for an image
  written whole: the layouts agree, bazalt stores one, and it records the same
  single barrier it always did.
- **`cmd.copy_image` copies every mip level the two images share.** It copied
  mip 0 and left the rest holding the destination's old pixels, which is a copy
  of the top level and not of the image.
- **A cross-Context transfer carries the source's own mip levels.** It used to
  regenerate them, so a chain you rendered per level arrived as a generated one.
  A prefiltered environment map is exactly that case.
- **DYNAMIC buffers carry the transfer usage flags**, so `copy_buffer` and
  `fill_buffer` work on them. The flags cost nothing on memory that the host can
  already see.
- **`ctx.create_buffer(...)` with STATIC memory no longer waits for the GPU.**
  It still submits the copy at the call, so a failure is still raised at the
  call. Only the wait is gone. That wait was a full drain of the graphics queue,
  so a program that made 30 meshes at startup drained the queue 30 times.
- **`ctx.create_image(array)` and `ctx.create_image([arrays])` no longer wait
  for the GPU**, for the same reason and with the same rule. `img.ready` is
  therefore `False` on the line after the call. That is not a problem to fix: a
  submit that samples the image waits for it, and `read()` waits for it.
- **`ctx.wait()` and `ctx.upload_progress` cover these new uploads too.** A
  copy with no file to decode joins the same batch as a `load_image`, already
  submitted, so one counter answers for both kinds.

### Changed (breaking)
- **`ctx.wait()` is the only wait verb.** It waits for everything this Context
  started, uploads and submits together. `ctx.wait_for_uploads()`,
  `ctx.wait_idle()` and `ctx.uploads_done` are removed: all three ended at the
  same timeline semaphore and answered the same question at a slightly
  different width. Replace every one of them with `ctx.wait()`. To wait for
  less, wait on the resource — `buf.wait()` and `img.wait()` stay.
  `ctx.upload_progress` stays and now counts every upload, so
  `while ctx.upload_progress < 1.0:` replaces `while not ctx.uploads_done:`.
- **`target.read_pixels()` is removed.** Use `target.color[0].read()`. It was
  the same call with the layer and mip choice taken away, and `img.read()` is
  the general reader. `renderer.read_pixels()` stays and keeps the name: a
  screenshot is a different operation, with the capture protocol behind it.
- **`target.mip(level)` is removed.** Use `target.layer(0, level)`, which is
  what it called.
- **`window.should_close()` is removed.** Use `not window.is_open()`.

### Notes
- **Shader printf has two costs, and that is why it is off by default.** The
  validation layers implement it, so you need them: `validation="off"` with
  `shader_printf=True` raises `InitializationError` instead of printing
  nothing, and `validation="auto"` on a machine with no layers warns. The layer
  also instruments every shader, and bazalt compiles that Context's shaders
  without the optimizer, because a print is a non-semantic instruction that the
  optimizer is allowed to delete.
- **Printf output reaches your callback whatever `min_severity` says.** The
  layer reports it at INFO and the default floor is Warning, so an obeyed
  filter would make the feature look broken. Bazalt also removes the layer's
  report text around your words, which for a print inside a loop is the
  difference between a tool and a wall of text.
- **An occlusion count is not exact.** A precise count needs the
  `occlusionQueryPrecise` feature, and without it the specification allows any
  value above zero. Read `q.samples > 0` as the reliable part.
- **`img.update()` is asynchronous.** Use `img.wait()`, `img.ready` or
  `ctx.wait()` when you need to know it has landed. A submit that
  samples the image waits for it either way.
- **A cross-Context transfer with a mip chain is slower now.** It reads and
  writes each level instead of regenerating them. The whole operation is
  already documented as a setup step that blocks the source queue, and correct
  data beats fast wrong data at setup time.
- **One rule for what blocks: every write is asynchronous, every read blocks.**
  A write returns a handle, so it can hand you the resource before the GPU has
  it, and the resource is its own future. A read returns an array, so it has
  nothing to hand you until the bytes arrive. One verb waits: `ctx.wait()`,
  for everything this Context started. `res.wait()` narrows it to one
  resource.
- **Tessellation, geometry shaders, indirect draw, a `RenderTarget` on images
  you own, and the window extras move to 0.19.**

## [0.17.0] — 2026-07-27

"What the pipeline still hard-codes". 0.16 removed the fixed blend mode, depth
write, polygon mode and load-op. Three fixed things were left: the **vertex
input** (one buffer, always advanced per vertex, three float formats), the
**stencil** (always off, and no format carried a stencil aspect), and **pipeline
creation** itself (no cache, no specialization constants, one blend state for
every attachment of an MRT target).

The two big ones are instancing and the stencil buffer. Instancing gives a
second vertex buffer that advances once per instance, so ten thousand objects
are one draw call with the mesh stored once. The stencil gives a pass a mask:
"draw only where something else was drawn", which is what an outline, a portal
and a decal are made of.

One break, and it is `draw_indexed_instanced`. See **Changed**.

### Added
- **`instance_format([...])` on the graphics pipeline.** Declares a second
  vertex binding whose attributes advance once per instance.
  `cmd.bind_vertex_buffer(buf, binding=1)` feeds it and
  `draw_indexed(36, instances=20000)` runs the mesh that many times. Locations
  continue after the vertex attributes, so `vertex_format` of three puts the
  first instance attribute at location 3.
- **`bind_vertex_buffer(buffer, binding=)`.** Which binding a buffer feeds.
- **`instances=` on `draw` and `draw_indexed`.** A non-indexed draw could not be
  instanced at all before this.
- **`VertexFormat.FLOAT`, `UBYTE4_NORM` and `UINT`.** `UBYTE4_NORM` is a colour
  or a set of weights in four bytes instead of the sixteen a `FLOAT4` takes, and
  it reads as 0..1 in the shader. `UINT` is an integer attribute, e.g. a
  material index.
- **`Topology.TRIANGLE_STRIP` and `LINE_STRIP`.** There is no restart index: one
  strip per draw.
- **`Format.DEPTH_STENCIL`.** A depth attachment with a stencil aspect. One name
  rather than two: the spec guarantees only that *one* of the two combined
  formats works on a given device, so bazalt picks. Attachment only — a combined
  texel has no numpy dtype, and `read()` says so.
- **`stencil_test(enable, compare=, ref=, pass_op=, fail_op=, depth_fail_op=,
  read_mask=, write_mask=)`.** The stencil test in one verb. It reuses the
  `CompareOp` the depth test and the compare samplers already use; only
  `StencilOp` is new. Front and back faces share the state.
- **`clear_stencil=` on `rendering` and `begin_rendering`.** The stencil clear
  value, exactly as `clear_depth` is the depth one.
- **`SwapchainRenderer(window, ctx, stencil=True)`.** Gives the window's depth
  buffer a stencil aspect, so a masked pass works on screen and not only into an
  offscreen target.
- **Specialization constants: `.constant(id, value, stage)` on the graphics
  builder and `.constant(id, value)` on the compute one.** A value baked into
  the SPIR-V when the pipeline is built, so one compiled shader serves several
  pipelines that differ by a number: quality levels, a kernel radius, a branch
  the driver can then delete. `bool`, `int` and `float`.
- **A pipeline cache.** One per Context, used by every `build()`. Nothing is
  written to disk: the blob's format belongs to the driver, and persisting it
  waits for 1.0 and a frozen API. Hot reload benefits most, because a rebuild
  differs from its predecessor by one shader.
- **`blend(..., attachment=)` and `color_mask(red, green, blue, alpha,
  attachment=)`.** A different blend state or write mask per colour attachment
  of an MRT target. Needs the new `Feature.INDEPENDENT_BLEND` when the
  attachments actually differ.
- **`storage_image(binding, stage, set)` on the GRAPHICS builder.** A fragment
  shader can now bind a storage image and `imageStore` into it. The automatic
  tracker still does not see those writes — that needs shader reflection — so
  pair it with `cmd.barrier(image, ...)`, exactly like an SSBO written from a
  graphics shader.
- **`depth_clamp(enable)`.** Clamps depth to the view volume instead of clipping
  the primitive, so a shadow caster between the light and the near plane still
  casts. Needs `Feature.DEPTH_CLAMP`, which was in the table with no API using
  it.
- **`alpha_to_coverage(enable)`.** Turns a fragment's alpha into an MSAA
  coverage mask: antialiased cutout foliage and hair with no sorting.
- **`cmd.copy_image(src, dst, src_access=)`.** Copies one image into another of
  the same size and format — the history buffer a temporal effect needs, or a
  compute ping-pong. `src_access` names where the source currently is, the same
  vocabulary `generate_mipmaps` uses.
- **`cmd.clear_image(image, color)`.** Fills a colour image with no pipeline and
  no pass. A depth image is refused: its clear belongs to the pass that renders
  into it.
- **`ctx.wait_idle()`.** Blocks until the device has finished everything.
  Releases the GIL while it waits.
- **`Format.R32_UINT` and `Format.R11G11B10F`.** An integer target is an id
  buffer, so mouse picking is `read_pixels` on one. `R11G11B10F` is HDR colour
  in four bytes instead of eight — a bloom or light-accumulation target at half
  the bandwidth of `RGBA16F`.
- **`AddressMode.CLAMP_TO_BORDER` with `create_sampler(border_color=)`, and
  `mip_lod_bias=`.** The standard shadow-map fix: past the edge of the map,
  `CLAMP` smears the edge texel over the whole scene, while a white border means
  "nothing occludes here".
- **Examples `22_instancing` and `23_outline`.** Twenty thousand cubes in one
  draw, and a two-pass stencil outline on a selected object.

### Changed (breaking)
- **`draw_indexed_instanced(n, k)` is gone; use `draw_indexed(n, instances=k)`.**
  A separate method name for one extra argument is the shape the design rules
  reject, and `draw` needed the same argument anyway. One line per call site.

### Fixed
- **A depth attachment that carries stencil is transitioned as one.** Every
  target used to name `DEPTH_ATTACHMENT_OPTIMAL`, which Vulkan forbids for an
  image with a stencil aspect. The layout and the aspect mask are now derived
  from the depth format, in one place, for the barriers, the views and the
  attachment infos alike.

### Notes
- **The pipeline cache is invisible.** It changes no result and needs no
  parameter. If it ever needs to be turned off, that is a Context flag, and
  nothing has asked for one.
- **Specializing a compute workgroup size needs Vulkan 1.3.**
  `layout(local_size_x_id = 0)` compiles to `OpExecutionMode LocalSizeId`, which
  requires `maintenance4`. The baseline is 1.2, so specialize the numbers the
  shader reads instead.
- **A `DEPTH_STENCIL` attachment cannot be sampled.** Its view carries both
  aspects and Vulkan forbids sampling through such a view, so bazalt does not
  ask for `SAMPLED` usage on it. A sampleable depth buffer stays `D32F`.
- **Tech debt #4 stays open, but it now closes itself.** The sync-validation
  negative test needs validation layer 1.4.350; LunarG's newest package for
  Ubuntu noble is still 1.4.313. CI used to declare the skip with a hard-coded
  environment variable, and now computes it from the installed package version.
  When a new enough layer is packaged, the test starts running with no change
  here — and the job says which version it found either way.

## [0.16.0] — 2026-07-27

"Window modes and input": a window switches between windowed, frameless and two
fullscreen modes while the application runs. The swapchain half of that needed no
new code. A mode change resizes the framebuffer, and the resize path already
recreates the swapchain, so the work was the window state and the monitor choice.

Two input gaps blocked the feature's own demo. There was no scroll wheel at all,
and `is_key_pressed` reports the level, so an F11 toggle needed a hand-written
"was it down last frame" flag. Both are fixed by one mechanism: `poll_events()`
counts its cycles, and every per-cycle query promotes what the callbacks
collected the first time it is read in a new cycle. One counter serves the key
edges, the mouse delta and the scroll.

Alongside them, four things the pipeline and the pass used to hard-code: the
load-op, the blend mode, the depth write flag with its compare op, and the
polygon mode.

One break, and it is the mouse delta. See **Changed**.

### Added
- **`WindowMode`.** `window.set_mode(bz.WindowMode.FULLSCREEN)` and
  `bz.Window(..., mode=)`. Four exclusive states: `WINDOWED`, `FRAMELESS`,
  `FULLSCREEN` and `FULLSCREEN_WINDOWED`. `WINDOWED` returns to the position and
  size the window had before it left that mode. Fullscreen takes the monitor the
  window covers most of, so a two-display setup gets the display the window is
  on. This is not exclusive fullscreen: the swapchain stays composited, because
  exclusive fullscreen needs `VK_EXT_full_screen_exclusive`.
- **Window attributes.** `set_size`, `set_position`, `set_resizable`,
  `set_always_on_top`, `set_opacity`, and the `position`, `resizable`,
  `always_on_top`, `opacity` and `content_scale` properties. `content_scale` is
  framebuffer pixels per screen coordinate, which is why `window.width` and the
  swapchain extent can disagree on a HiDPI display.
- **`window.was_key_pressed(key)` and `was_mouse_button_pressed(button)`.** The
  edge, where the existing queries give the level. Auto-repeat is not an edge.
  Reading twice inside one frame gives the same answer.
- **The scroll wheel.** `MouseState` gains `scroll_dx` and `scroll_dy`, plus `x`
  and `y` for the cursor position. There was no scroll callback before this.
- **`cmd.rendering(target, clear_color=None)`.** Preserves the colour and the
  depth instead of clearing them, which is what puts a second pass on one
  target: opaque, then transparent, then a UI. The first pass of a frame must
  still clear, because an acquired swapchain image starts undefined. Raises
  `ResourceError` on a multisampled target, which has nothing to preserve.
- **`renderer.set_present_mode(mode)`.** Switches vsync at runtime through the
  swapchain recreation that already existed. The mode stays a preference, so
  `present_mode` still reports what the driver gave you.
- **`blend(enable, mode=)`.** `BlendMode.ALPHA` is the previous behaviour and
  the default. `ADDITIVE` accumulates, which is what particles and glow need.
  `PREMULTIPLIED` composites a colour that already carries its alpha.
- **`depth_test(enable, write=, compare=)`.** `write=False` tests without
  writing, which a transparency pass needs. `compare=` replaces the
  `LESS_OR_EQUAL` that used to be fixed, and it reuses the `CompareOp` the
  compare samplers already had. `depth_test(False)` still writes nothing.
- **`polygon_mode(mode)`.** `PolygonMode.LINE` is the wireframe view. It needs
  the `WIREFRAME` feature (`fillModeNonSolid`), so ask for it with
  `Context(optional=[bz.Feature.WIREFRAME])` and `build()` says so if it is
  missing.
- **`clear_depth=` on `rendering` and `begin_rendering`.** The depth clear value,
  which used to be fixed at 1.0. That fixed value is also what made
  `depth_test(compare=GREATER)` useless, because nothing is ever greater than the
  far plane: reversed depth needs `clear_depth=0.0`. Ignored when the pass
  preserves.
- **`line_width(width)`.** For `PolygonMode.LINE` and `Topology.LINE_LIST`. A
  1-pixel wireframe nearly disappears on a HiDPI display. Anything other than
  1.0 needs the `WIDE_LINES` feature, because a driver may support exactly one
  width.
- **`depth_bias(constant, slope=0.0)`.** The fix for shadow acne. Note that a
  floating-point depth buffer scales `constant` by about 2^-24, so the small
  numbers from a D24 tutorial do nothing and a visible offset needs five or six
  digits.
- **`create_sampler(name=)` and `sampler.name`.** The sampler was the one object
  with no debug name. Because the cache shares one sampler between identical
  descriptions, names accumulate: two calls that differ only by name give one
  object named "a + b". Naming only the first caller would drop a name in
  silence, and putting the name in the cache key would let a debug label change
  what the program allocates.
- **`compile_shader(include_dirs=)`.** Extra directories for `#include`, tried in
  order and only when the name is not beside the including file. Adding a
  directory therefore cannot change what an existing shader includes.
- **`compile_shader(entry_point=)`.** Names an HLSL entry point, for one file
  that holds VSMain and PSMain. It is an error for GLSL, whose entry point must
  be main.
- **`compile_shader(source=)` takes SPIR-V bytes.** `str` is compiled as text and
  `bytes` is taken as ready SPIR-V: nothing is compiled, the extension of `path`
  stops mattering, and the words get the same magic-number and stage checks a
  `.spv` file gets. `ctx.compile_shader("v", stage, source=other.spirv)` is a
  round trip with no file involved.
- **Example `21_window_modes`.** Cycles every window mode, toggles wireframe and
  vsync, zooms with the wheel, and draws its bar in a second preserved pass with
  additive blending.

### Fixed
- **A pipeline stage names the entry point its module was compiled with.** It
  always said `main`, which is right for GLSL and for HLSL compiled the old way.
  With `entry_point=` the SPIR-V declares that name instead, and a stage asking
  for `main` fails to create the pipeline. Compute stages get the same rule, so
  an HLSL `CSMain` works too.

### Changed
- **`MouseState.dx` and `.dy` are the delta for the last poll cycle**, not a
  running total since the window opened. Every caller used to subtract the
  previous total to get a frame delta, and eight examples carried the same three
  lines to do it. Those lines are gone. If you kept a `last_mouse_dx`, delete it
  and read `mouse.dx`.
- **The depth attachment is always stored.** It used to be `DONT_CARE` unless
  the depth would be sampled later. That is cheaper right up until a second pass
  preserves it, where it makes the depth undefined the moment the first pass
  ends. The cost is depth bandwidth on a tiled GPU.

### Notes
- **The input rotation is driven by the reader, not by `poll_events()`.**
  `poll_events()` has no list of live windows, and giving it one means a global
  mutable list plus a lock for work a query can do itself. Consuming the state
  on read was the other option, and it answers `was_key_pressed` True then False
  inside one frame, which is a trap.
- **`set_present_mode` refuses to run between `acquire()` and `present()`**, and
  raises `ResourceError` naming the reason. Recreating the swapchain there would
  free the image the frame is holding.
- **The window tests need a display**, so CI skips them, the same as the
  multi-window tests. The load-op and the pipeline state are headless and run
  everywhere.
- **An HLSL entry point that matches no function is not an error.** glslang
  synthesizes an empty one under the requested name, so the compile succeeds and
  the shader draws nothing. Catching that needs SPIR-V reflection, which is tech
  debt #3. The behaviour is pinned by a test so a future glslang that does
  complain is noticed.
- **A shader carries its `include_dirs` and `entry_point`.** A hot reload only
  has the module, so a recompile that dropped them would resolve a different
  include or pick a different function, and the failure would look like a broken
  shader edit.

## [0.15.0] — 2026-07-26

"Multi-context / multi-GPU": any number of Contexts can be alive at once, on the
same GPU or on different ones. This pays off tech debt #1, the oldest entry in
the register. volk installs its Vulkan function pointers as **process globals**
and `volkLoadDevice` binds them to one `VkDevice`, so a second Context would have
silently redirected the first one's GPU calls at its own device — an access
violation with no diagnostic. bazalt had been refusing the second Context outright
since 0.5. Each Context now loads its entry points into a dispatch table of its
own (`volkLoadDeviceTable`), and the device-level globals are deliberately never
loaded at all, so a call site that skipped the table fails immediately instead of
landing on whichever device was created last.

Nothing breaks: this release removes a restriction and adds one overload.

### Added
- **N live Contexts.** `bz.Context()` twice no longer raises. Each owns its
  device, dispatch table, frame ring, upload worker and hot-reload watcher;
  nothing is shared. Combined with `Context(device=...)` from 0.14, that is
  bake-on-one-GPU/render-on-the-other, or a compute Context on the integrated
  chip beside a render Context on the discrete one.
- **`ctx.create_image(image)`.** A fourth overload of `create_image`, taking an
  Image from another Context (or this one — that is a clone). Carries size,
  format, array layers, cube-ness and whether the source was mipped;
  `other.create_image(img.read())` cannot, because `read()` returns mip 0 of layer
  0 as a bare array and a cubemap would arrive flattened. Blocking on the *source*
  Context and routed through host memory — without external memory there is no
  portable device-to-device path — so it is a setup step, not a per-frame one.
  The result is an ordinary async Image of the target Context.
- **Example `20_multi_context`.** Compute bakes a texture on the first device, the
  image crosses over, a window on the second device draws with it.

### Changed
- **Resources do not cross Contexts.** Passing a Context's image, buffer,
  pipeline, descriptor set or target to another Context's command buffer,
  descriptor set, pool or pipeline build raises `ResourceError` naming the
  operation. A mistake that was unreachable before is now one object away, and
  its unguarded symptom is a driver crash or a validation message that does not
  mention bazalt. Costs a pointer comparison at record time.
- **The Vulkan 1.2 + `VK_KHR_dynamic_rendering` path is tested.** CI now runs the
  full suite on lavapipe twice, once with the API version pinned to 1.2
  (`BAZALT_FORCE_VULKAN_1_2=1`, a test knob rather than public API). That path —
  where the dynamic-rendering entry points arrive under their KHR names and are
  aliased onto the core ones — had no coverage anywhere since 0.5, because both
  CI and the development GPU report 1.3 or newer. Tech debt #2, closed.

### Notes
- **Instance-level calls stay on volk's globals, on purpose.** `vkGetPhysicalDevice*`,
  the WSI queries and `vkSetDebugUtilsObjectNameEXT` are loader trampolines that
  dispatch on the handle passed to them, so one pointer is correct for every
  instance in the process. Only device-level dispatch was ever the problem.
- **Sync-validation tests run in-process again.** They used to execute a script in
  a subprocess purely because `validation="sync"` needs a Context of its own; two
  Contexts with different validation settings now coexist, which is also the
  sharpest available proof that the feature works.
- **Multi-context needs no second GPU to be exercised.** What broke was dispatch
  between two `VkDevice`s, which happens on one card just as readily, so
  `test_multi_context.py` runs everywhere including CI's lavapipe.

## [0.14.0] — 2026-07-26

"Multi-window": a Context can now drive any number of windows. Getting there
meant naming a thing the API had been conflating: the **frame** belongs to the
Context (its ring slot indexes the command buffers, dynamic buffers and
descriptor sets the Context allocates), while **acquiring and presenting a
swapchain image** belongs to a window. With one window the two coincided, so
`renderer.begin_frame()` could do both; with two, the second window advanced the
ring a second time and every `update()` landed in a slot nobody read. So the
frame moved onto `ctx.begin_frame()` and a window's verbs became `acquire()` and
`present()` — which is also the whole reason the one-renderer-per-Context guard
could go. Alongside it: `bz.list_devices()` and `Context(device=...)` make GPU
selection visible and overridable instead of automatic-and-silent, and
`FIFO_RELAXED` completes the present modes.

### Added
- **N windows on one Context.** `bz.SwapchainRenderer(window_b, ctx)` no longer
  raises. Each renderer owns its swapchain, surface, semaphores, fences, depth and
  MSAA images, so resizing or closing one window leaves the others rendering.
- **`ctx.begin_frame()`.** Opens one logical frame: advances the ring slot,
  applies pending hot reloads, reclaims deferred handles. Once per frame no matter
  how many windows draw into it. `ctx.frame_index` reports the slot.
- **`renderer.acquire()` / `renderer.present(cmd)`.** A window's two verbs.
  `acquire()` returns `False` when that window sits the frame out (minimized,
  mid-resize) without stopping the others.
- **`bz.list_devices()` and `Context(device=...)`.** Every GPU on the machine —
  `name`, `type`, `memory_mb`, `api_version`, `supports(Feature)`,
  `supports_multiview()` — before a Context exists, so the choice can be informed.
  Pass one back as `device=`; matching is by `deviceUUID`, never by name or index.
  `device=None` (the default) keeps the previous automatic selection, and an
  explicit device is still filtered by `features=`.
- **`PresentMode.FIFO_RELAXED`.** vsync that lets a late frame present immediately
  rather than waiting for the next interval. Falls back to FIFO like the others.
- **`renderer.gpu_time_ms`.** Was `frame.gpu_time_ms`; the timestamp pool is
  per-renderer, so with two windows there are two GPU frame times.
- **`bz.poll_events()`.** Was `window.poll_events()`. There is no such thing as
  polling one window: the OS message queue is per-thread and `glfwPollEvents`
  takes no window, so `window_a.poll_events()` read as A's events while pumping
  everyone's — and being a method forced a multi-window loop to keep a *closed*
  window alive just to have something to call it on, leaving a frozen window on
  screen until the app exited. The per-window distinction is unchanged and lives
  where it is real: `is_key_pressed`, `get_mouse_state`, `is_open` and
  `renderer.acquire()` each read one window's own state. Calling it with no
  window open raises `WindowError` instead of silently doing nothing (GLFW is
  only initialized while a Window exists).
- **Example `19_multi_window`.** Two windows, one Context, one pipeline, one mesh;
  different cameras, tints and present modes.

### Changed (breaking)
Every windowed loop gains one line and renames two calls. Headless code is
untouched — `ctx.submit()` still advances the ring itself.

| 0.13 | 0.14 |
|---|---|
| `if frame := renderer.begin_frame():` | `ctx.begin_frame()` then `if renderer.acquire():` |
| `frame.submit(cmd)` | `renderer.present(cmd)` |
| `frame.gpu_time_ms` | `renderer.gpu_time_ms` |
| `frame.frame_index` | `ctx.frame_index` |
| `bz.Frame` | removed — there is nothing left for it to be |
| `window.poll_events()` | `bz.poll_events()` |

```python
while window.is_open():
    bz.poll_events()
    ctx.begin_frame()
    if renderer.acquire():
        renderer.present(cmd)
```

The `Frame` object's guards survive as errors on the new verbs: acquiring twice
in one frame (in practice, a loop that forgot `ctx.begin_frame()`) and presenting
without an acquired image both raise `ResourceError`.

### Fixed
- **A failed submit could crash instead of raising.** Both submit paths run with
  the GIL released, where raising a Python exception is undefined; an upload that
  failed to become resident did exactly that. Errors now travel back as values and
  are raised by the caller.

### Notes
- **One CommandBuffer per window.** A `CommandBuffer` holds one command buffer per
  frame slot, and both windows now render on the same slot, so replaying one in two
  windows would overwrite work in flight. Doing so raises `ResourceError` with that
  sentence rather than a pending-state validation message (which a build without
  the validation layers would not print at all).
- **Multi-window is verified by hand.** Two swapchains need two real surfaces, so
  CI's lavapipe skips those tests; they run locally against a real driver, and the
  validation-as-assert fixture is the referee. The frame/window split itself is
  covered headless.
- **Still one Context per process** (tech debt #1) — that is the 0.15 feature.
  Multi-window deliberately needs no volk dispatch tables: N windows are one device.

## [0.13.0] — 2026-07-24

"Render-to-layer & Multiview": a graphics pass can now rasterize a scene into one
subresource — an array layer, a cubemap face, or a mip level — of a render
target, one at a time or (via multiview) into every layer in a single pass.
Compute could write layers/faces/mips since 0.10; this closes the gap for the
raster pipeline, which is what dynamic environment capture (real-time
reflections), cascade shadow maps, and render-to-mip are built on. A
`RenderTarget` grows layered/cube/mipped attachments (the same `layers=` /
`cube=` / `mip_levels=` kwargs `create_image` already had), and `target.layer(i)`
/ `target.mip(m)` hand back a lightweight view you render into with the existing
`cmd.rendering(...)`; `target.all_layers()` renders them all at once with the
shader keying off `gl_ViewIndex`. The whole thing infers from the target: the
attachment barriers narrow to the layer/mip the view covers, `renderArea`/viewport
follow the mip's size, and the multiview mask flows from the target into both the
pass and the pipeline — no new knob on the rendering verb.

### Added
- **`layers=` / `cube=` / `mip_levels=` on `RenderTarget`.** `bz.RenderTarget(ctx,
  w, h, color=..., depth=..., cube=True)` makes the colour attachment a cubemap
  (so `target.color[0]` samples as a `samplerCube`) with a matching 6-layer depth
  buffer; `layers=N` makes every attachment an N-layer array; `mip_levels=M`
  allocates the mip chain. Mirrors `create_image` exactly.
- **`target.layer(i, mip=0)` / `target.mip(m)`.** A lightweight `RenderTarget` view
  of one array layer / cube face and optionally one mip, passed straight to
  `cmd.rendering(...)`. `layer(i, mip=m)` selects both axes (e.g. a mipped cube for
  prefiltered/roughness-based reflections). Cube face `i` is layer `i`, Vulkan order
  `+X, -X, +Y, -Y, +Z, -Z`. The pass transitions and covers exactly that
  subresource — a `.mip(m)` pass renders at the mip's own size.
- **MSAA + layers.** `samples>1` now composes with `layers=` / `cube=` (only
  `mip_levels>1` is rejected — a multisampled image has no mip chain). The
  multisampled attachment is layered and resolves per layer, so
  `cmd.rendering(msaa_target.layer(i))` antialiases each layer — MSAA environment
  capture and antialiased cascade shadows.
- **`target.all_layers()` (multiview).** Renders into EVERY layer / cube face in ONE
  pass instead of a pass per layer; the shader selects per-layer work with
  `gl_ViewIndex` (e.g. a per-face matrix for cube capture). Needs a layered target
  and `ctx.supports_multiview()`; composes with MSAA (each view resolves into its
  own layer). Enabled when the GPU advertises the multiview feature.
- **`ctx.supports_multiview()`.** Whether one-pass `all_layers()` is available.
- **Examples `16_env_capture`, `17_cascade_shadows`, `18_multiview`.** Real-time
  cubemap reflection (six-pass and one-pass multiview variants) and a 3-cascade
  shadow array, driven by `target.layer(i)` / `target.all_layers()`.

### Fixed
- **STATIC uniform buffers.** `create_buffer(data, UNIFORM, STATIC)` was created
  without `VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT` (the STATIC path had no `UNIFORM`
  case), so it failed at bind time with a cryptic validation error. Constant data
  in a STATIC uniform buffer now works.

### Notes
- **No breaking changes** — additive. `layers`/`cube`/`mip_levels` default to the
  scalar case, and a target with none of them is byte-for-byte the previous
  behaviour (the new subresource defaults, `view_mask()==0`, and `viewMask==0` on
  the pipeline all reproduce what the code hardcoded before).
- **Deliberate ceilings.** MSAA does not compose with `mip_levels>1` (a multisampled
  image has no mip chain). The attachment `Image` holds one layout for the whole
  image, so with per-layer passes render every layer/mip you intend to sample before
  sampling a layered target (`all_layers()` renders them all at once, so it has no
  such caveat); sampling a partially rendered target is undefined and validation
  flags it.

## [0.12.0] — 2026-07-24

"MSAA": hardware multisampling, the last effect of the resource/renderer phase.
It lives entirely on the render target — `samples=N` on a `RenderTarget` or
`SwapchainRenderer` — which allocates a multisampled image and resolves it into
the single-sample attachments you already sample and present. A pipeline reads
the sample count off the target it builds against, so nothing else in a frame
changes. Depth resolves too (SAMPLE_ZERO), so `target.depth` stays sampleable
under MSAA. Rides along: per-attachment clears, `RenderTarget` debug names,
`ctx.max_samples()`, and a sample-rate-shading knob.

### Added
- **`samples=` on `RenderTarget` and `SwapchainRenderer`.** `bz.RenderTarget(ctx,
  w, h, color=..., depth=..., samples=4)` renders into a multisampled image and
  resolves into `target.color[i]` / `target.depth` (which stay single-sample and
  sampleable — depth resolves with `SAMPLE_ZERO`). `bz.SwapchainRenderer(window,
  ctx, samples=4)` does the same for a window, resolving into the swapchain image
  on present (its MSAA colour + depth are recreated with the swapchain on resize).
  Must be a power of two `<= ctx.max_samples()`. A pipeline built against an MSAA
  target picks up the count automatically — there is no pipeline `samples` knob.
- **`ctx.max_samples()`.** The highest sample count the GPU backs for both colour
  and depth — the valid ceiling for `samples=`.
- **Per-attachment clears.** `clear_color` on `begin_rendering` / `cmd.rendering`
  now accepts either a single `[r, g, b, a]` applied to every attachment (as
  before) or a per-attachment list `[[r,g,b,a], …]` for MRT.
- **`name=` on `RenderTarget`.** Labels the attachments in validation messages
  (the 0.8 debug-name mechanism, previously Buffer/Image/Pipeline only).
- **`builder.sample_shading(enable=True, min_fraction=1.0)`.** Per-sample fragment
  shading on an MSAA target (cleans up interior/specular aliasing plain MSAA
  leaves). Needs the `SAMPLE_RATE_SHADING` feature on the Context; `build()` raises
  `ShaderError` otherwise.
- **`Image.samples`.** The sample count of an image (1 for everything a
  RenderTarget hands out, since those are the resolved attachments).

### Notes
- **No breaking changes** — everything is additive (`samples` defaults to 1, and
  the single-clear form is unchanged). The non-MSAA path is byte-for-byte the same:
  every MSAA getter answers "1 / no resolve" at samples=1.
- **Deliberate ceilings.** The multisampled image keeps only attachment usage and
  is not `TRANSIENT`/lazily allocated (so MSAA costs its full memory — upgrade if
  it bites). A pipeline's sample count must match the target it draws into, exactly
  as its formats must — build against the target you render to.
- Raw multisampled images are not exposed to Python (they can't be sampled,
  uploaded to, or read back), so MSAA is reachable only through a render target.

## [0.11.0] — 2026-07-23

"Mipmaps": the mip chain is no longer hardcoded away for everything except
`load_image`. Numpy textures can opt into a full chain, file loads can opt out,
empty images can allocate levels, and `cmd.generate_mipmaps` fills a chain from
mip 0 for images written by compute or a render pass — the machinery that only
the file loader used since 0.5 is now the whole surface. Plus a fix to the
manual image barrier so it composes with the automatic tracker.

### Added
- **`mipmaps=` on the pixel/file image calls.** `ctx.create_image(array, *,
  mipmaps=True)` and `ctx.create_image([...], *, mipmaps=True)` generate the full
  mip chain for a numpy texture (default `False` — arrays are data and get no
  surprise filtering). `ctx.load_image(path, *, mipmaps=False)` / `load_image([...],
  *, mipmaps=False)` turn the chain off for files (default `True` — pictures are
  mipped, e.g. a UI sprite sampled 1:1 opts out).
- **`mip_levels=` on empty images.** `ctx.create_image(w, h, fmt, *, mip_levels=N)`
  allocates a mip chain (1..full chain for the size); the extra levels start empty,
  to be filled by writing mip 0 (compute / a render pass) and then
  `cmd.generate_mipmaps`.
- **`cmd.generate_mipmaps(image, *, src=Access.SHADER_READ)`.** Fills mip levels
  1..N by blitting mip 0 down the chain (every array layer / cube face at once),
  leaving each level sampleable in `SHADER_READ_ONLY`. `src` names mip 0's current
  layout in `cmd.barrier`'s vocabulary — `SHADER_READ` (an uploaded / already-baked
  image, the default) or `SHADER_WRITE` (mip 0 fresh from a compute `imageStore`);
  its scope doubles as the barrier waiting on that producer. Refused on a
  single-level image, a non-blittable format, or inside a rendering scope.

### Fixed
- **Manual `cmd.barrier(image, …)` now updates the automatic tracker.** Mixing it
  with automatic uses of the same image in one recording (a manual transition
  followed by an auto sample) is now safe: the tracker learns the post-barrier
  layout instead of re-transitioning from a stale one, which had produced a
  mismatched-`oldLayout` validation error and a redundant barrier. `generate_mipmaps`
  seeds the tracker the same way.

### Changed
- The C++ source is now formatted by a project `.clang-format` (Allman braces,
  120-column) — a one-time sweep, no behaviour change.

### Notes
- Arrays default to a single level and files default to a full chain — the "arrays
  are data, files are pictures" split from 0.5, now with an explicit override on
  both sides. Mip generation reuses the existing blit cascade (falling back to a
  single level when the format can't be blitted/linearly filtered), so a cubemap
  or texture array mips across all layers in one pass.
- **Out of scope (unchanged from 0.10):** rendering *into* a specific mip level
  or cubemap/array layer with the graphics pipeline (dynamic environment capture,
  render-to-mip) still needs per-subresource render-target views — it rides with
  MSAA / render-to-layer in a later release. In 0.11 an empty mipped image is
  filled by writing mip 0 (upload or compute) then `generate_mipmaps`.

## [0.10.0] — 2026-07-23

"Cubemaps and texture arrays": images can now have more than one layer. The
existing `ctx.create_image` and `ctx.load_image` grow a `cube=` flag and accept
a **list** of layers (numpy arrays or file paths) — a cubemap is six faces, a
texture array is N, one mechanism (`arrayLayers > 1`) with two views: `CUBE` for
sampling (`samplerCube`), `2D_ARRAY` for compute storage writes. Skyboxes,
environment maps and procedural cubemaps without leaving the
`create_image`/`load_image` surface, and no new function or type.

### Added
- **Layered images through the existing calls — no new names, no new types.**
  - `ctx.create_image(images, *, cube=False)` accepts a **list** of numpy arrays
    → a texture array (view `2D_ARRAY`); `cube=True` (exactly 6 square faces,
    order +X,-X,+Y,-Y,+Z,-Z) → a cubemap (view `CUBE`). A single array is still a
    2D image, unchanged.
  - `ctx.create_image(w, h, fmt, *, layers=N)` / `cube=True` makes an **empty**
    texture array / cubemap, to be filled by a compute storage image (a
    procedural skybox writes all six faces with `imageStore` to an
    `image2DArray`).
  - `ctx.load_image(paths, *, cube=False)` accepts a **list** of file paths → a
    layered image loaded from disk (async, sRGB + mips), array or cubemap.
- **`Image.array_layers` and `Image.is_cube`** report the layer count and whether
  an image is a cubemap (alongside `width`/`height`/`mip_levels`).
- **Manual image barrier: `cmd.barrier(image, src, dst)`** (the image counterpart
  of the buffer overload). The layout follows the access — `SHADER_WRITE` =
  `GENERAL`, `SHADER_READ` = `SHADER_READ_ONLY` — so the one case the automatic
  tracker can't reach, a compute-baked image sampled across *later* submits,
  becomes one call: bake it once, `cmd.barrier(img, SHADER_WRITE, SHADER_READ)`,
  then sample it every frame without regenerating. Covers every mip and layer.
- New example `14_skybox`: an empty cubemap is filled face-by-face by a compute
  shader **once**, made sampleable with `cmd.barrier`, and sampled as a
  `samplerCube` skybox with a per-pixel world-space ray.

### Changed
- Image upload, mip generation and the automatic layout barriers now cover every
  layer: one layered buffer→image copy, one blit per mip across all faces, and
  `layerCount = array_layers` in the tracker's image-memory barriers. A cubemap
  transitions `GENERAL → SHADER_READ_ONLY` across all six faces in a single
  barrier before it is sampled — checked by sync-validation-as-assert.

### Notes
- A cubemap carries two views over one `VkImage`: a `CUBE` view for sampling and
  a parallel `2D_ARRAY` view for storage writes (a `CUBE` view is illegal as a
  storage image). `set_image` binds the former, `set_storage_image` the latter;
  the caller never sees the distinction. `array_layers()` reads the same, and the
  automatic barriers cover all layers.
- `cube=` is the one disambiguator a layered image needs: six layers alone could
  be a cubemap or a six-slice array, and only the caller knows which. It sets the
  image's creation flag and view type — it is not a sampler setting (the sampler
  is identical for 2D, arrays and cubemaps).
- **Out of scope for now (documented, not a regression):** rendering *into*
  cubemap/array layers with the graphics pipeline (dynamic environment capture,
  cascade shadow maps) needs per-layer render-target views — a later release. In
  0.10 an empty layered image is filled by uploading pixels or by a compute
  storage image (baked once with `cmd.barrier`, then sampled every frame). The
  remaining 0.5 backlog (async headless submit, async `StaticBuffer`, Sampler
  debug names) stays deferred.

## [0.9.0] — 2026-07-22

"Storage Images": compute shaders can now write images, not just buffers.
`imageStore`/`imageLoad` to a storage image opens post-processing and procedural
image generation in compute, and the `ResourceTracker` — buffer-only since 0.6 —
learns image barriers, so the layout transitions those paths need are recorded
for you. Plus per-scope GPU timers that read back headless.

### Added
- **Storage images in compute.** `ComputePipelineBuilder.storage_image(binding)`
  declares a read/write image binding; `DescriptorSet.set_storage_image(binding,
  image)` binds one (no sampler — accessed by coordinate, in `GENERAL` layout).
  `create_descriptor_pool` grows a `storage_images=` count. `create_image` already
  gives every format its legal usages, so a storage image needs no special
  creation — `ctx.create_image(w, h, fmt)` is enough where the format supports it.
- **The tracker learns image barriers** (pays down the "storage images
  untracked" debt). Per-image state now carries a layout on top of the buffer
  hazard logic, so a use emits an image-memory barrier with the right transition:
  `UNDEFINED → GENERAL` before the first write, `GENERAL → GENERAL` (a
  read-after-write / write-after-write memory barrier) between dispatches, and
  `GENERAL → SHADER_READ_ONLY` before a graphics pipeline samples a
  compute-written image — hoisted before `begin_rendering`, since a pipeline
  barrier is illegal inside dynamic rendering. Uploaded textures the tracker
  never saw are left untouched, so ordinary texturing costs nothing. Every
  auto-barrier path is checked by sync-validation-as-assert in the test suite.
- **GPU timers.** `cmd.timer()` records a timestamp and returns a `Timer`
  handle; stop it with a `with` block or `t.stop()` and read the GPU wall-clock
  in milliseconds off `t.ms`. The handle is the identity — no names, no keys —
  so several, nested and overlapping timers all work, and there is no forced
  `with`. Unlike `frame.gpu_time_ms` this needs no window and no `begin_frame`:
  a blocking headless submit means `t.ms` is ready as soon as `ctx.submit()`
  returns (profiling a dispatch is the use case). A handle read after the
  command buffer was re-recorded reports `None` (its slots now belong to a
  different timer). Self-gating: the query pool exists only once a timer is used,
  so apps that don't time pay nothing, no Context flag. Best-effort: a device
  without timestamp support reports `None`.
- New example `13_compute_postprocess`: a compute shader generates an animated
  image into a storage image and a fullscreen pass samples it, timing the
  dispatch.

### Notes
- **`upload_progress` semantics are now final** (settled here after being
  deferred since 0.5). "Batch" means everything queued since uploads last fully
  drained: when all in-flight uploads finish, progress resets to `1.0` and the
  next `load_image` starts a fresh batch from `0`, so a second loading screen
  counts only its own images.
- **Ceilings (documented, not regressions):** storage-image contents are not
  carried between submits by the tracker — each replay re-establishes the image
  from `UNDEFINED` (a discard, legal from any layout), which is exactly right for
  a compute post-process that overwrites its output every frame. Storage images
  are compute-only for now; a fragment-shader `imageStore` would need reflection
  the tracker doesn't have, so it waits for a manual image barrier. Remaining 0.5
  backlog (async headless submit, async `StaticBuffer`, per-attachment clears)
  rides with the release that needs it.

## [0.8.0] — 2026-07-20

"Hot Reload": `bz.Context(logger, hot_reload=True)` watches the files bazalt
loaded — shaders (and their `#include`s) and images — and applies edits to the
running program. A changed shader recompiles and rebuilds its pipelines in
place; a changed image re-uploads into the same handle. A bad edit is logged
and the last good version keeps rendering, so a typo mid-session never takes the
application down. Plus two diagnostics: per-frame GPU timing and debug names.

### Added
- **Hot reload.** One kwarg, `hot_reload=True`, covers both kinds ("watch what
  you loaded"). A background thread polls file mtimes; changes are applied on
  the main thread at `begin_frame()` and at `ctx.submit()`.
  - **Shaders:** editing a `compile_shader(path)` file or any file it `#include`s
    recompiles it and rebuilds every pipeline built from it, in place — deferred
    command recordings pick up the new pipeline with zero re-recording. A compile
    or pipeline error logs a `ShaderError` (`Source.SHADER`) and the previous
    pipeline keeps rendering.
  - **Images:** re-saving a `load_image(path)` file re-uploads into the existing
    `VkImage` through the upload worker (mips regenerated), so descriptor sets
    need no rewrite. Same size and format only — a resize or a corrupt/undecodable
    file logs a warning (`Source.UPLOAD`) and keeps the old contents.
  - Models and buffers are out of scope by construction: bazalt never sees their
    file path (you load them yourself into `create_buffer`).
  - `BAZALT_HOT_RELOAD_POLL_MS` (default 250) tunes the poll interval — a test/CI
    knob; the API stays one kwarg.
- **`frame.gpu_time_ms`.** The GPU duration in milliseconds of the frame
  submitted `frames_in_flight` ago (a timestamp pair around each submit).
  Opt-in with `Context(gpu_timing=True)` — the timestamp pool reset and two
  writes ride in every frame's command buffer, and per-frame queries are not
  guaranteed free on every GPU, so a profiling diagnostic stays off by default:
  no query pool, no per-frame cost, `gpu_time_ms` is `None`. When on: `None`
  until the ring has cycled once, and on devices without timestamp support.
  Windowed only — a headless submit is a blocking wait-idle, where wall-clock
  time already is the GPU time.
- **Debug object names.** `name=` on `create_buffer` / `create_image` /
  `load_image`, and `.name()` on both pipeline builders, attach a
  `VK_EXT_debug_utils` object name so validation messages name the culprit. A
  no-op (zero cost) when validation — and therefore the extension — is off.

### Changed
- A built `Pipeline` now keeps its `ShaderModule`s alive (they are the
  rebuildable description hot reload swaps against). Previously the modules could
  be dropped right after `build()`.

### Notes
- Backlog 0.5 (async headless submit, async `StaticBuffer`, per-attachment
  clears, `upload_progress` semantics) is deferred once more — hot reload needs
  none of it; triage returns in 0.9.

## [0.7.0] — 2026-07-20

"Shader Toolbox": every way a shader can arrive is now one function.
`compile_shader(path, stage, source=...)` — the extension of `path`
decides the language (`.hlsl`) or format (`.spv`), `source=` compiles
a string with `path` as a virtual name. GLSL `#include` works and the
pulled-in files are recorded per module — the contract the 0.8 hot
reload watcher will consume. Plus two small additions: compare
samplers (hardware PCF) and a present-mode knob on the swapchain.

### Added
- **In-memory compilation.** `ctx.compile_shader("name.frag", stage,
  source=text)` — no file on disk; the virtual name still supplies the
  language, `ShaderError.path`, and the `#include` base directory.
- **GLSL `#include`.** Both `"..."` and `<...>` resolve relative to
  the directory of the *including* file, recursively. The files used
  are recorded in **`ShaderModule.includes`** (absolute, normalized).
  A missing include is a `ShaderError` — the compiler discovered it —
  while a missing top-level file stays a `ResourceError`.
- **Prebuilt `.spv` loading.** `compile_shader("shader.spv", stage)`
  loads the binary, checks the SPIR-V magic, and verifies `stage`
  against the module's `OpEntryPoint`s — binding a fragment binary as
  VERTEX is one readable error instead of a validation storm.
- **HLSL.** `.hlsl` extension selects the language; entry point is
  `main`, one file per stage. Use `[[vk::binding(n, set)]]` on
  resources — bare `register()` piles into one Vulkan binding space.
- **`ShaderModule.path` / `.includes` / `.spirv`** — the module knows
  what it was built from, and `.spirv` round-trips: write it to a file
  and `compile_shader("*.spv", stage)` loads it back.
- **Compare samplers.** `ctx.create_sampler(compare=bz.CompareOp.LESS)`
  → GLSL `sampler2DShadow`; reads return the comparison result and
  LINEAR filtering averages four results — free hardware PCF. New
  `bz.CompareOp` enum (full eight VkCompareOp values). Example
  `09_shadow_map` now uses it.
- **Present mode.** `bz.SwapchainRenderer(window, ctx,
  present_mode=bz.PresentMode.FIFO)` — `FIFO` (vsync), `MAILBOX`
  (default preference, uncapped, no tearing), `IMMEDIATE` (uncapped,
  for measurements). Unsupported modes fall back to FIFO with an Info
  log — never an error (FIFO is the only spec-guaranteed mode).
  `renderer.present_mode` reports the mode actually in use, and the
  preference is re-negotiated on every swapchain recreation.

### Changed
- **`ShaderError.path` now names the file the error is actually in** —
  with includes, that can be the included `.glsl`, not the top-level
  shader. `.line` is the line within that file. (Previously: always
  the top-level path.)

### Notes
- Backlog 0.5 (async headless submit, async `StaticBuffer`,
  per-attachment clears, `upload_progress` semantics) deferred again
  in full — nothing in this release needs it; triage returns in 0.8.

## [0.6.0] — 2026-07-19

"Compute": compute pipelines with automatic barriers. Compute is a
first-class citizen — the same deferred recording, the same chaining,
one new verb (`dispatch`) — and the barriers between a dispatch and
whatever consumes its output are computed for you, with a fully manual
mode when you want the wheel. First release whose GPU test suite runs
in CI (lavapipe).

### Added
- **Compute pipelines.** `ctx.compute_pipeline().shader(comp)
  .storage_buffer(0).push_constant(4).build()` — declarators take no
  `stage` argument (compute has exactly one stage) and `build()` takes
  no target (compute has no attachments). `bz.ShaderStage.COMPUTE`
  compiles `.comp` GLSL through the existing `compile_shader`.
- **`cmd.dispatch(gx, gy=1, gz=1)`** — chains like every other
  recording method and mixes freely with rendering scopes in one
  command buffer.
- **Automatic buffer barriers** (`Context(auto_barriers=True)`, the
  default). Hazards between recorded uses — dispatch → dispatch,
  dispatch → draw (descriptor read or vertex fetch), draw → dispatch —
  get their barriers computed at record time; deferred recording makes
  them valid for every replay, including replay-to-replay ordering.
  Barriers discovered inside a rendering scope are hoisted before it
  (`vkCmdPipelineBarrier` is illegal inside dynamic rendering).
  Attachment layout transitions stay automatic always — they are the
  RenderTarget contract, not resource barriers. Known limit: SSBO
  *writes* from graphics shaders are not tracked (no shader
  reflection); `cmd.barrier()` covers that by hand.
- **Manual mode**: `Context(auto_barriers=False)` or
  `ctx.create_command_buffer(auto_barriers=False)` per command buffer;
  `cmd.barrier(buffer, bz.Access.SHADER_WRITE, bz.Access.VERTEX_READ)`
  with the new **`bz.Access`** enum (SHADER_READ, SHADER_WRITE,
  VERTEX_READ, INDEX_READ, UNIFORM_READ).
- **`validation="sync"`** — validation "on" plus synchronization
  validation, the only mode that reports *missing* barriers. (On SDK
  1.4.350 this also enables the layer's shader-accesses tracking,
  without which descriptor hazards go unreported.)
- **`.topology(bz.Topology.POINT_LIST | LINE_LIST | TRIANGLE_LIST)`**
  on the graphics builder — the hardcoded triangle list is now just the
  default.
- STATIC STORAGE buffers carry VERTEX usage: a compute-written SSBO
  feeds `bind_vertex_buffer` directly.
- **CI runs the full GPU suite on lavapipe** (Mesa software rasterizer,
  ubuntu-24.04) on every push — the first CI leg that actually renders.
  Honest scope note: lavapipe there is Vulkan 1.3, so the 1.2+KHR-alias
  path remains untested.
- Example: `11_particles` — a compute shader integrates particles in a
  storage buffer that doubles as the vertex buffer, drawn as points.
- Tests: 116 → 134 (compute end-to-end on numpy asserts, barrier
  hazards under sync validation in subprocesses, README compute
  snippet).

### Changed (breaking)
- `ctx.pipeline_builder()` → **`ctx.graphics_pipeline()`**, class
  `PipelineBuilder` → **`GraphicsPipelineBuilder`** — the builder split
  is what lets compute declarators drop their `stage` arguments.

## [0.5.0] — 2026-07-19

"Images & Uploads": asynchronous texture streaming, the Texture →
Image + Sampler split, configurable render-target formats with MRT and
shadow-map (depth-only) targets, a runtime frame ring, and chainable
command recording. The largest release to date; API breaks are batched
here per the pre-1.0 policy.

### Added
- **Async image uploads.** `ctx.load_image(path)` returns immediately;
  the decode and GPU copy run on a background worker and every submit
  that samples the image waits for exactly that upload, GPU-side, via a
  context-wide timeline semaphore. `img.ready`, `img.wait()`,
  `ctx.wait_for_uploads()`, `ctx.uploads_done` and `ctx.upload_progress`
  (per-batch, 0.0–1.0) give explicit control — a loading screen needs no
  user-side threads and never serializes behind its own cargo.
- **`bz.Format`** pixel formats (RGBA8, RGBA8_SRGB, BGRA8, R8, RG8,
  R16F, RGBA16F, R32F, RGBA32F, D32F) with one table driving Vulkan
  formats, byte sizes and numpy dtypes.
- **`bz.Image`**: `ctx.create_image(w, h, format=)`,
  `ctx.create_image(numpy_array)` (shape+dtype pick the format; UNORM —
  arrays are data, files are pictures), `img.read()` → numpy with the
  format's dtype/shape, automatic mipmaps on `load_image`. No `usage=`
  parameter: every legal usage is enabled, driver-filtered.
- **`bz.Sampler`**, cached on the Context: `ctx.create_sampler(filter=,
  address_mode=, anisotropy=)` — identical descriptions return the
  identical object. `bz.Filter`, `bz.AddressMode`.
- **Configurable render targets**: `RenderTarget(ctx, w, h,
  color=None|Format|[Format, ...], depth=None|Format)`. MRT renders
  into every attachment in one pass; `color=None, depth=D32F` makes a
  shadow map, and a depth-only pipeline may omit the fragment shader.
  Attachments are ordinary Images (`target.color[i]`, `target.depth`) —
  render-to-texture and shadow sampling need no further API.
- **Chainable recording**: every `CommandBuffer` method returns the
  command buffer, so `cmd.begin_rendering(t).bind_pipeline(p).draw(3)`
  works; plus `with cmd.rendering(target, clear_color=...):` which
  records `end_rendering` on exit, exceptions included.
- **`Context(frames_in_flight=N)`** (1–4, default 2) replaces the
  compile-time constant.
- **`buffer.read(dtype)`** → 1-D numpy array (dtype mandatory — buffers
  carry no format). STATIC reads round-trip the GPU; DYNAMIC reads map
  the current frame's copy.
- Descriptor sets return to their pool when garbage-collected; pools
  are no longer one-way.
- Examples: `09_shadow_map` (two passes, one command buffer),
  `10_gbuffer_mrt` (RGBA16F + RGBA8 g-buffer with deferred composite).

### Changed (breaking)
- `Texture` → `Image` + `Sampler`; `load_texture` → `load_image`;
  `DescriptorSet.set_texture` → `set_image(binding, image,
  sampler=None)`.
- `begin_frame()` returns `Frame | None` instead of `bool`, and
  `renderer.submit(cmd)` moved to `frame.submit(cmd)`. A Frame
  submitted twice or held across ticks raises `ResourceError`.
- `RenderTarget(depth=True)` → `depth=bz.Format.D32F` (a bool raises
  with a migration hint).
- An offscreen target's depth attachment now ends every pass
  sampleable (`SHADER_READ_ONLY`); the swapchain's scratch depth is
  unchanged.

### Fixed
- Headless `ctx.submit()` never advanced the frame ring: DynamicBuffer
  slots and frame descriptor set copies beyond slot 0 were never
  exercised headlessly. The ring now advances after each headless
  submit (and at `begin_frame` in windowed mode), keeping `update()`
  and the submit that consumes it on the same slot.
- Dropping a resource whose only owner was a recorded command buffer
  freed GPU handles the previous in-flight frame could still be
  reading. All resource destruction now goes through a deletion queue
  keyed by the submission timeline.
- Per-texture `VkSampler` objects (pure waste) and their `maxLod = 0`
  (which would have clamped away every mip) — samplers are cached with
  `VK_LOD_CLAMP_NONE`.
- VMA was told API 1.3 even on the 1.2 fallback path.
- A script ending with its Context still in scope could die at
  interpreter shutdown ("could not acquire lock for stderr") — the
  logger's drain thread was calling into Python while the interpreter
  finalized. Drain threads are now joined via `atexit`, before teardown.

## [0.4.2] — 2026-07-18

A hotfix for fragment shaders that use `discard` on Vulkan 1.3.

### Fixed
- Fragment shaders using `discard` triggered a
  `vkCreateShaderModule` validation error (`SPIR-V Capability
  DemoteToHelperInvocation was declared…`) and a massive frame-rate
  collapse on Vulkan 1.3 devices (example 07 dropped from ~350 FPS to
  ~2 FPS). When shaders are compiled for SPIR-V 1.6, glslang translates
  `discard` into `OpDemoteToHelperInvocation`/`OpTerminateInvocation`,
  but the device was created without the matching
  `shaderDemoteToHelperInvocation`/`shaderTerminateInvocation` features.
  Both are mandatory in Vulkan 1.3 and are now enabled on the core-1.3
  path.
- `Context.api_version()` reported the raw device version instead of the
  negotiated one: a 1.3-capable GPU behind a 1.2 loader takes the
  1.2 + `VK_KHR_dynamic_rendering` path, yet shaders were still compiled
  targeting SPIR-V for Vulkan 1.3, which such a device can reject. The
  shader target now follows the version the device was actually created
  against.

## [0.4.1] — 2026-07-17

A source-quality release: bug fixes, refactoring, and C++23 adoption.
No public Python API changes (one behavioural fix: `read_pixels()` on a
never-rendered target now raises `ResourceError` instead of returning
undefined VRAM contents).

### Fixed
- `read_pixels()` could return discarded content: the internal "has been
  rendered to" flag was never set, so the readback barrier always declared
  the image layout as `UNDEFINED`, which permits the driver to drop the
  rendered pixels. The flag now flips when rendering work is actually
  submitted, and `read_pixels()` on a never-rendered target raises
  `ResourceError` instead of returning uninitialised VRAM.
- `Buffer.update` on a STATIC buffer, oversized updates, and binding a
  DYNAMIC buffer to a static `DescriptorSet` now raise `bz.ResourceError`
  instead of a bare `RuntimeError` that `except bz.BazaltError` missed.
- Every `VkResult` on the one-shot upload/readback path (allocate, submit,
  wait, map) is now checked; a device loss during an upload surfaces as
  `DeviceLostError` at the failing call instead of garbage later.
- Swapchain creation failures propagate the real `VkResult` instead of
  collapsing to a bare "Failed to create swapchain"; mid-frame failures
  log the `VkResult` name.
- `DescriptorSet.set_buffer`/`set_texture` validate the binding against
  the pipeline layout: a typo'd binding index or a buffer/texture
  mismatch raises `ResourceError` at the call site instead of being
  silently written (a nonexistent binding used to be *assumed* to be a
  uniform buffer) and diagnosed, at best, by the validation layers at
  submit time.

### Changed
- CI now builds `release/**` branches and smoke-tests every built wheel
  (`import bazalt` + stub consistency); the GPU test suite remains local.
- Internal C++23 modernisation: `std::format` for diagnostics,
  `std::ranges` for searches and folds, `constexpr` lookup tables,
  `std::span<const std::byte>` on the buffer-update path, and deducing
  `this` for the pipeline builder's chained setters. Building from source
  now requires GCC 14+ or MSVC 19.36+ (prebuilt wheels are unaffected).

### Added
- Behaviour-pinning tests for previously untested bindings: non-indexed
  `draw()`, `push_constants`, `DescriptorSet`/`DescriptorPool` end-to-end,
  `STORAGE` buffers, `blend()`, and `load_texture` sampling.
- This changelog.

## [0.4.0] — 2026-07-16

The "Foundations" release: three interdependent pillars, API breaks batched.

### Added
- **Unified error handling**: `bz.BazaltError` exception hierarchy
  (`InitializationError`, `DeviceLostError`, `OutOfMemoryError`,
  `ShaderError` with `.path`/`.line`, `WindowError`, `ResourceError`);
  structured `Logger`/`LogMessage` with `severity` and `source` as data;
  default stderr logger; GLFW diagnostics routed through `WindowError`.
- **Feature negotiation**: `Context(features=[...], optional=[...])`,
  `ctx.supports(Feature.X)`, `ctx.device_name`, `ctx.api_version`,
  `ctx.headless`; Vulkan 1.2 baseline with 1.3 preferred.
- **Headless rendering**: `bz.RenderTarget(ctx, w, h)` + `ctx.submit(cmd)` +
  `read_pixels()`; `SwapchainRenderer` is now just one `RenderTargetBase`.
- First pytest suite (56 tests) with validation-layers-as-assert fixture.

### Changed (breaking)
- `Logger.on_error` → `Logger.on_message` with structured `LogMessage`.
- `begin_rendering(target, ...)` — target is now required.
- Zero-argument `set_viewport()`/`set_scissor()` removed
  (`begin_rendering` emits full-target versions automatically).
- `Format` → `VertexFormat`.
- `push_constants` no longer takes a `stage` argument.
- `build(renderer)` → `build(target)`.
- `create_command_buffer` moved from renderer to `Context`.
- Non-contiguous numpy arrays now raise `ResourceError` instead of
  silently uploading garbage.
- `RuntimeError` replaced by the `BazaltError` hierarchy at the API boundary.

## [0.3.0] — 2026-07-15

### Added
- `MemoryUsage` enum: `STATIC` (GPU-local) and `DYNAMIC` (host-visible,
  multi-buffered, updatable per frame) for explicit control over resource
  memory strategy.
- Keyword arguments on all Python bindings plus `_core.pyi` stubs — IDE
  autocompletion and type hints.

### Changed
- Core Vulkan environment (`Context`) separated from presentation
  (`SwapchainRenderer`): the GPU can be initialised and shaders compiled
  without creating any window.
- Swapchain creation reimplemented on raw Vulkan calls instead of
  vk-bootstrap, so a window can be attached after `Context` creation and
  swapchain recreation is robust.

### Fixed
- Vulkan teardown crashes during Python garbage collection: resource
  objects (`Buffer`, `Texture`, `Pipeline`, …) now keep the `Context`
  alive via `std::shared_ptr`.

## [0.2.0] — 2026-07-13

### Added
- Native OS window integration via `win32_hwnd` (embedding in PyQt/PySide).

### Changed
- Renderer decoupled from GLFW windowing.
- `Logger` extracted into a standalone module.

## [0.1.0] — 2026-07-08

### Added
- `DescriptorPool` and `DescriptorSet` API for custom descriptor
  management.
- More examples and expanded README documentation.

## [0.0.1] — 2026-07-06

Initial preview release: window with event handling, threaded logging
system, and basic shader prototyping.
