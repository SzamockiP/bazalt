# Changelog

All notable changes to **bazalt** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/) (pre-1.0: minor versions may break the API,
patch versions never do).

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
