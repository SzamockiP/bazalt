# Design notes

`CHANGELOG.md` records **what** changed in each release. This file records **why**, and it
is organized by subject, not by version. Git history and the changelog already order things
by time, so a second timeline here would only rot.

What lives here: the design rules, the scope test, the decisions that still bind the code
today, the technical debt register, the deferred and proposed work, and the things we
rejected together with the reason. A decision without its reason gets undone by a future
maintainer, so the reason is the point of the file.

---

## Design rules

These settle most arguments.

1. **One obvious way for one thing.** A new resource variant that differs by one parameter
   is a kwarg on the existing function, not a new name (`create_image` / `load_image` with
   `cube=` / `mipmaps=`).
2. **Never lower the ceiling.** Split the modules further. Always leave an escape hatch
   (`cmd.barrier()`, `raw_extensions`).
3. **Reach: run on more than 90% of machines.** A Vulkan 1.2 baseline plus feature
   negotiation. New capabilities are additive through `Feature`, never through a 2.0.
4. **Prototyping first.** Bazalt is a library for **prototyping**, so features that *make
   effects* come first (storage images, cubemaps, mipmaps, MSAA, render-to-layer).
   Technical and architectural plumbing waits until a real need forces it. This is the
   tie-breaker whenever two items look equally worthy.
5. **One minor release = one big feature plus small related additions.** Pay a debt off with
   the feature that really needs it, not in a separate "debt release". One session must be
   able to execute the plan without filling the context.
6. **An API break is acceptable before 1.0, but you batch the breaks into one release.**

Rules 4 and 5 have a corollary that the 0.9 to 0.15 sequence confirmed: **the most invasive
work goes last.** The volk dispatch-table pass touched every `vk*` call. Done early it would
have taxed every later feature, because each new call site would have to go through a fresh
dispatch layer. Done after the resource types settled, it was one mechanical pass over a
complete, frozen set of call sites.

---

## The scope test — what belongs in bazalt

**Two questions** draw the line, not the label "engine versus wrapper":
1. Does it need **painful glue at the Vulkan level**?
2. Does the **Python ecosystem already solve it**?

| Candidate | Vulkan glue? | Ecosystem? | Verdict |
|---|---|---|---|
| `load_model` | no (numpy) | yes (trimesh) | **out, firmly** — bazalt does not know the file path |
| ImGui backend | yes | no | **in scope, but a companion package** on the public primitives, not core |
| raw-handle interop / `external_memory` | yes | no | an escape hatch — proposed, ON REQUEST |
| audio / physics | no | yes | **out** — do not mix them into a graphics library |

---

## Decisions that still bind

Each entry names the release where it landed, so you can find the code and the changelog
entry. The release is a label, not the organizing axis.

### Context, frames and windows

- **The frame belongs to the Context, not to a renderer** (0.14, breaking on purpose). The
  ring slot indexes `CommandBuffer::command_buffers_`, `DynamicBuffer::buffers_` and the
  per-frame descriptor sets, and the **Context** pools allocate all of them. A renderer
  holds only its own fences and semaphores on the slot, so it has no right to advance it:
  with N windows it would advance the slot N times, and every `update()` would land in a
  slot that nobody reads. So `ctx.begin_frame()` advances the ring, drains hot reload and
  flushes the deletion queue, once per frame whatever the window count.
  `renderer.acquire() -> bool` returns False when this window sits the frame out and the
  others continue, and `renderer.present(cmd)` finishes it.

  The reason for choosing a breaking rename over a non-invasive rule matters more than the
  choice: *if the API does not express what happens underneath, you change the API, you do
  not wrap it in a clever rule*. A clever rule would have fixed implicit frame ownership in
  place, and that is exactly what someone decodes at 3 in the morning.

- **A window belongs to a Context** (`bz.Window(ctx, ...)`, 0.14). This is why multi-context
  (0.15) broke no signature: the ownership edge was already in the right direction, and the
  split cost zero rework.

- **`Frame` is gone, but its guards live on** (0.14). A double `acquire()` on the same
  serial (in practice a forgotten `ctx.begin_frame()`) and a `present()` with no image both
  raise `ResourceError`, from two fields on the renderer (`acquired_serial_`,
  `image_acquired_`). One check catches both errors, because they are the same error seen
  from two sides.

- **`poll_events()` is a free function, not a method** (0.14). `glfwPollEvents()` is per
  process, and a per-window pump is impossible: `glfwPollEvents(void)` takes no window and
  the message queue is per thread. The per-window distinction already lives in the queries
  (`is_key_pressed`, `get_mouse_state`, `is_open`, `acquire`), because each of them reads
  the state of its own window, and the pump is what updates that state. A call with no
  window raises `WindowError` instead of doing nothing silently.

- **`Device` is dead data** (0.14). `bz.list_devices()` builds a bare VkInstance,
  enumerates, destroys it and returns `list[Device]`. `Context(device=)` then matches on
  `deviceUUID` inside its own instance. `Device` holds neither `VkPhysicalDevice` nor
  `VkInstance`, so it has no affinity to any instance: it outlives any Context and cannot
  dangle. Instance ownership stays **invisible from Python**, so the inside of
  `list_devices()` can be rewritten without breaking a line of user code. The enumeration
  goes through `vkGetInstanceProcAddr`, not `volkLoadInstance`, because the second one would
  point the instance-level globals of a live Context at an instance that disappears a moment
  later.

### Dispatch and volk

- **Every device-level call goes through the per-Context table** (`ctx.vk()`, 0.15). 131
  call sites moved. Recorded lambdas read the table from `FrameContext::vk`.

- **`volkLoadInstanceOnly`, not `volkLoadInstance`** (0.15). This is the verification of the
  pass, not a detail. The compiler cannot catch a missed call site, because the globals
  still exist as symbols. `volkLoadInstanceOnly` loads only the instance-level functions, so
  every device-level global stays NULL and the first missed `vk.` crashes on a null pointer
  instead of silently using another Context's device. The test suite is the referee here,
  not the compiler.

- **The instance level stays global, and that is not a shortcut** (0.15).
  `vkGetPhysicalDevice*`, the WSI queries and `vkSetDebugUtilsObjectNameEXT` are loader
  trampolines that dispatch on the handle you pass them, so one pointer is correct for every
  instance in the process. `vkSetDebugUtilsObjectNameEXT` has a second reason: debug utils
  is an **instance** extension, so `vkGetInstanceProcAddr` is the sanctioned route, and
  `vkGetDeviceProcAddr` may legally return null for it.

- **The table reaches deferred lambdas through `FrameContext`, not through a capture**
  (0.15). About 35 `vkCmd*` calls sit in deferred lambdas, and `FrameContext` already goes
  into each of them, so one field covers all of them instead of 35 captures. It is a raw
  pointer to a Context field, and the Context outlives its recordings, so it keeps the rule
  "a deferred lambda holds nothing that indirectly holds the Context". The exception is
  about 15 `defer_destroy` lambdas that capture `vk = &context_->vk()` explicitly, because
  they receive no FrameContext. That is safe, because the Context destructor drains the
  queue.

- **`BAZALT_FORCE_VULKAN_1_2` forces the whole negotiation** (0.15). It is a test knob, not
  public API. Clearing `has_1_3` in `create_instance_` pulls the device choice,
  `enable_extension_if_present`, `negotiated_api_version_` and the shaderc environment
  version along with it. Zeroing the entry points after the table loads is wrong and
  crashes, because a device that never enabled `VK_KHR_dynamic_rendering` has no KHR
  pointers to substitute.

### Render targets and resources

- **Subresource rendering is kwargs plus a view handle, not a new verb** (0.13). `layers=` /
  `cube=` / `mip_levels=` on `RenderTarget` mirror `create_image` (rule 1).
  `target.layer(i, mip=)`, `target.mip(m)` and `target.all_layers()` return a light view
  object that goes straight into `cmd.rendering(...)`. The verb gets no knobs:
  `begin_rendering` **infers** the subresource, the multiview state and the viewport from
  the target.

- **The view and the barrier come from the same `(layer, mip)`** (0.13).
  `RenderTarget::Subresource` with `color_subresource()` and `depth_subresource()` defaults
  to the old hard-coded `{0,1,0,1}`, so existing targets do not change, and a per-mip scaled
  `extent()` covers renderArea, viewport and scissor with no extra code. Because both the
  view and the barrier read the same source, they cannot drift apart.

- **One layout for a whole image** (0.13, and it holds today). `on_rendering_recorded` marks
  the WHOLE image, because an `Image` holds one `layout_`. Render every layer and every mip
  before you sample. A partial render followed by a sample fires validation on purpose. The
  same model is why `read()` transitions `array_layers_` in both directions.

- **MSAA composes with layers and with multiview, but not with mips** (0.13). A layered
  multisampled attachment resolves per layer, and a multiview one resolves per view. Only
  `mip_levels>1` is forbidden with MSAA.

- **The pipeline infers sample count and formats from its target** (0.12). No parallel knobs
  on the builder. The target is the single source, so the two cannot disagree.

- **Cross-Context image transfer is the fourth overload of `create_image`** (0.15).
  `create_image` already had three overloads that differ by the type of the first argument
  (a size, a numpy array, a list of numpy arrays), so a fourth joins the existing pattern.
  The overload carries the format, `array_layers`, `cube` and "was it mipped", which is what
  a bare `read()` round trip loses: `read()` returns mip 0 of layer 0, so a cubemap would
  arrive flattened.

- **A resource stays with its Context** (0.15). Hand one to the wrong command buffer and you
  get a `ResourceError` that names the mistake, not a driver crash. The guard lives in
  `main.cpp`, not in the headers: `CommandBuffer` methods return `*this` for chaining and
  have no error channel, and this catches a **user error**, not a C++ invariant, so it
  belongs in the binding layer where the GIL is held and `raise_error` is legal. Every class
  got an `owner()` accessor (not `context()`, because `SwapchainRenderer` already has a
  `context()` that returns a `shared_ptr`).

- **`Image` is a future, not a `Future[Image]`** (0.5). The resource is ready to record with
  at once, and only the submit needs residency. `.wait()` and `ctx.wait_for_uploads()` are a
  separate verb, not a second version of `load_image`.

- **The handle is the identity** (0.9). No string keys on an API whose primitive is an index
  or a handle: `cmd.timer()` returns a `Timer` and you read `t.ms`.

### Errors and the pybind boundary

- **The exception type is the recoverability contract.** `ErrorCode` maps 1:1 onto a Python
  exception class through the `unwrap()` helpers. `ShaderError` must be catchable on its
  own, or hot reload is pointless.

- **Anything under a released GIL returns `std::expected`** (0.14). A pybind throw
  constructs a Python object, so it needs the GIL, and `raise_error` under
  `py::gil_scoped_release` is an access violation, not an exception. Both submit paths run
  without the GIL, so `unwrap` runs only after the GIL is back.

---

## Technical debt register

The numbering is historical and stays. Every entry means "do it when it really hurts", not a
permanent ceiling.

1. ✅ **One live Context per process** — PAID in 0.15. `volkLoadDeviceTable` plus a
   per-Context table, and the device-level globals stay unloaded on purpose.
2. ✅ **The 1.2 path was untested in CI** — PAID in 0.15. A second lavapipe job runs with
   `BAZALT_FORCE_VULKAN_1_2=1`, and the same knob works locally on any driver.
3. **ResourceTracker: no SPIR-V reflection** — the tracker does not read the bindings from
   SPIR-V, so SSBO and `imageStore` writes from **graphics** shaders stay untracked. The
   ceiling is a manual `cmd.barrier()`, which exists for buffers and images since 0.10.
   Auto-tracking arrives with reflection, and a storage image on the graphics builder
   arrives with it. Target: before 1.0.
4. **The sync-validation test is skipped in CI** — the LunarG layer for noble (1.4.313) does
   not report shader hazards. Version 1.4.350 does. Remove the skip after a newer package
   arrives. Target: 1.0, and it depends on the environment.

### Ceilings accepted on purpose

These are not debt to pay, they are limits we chose. Each one names its upgrade path.

- **`recreate_swapchain` takes `vkDeviceWaitIdle`** (0.14), so a resize of one window
  stutters the other for a moment. It is correct, it only stutters. Upgrade path: narrow it
  to one swapchain.
- **A cross-Context transfer regenerates the mip levels above 0** instead of copying them
  (0.15), so a hand-authored chain flattens into a generated one. Upgrade path: a per-mip
  copy.
- **A cross-Context transfer goes through host memory and blocks the source queue** (0.15).
  Without `external_memory` there is no portable alternative, so the documentation calls it
  a setup operation.
- **Per-subresource layout tracking does not exist in `Image`** (debt #3 territory). It
  touches the core and the gain is small, because it only removes the loud edge of a partial
  render.

---

## Deferred until needed

Additive work with no assigned release. Pull each one in when it really hurts.

- Async headless submit.
- An async `StaticBuffer`.
- Sampler debug names.
- A narrower `recreate_swapchain` (see the accepted ceilings above).
- A per-level mip copy on a cross-Context transfer.

---

## Proposed features

All of them are **additive**, so they break no API and can enter as small additions beside a
big feature. Each one needs a decision.

**Ranked first by rule 4, because they make effects:**

- **Blend modes.** `blend(enable)` is a bool hard-wired to alpha today (`SRC_ALPHA` and
  `ONE_MINUS_SRC_ALPHA`). The lack of additive and premultiplied blending hurts immediately
  with particles and glow. Rule 1 says a kwarg on the existing `blend()`, not a new method.
- **`depth_write` / `depth_compare`.** `depthWriteEnable` is glued to `depth_test`, and
  `depthCompareOp` is hard-coded to `LESS_OR_EQUAL`. `depth_test(True, write=False)` is a
  condition for a correct transparency pass.
- **Fragment-shader storage images.** Reachable today with a manual image barrier
  (`cmd.barrier`, since 0.10). Auto-tracking needs SPIR-V reflection, so this arrives with
  debt #3.

**Shader compilation:**

- **`include_dirs` for `#include`** — a search path as a second resolution rule, a kwarg,
  when somebody asks.
- **HLSL `entry_point=`** — files with several entry points (VSMain, PSMain). It would be
  the first compilation knob that exists for one language only.
- **In-memory SPIR-V bytes** — `source=` takes only text today. Load ready bytes when a use
  case appears.

**Performance:**

- **Pipeline cache**, in memory first. It goes to disk only at 1.0, when the frozen API
  gives a stable format.
- **Specialization constants.**
- **`VERTEX_INPUT_RATE_INSTANCE`** (instancing).

These three are small and independent, and they optimize a complete surface now that every
resource type and pipeline type exists.

**Escape hatches and integration:**

- **Raw-handle interop / `external_memory`** — raw access to the Vulkan handles
  (`VkDevice`, `VkImage`, `VkBuffer`, `VkCommandBuffer`) for C++ ImGui with no copy, CUDA
  interop, a video decoder and OpenXR. Rule 2, and it fits several libraries at once. YAGNI
  until a concrete integration asks for it.
- **ImGui integration** — a companion package or an example on the public primitives, not
  core. See the rejection below for why.

---

## Rejected, and why

Scope rejections and implementation alternatives we looked at and turned down. Each entry
says what we did instead.

**Out of scope:**

- **`load_model`** — no Vulkan glue and trimesh already solves it. Bazalt does not know
  about file paths for geometry.
- **Audio and physics** — no Vulkan glue, and the ecosystem covers them. Do not mix them
  into a graphics library.
- **ImGui inside core.** ImGui makes no pixels: every frame it emits `ImDrawData`, which is
  vertices, indices, draw commands, a scissor and an atlas id. Bazalt already has every
  primitive for that (`DynamicBuffer.update`, `graphics_pipeline`, an atlas through
  `create_image` and a sampler, `draw_indexed` with `set_scissor`, `push_constants`). The
  backend is about 90% glue over the public API plus input forwarding, so it belongs beside
  bazalt, not inside it.
- **A 2.0 for new hardware capabilities.** Ray tracing and mesh shaders become new entries
  in `Feature` (rule 3). Nothing about them needs a major break.

**Implementation alternatives:**

- **`volkLoadInstanceTable`** (0.15). It sets the *global* `vkGetDeviceProcAddr` from one
  instance anyway, and volk says so in its own comment, so an instance table would remove no
  global at all. It would add 18 call sites for nothing. We kept the instance level global.
- **A `source=` kwarg for the cross-Context transfer** (0.15). It would invalidate `width`,
  `height` and `format` in the same signature. We used a fourth `create_image` overload.
- **"`Device` holds its instance"** (0.14). A Device from instance A, given to a Context
  that wants instance B, would force a choice between adoption and re-enumeration, and it
  would duplicate `validation=` and `raw_extensions=` across two objects. We made `Device`
  dead data instead.
- **A clever ring rule instead of the frame/window split** (0.14): "advance the ring only if
  you already consumed the current serial". It worked, it had a zero diff, and it would have
  fixed implicit frame ownership in place. We took the breaking rename, and it cost one line
  per loop.
- **A per-window `poll_events`** (0.14). Impossible, not merely undesirable:
  `glfwPollEvents(void)` takes no window and the message queue is per thread.
- **`with cmd.compute()`** — a compute dispatch has no teardown, so a context manager would
  be a false symmetry. Chaining covers the aesthetics.
- **String keys anywhere the Vulkan primitive is an index or a handle** — return a handle
  and read the result off it (the 0.9 timers).
- **A separate "debt release"** — rule 5. A debt gets paid by the feature that really needs
  it.

---

## What 1.0 means

1.0 is a stability release. The content is the freeze, not new features.

- **Freeze the API.** Deprecation gets one minor release of warnings, and there are no
  silent removals.
- **Every public symbol from `_core.pyi` is touched by a test.** An unexercised binding is
  an unimplemented binding.
- Add the `KEY_*` and `MOUSE_*` constants to `__all__` in `bazalt/__init__.py`. They work
  through the star import today, but they are not in `__all__`.
- Performance: a pipeline cache on disk, and descriptor indexing.
- Indirect draw / GPU-driven work and multi-submit: **ship them, or defer them with an
  explicit note.** Do not leave the question open in 1.0. For a Python library, indirect
  draw is a niche, so do not inflate the sequence for its own sake.
- Close debt #3 and #4, or write them down as accepted ceilings.
- **New capabilities after 1.0 are additive** (rule 3).

---

## Lessons and traps

Lasting engineering conclusions, distilled from the retrospectives. Do not repeat them.

### API design

- **When one API concept glues two Vulkan concepts together, split the API. Do not wrap it
  in a clever rule.** `renderer.begin_frame()` did the frame (the Context) and the acquire
  (the window) at once, and with one window they overlapped.
- **A guard does not have to die with the class it protected.** `Frame` carried the
  generation guard, and after its removal two fields on the renderer catch the same error.
- **A new resource variant that differs by one parameter extends the existing function.**
  `cube=` must exist because the difference sits on the image (the `CUBE_COMPATIBLE` flag
  and the view type), not on the sampler.
- **Infer from the target, and put no new knobs on the builder.** Self-synchronization
  instead of a parallel API.
- **Do not add a `with` block or a context manager without a real "end".**
- **If a method does not touch `this`, it is not a method.** In the same way: if a free
  function needs state that every caller already has, take it as a parameter
  (`record_image_transition(vk, cmd, …)`), do not reach for a global.

### Mechanical refactors

- **Make sure the old road STOPS working.** A compiler cannot check the replacement of 131
  `vk*` calls with a dispatch table, because the globals still link. `volkLoadInstanceOnly`
  leaves the device-level globals NULL, so every miss crashes on the first run. General
  form: if a refactor moves call sites from A to B, delete A, do not keep it "just in case".
- **Take the definition of "what the pass covers" from a tool, not from memory.** The list
  of device-level functions came straight out of `volkGenLoadDeviceTable` in `volk.c`. A
  hand-written list would have missed `vkQueuePresentKHR` (device-level in spite of the KHR
  name) or caught `vkGetPhysicalDevice*` (instance-level).
- **A test knob that fakes a missing capability must fake the WHOLE negotiation.** The right
  point is the start of the negotiation, and the extension and the tool versions follow from
  it.
- **Do the pass that touches every call site last**, after the types it touches have
  settled. See the corollary under the design rules.

### Barriers, tracker and sync

- **`vkCmdPipelineBarrier` is illegal inside dynamic rendering** — an auto-barrier
  discovered inside the scope must be HOISTED before the `begin_rendering` lambda
  (`commands_.insert()`, and deferred recording makes this cheap).
- **A manual image barrier MUST update the tracker** (`note_image_layout`). Otherwise an
  auto-sample of the same image in one recording transitions from a stale `oldLayout`, which
  is a validation error plus a useless barrier. Buffers do not need this: they have no
  layout, and the worst result is a useless memory barrier.
- **`validation="sync"` without `syncval_shader_accesses_heuristic` is a placebo** (SDK
  1.4.350). The layer does not track descriptor accesses, so a missing barrier between two
  dispatches is not reported. Turn it on through `add_layer_setting`, gated on
  `is_extension_available`, because it needs `VK_EXT_layer_settings`. Layer 1.4.313 reports
  nothing at all.
- **Replay wrap-around** — one conservative `VkMemoryBarrier` at the start of every replay,
  and only when the recording writes a tracked buffer. Otherwise frame N+1 races the
  in-flight frame N.
- **The input layout of an image on every replay is UNDEFINED (discard).** A storage image
  is re-established every frame, and the tracker does not carry contents between submits.
- **The tracker keys on `Buffer*` and `Image*`** — object identity, not `VkBuffer`, because
  a DynamicBuffer has a per-frame handle.

### pybind, C++ and lifetime

- **`raise_error` under `py::gil_scoped_release` is an access violation, not an exception.**
  The bug sat in `require_uploads_resident` since 0.5 on a rare path, and the 0.14
  CommandBuffer guard put it on the path of an ordinary user mistake.
- **`main.cpp` needs `/bigobj`** (MSVC). Since 0.14 it passes the limit of 65,536 COFF
  sections, and every pybind lambda is a template instantiation, so this only grows.
- **Chaining bindings return a `shared_ptr` to self** — a `.def(&...)` returning
  `CommandBuffer&` would try a COPY under the automatic policy.
- **A deferred lambda captures the `shared_ptr` by value**, and that is the hot-reload
  mechanism, not a bug: a reloaded `VkPipeline` is picked up on the next replay with no
  re-recording. The bug was an immutable `Pipeline`, not the deferral.
- **`atexit` joins every C++ thread that calls Python, not a destructor.** The destructors of
  module objects run during interpreter finalization ("could not acquire lock for stderr").
- **`VkCommandPool` is externally synchronized in BOTH directions.** The worker recycles its
  own command buffers, and only the staging buffers go through the shared deletion queue
  (VMA is internally synchronized).
- **`VkQueue` needs external sync** — a `queue_mutex()` around every `vkQueueSubmit`,
  `vkQueuePresentKHR` and `vkQueueWaitIdle`, because the worker submits from another thread.
  Release the GIL before every blocking wait.
- **A deferred lambda holds nothing that indirectly holds the Context.** A descriptor set
  holds the pool, the pool holds the Context, and that is a permanent cycle. The pattern is
  a raw handle plus the FIFO order of a deletion queue keyed on the submit serial.
- **`std::unreachable()` after an exhaustive switch on a pybind enum is a trap.** pybind
  enums accept any int (`bz.ShaderStage(7)` passes), so the UB lands in the user's hands.
  Leave safe fallbacks. A `to_vk` with no `default` does catch a missing new variant at
  compile time.

### numpy, Windows and shaderc

- **`memcpy` ignores strides.** `arr.T` and `arr[::2]` upload garbage. Use
  `py::array::c_style | forcecast`, or check contiguity explicitly and raise with the hint
  `ascontiguousarray` instead of copying silently. Skip dimensions of size 1, because their
  stride is arbitrary.
- **`Buffer` holds a `DataType`**, so `bindIndexBuffer` chooses UINT16 or UINT32. Before
  that, UINT16 read as UINT32 gave half of the indices.
- **The macros from `<windows.h>`** arrive through volk with `VK_USE_PLATFORM_WIN32_KHR`.
  `max` and `min` catch qualified calls, so write `(std::ranges::max)(...)`. `ERROR`,
  `LoadImage`, `CreateSemaphore`, `near` and `far` collide with our names.
- **volk loads `vkCmdBeginRendering` only on 1.3.** On 1.2 only the `...KHR` name exists and
  the core symbol is null, so `alias_dynamic_rendering_entry_points()` substitutes them. The
  timeline semaphore is the same story: core in 1.2, but the feature bit needs an explicit
  enable.
- **The watcher needs a two-poll debounce.** Editors save in bursts (truncate and write, or
  rename and replace), and a single poll catches the file half-written. Compare with `!=`,
  not with `>`, because a copy-over moves mtime backwards. The tests write through
  `os.utime(mtime+2)`.
- **The shaderc includer**: every `shaderc_include_result` must live until `ReleaseInclude`,
  so use a heap Holder per result. Without locks this is correct only when one compilation
  means one includer per thread, and the watcher recompiles on the main thread.
- **A test for "an edit to an include changes the result" must change the SEMANTICS**, that
  is a different constant. `SetOptimizationLevel(performance)` would fold a cosmetic change
  into identical SPIR-V, and the test would lie green.

### Tests and CI

- **The `ctx` fixture asserts ONLY on `Source.VALIDATION` with `ERROR`.** Tests that cause
  errors on purpose also log, and the structural `source` field makes the distinction
  possible. Do not assert on WARNINGs: the loader reports other people's layers (OBS, Epic
  Overlay).
- **`Logger.flush()` must release the GIL.** Delivery is async, and without a flush
  validation-as-assert asserts on "it has not arrived yet".
- **The Context is session-scoped** (one per process in the fixture) and every resource holds
  its Context, so a per-test Context is a leak trap. A test that needs its own Context asks
  the `extra_context` factory, which applies the same referee.
- **On lavapipe, a deliberately invalid draw is NOT usable as a diagnostic.** Validation does
  not stop execution, and lavapipe segfaults before the Logger drains, so diagnostics for
  validation errors must be host-side.
- **Two swapchains need two surfaces**, so `test_multi_window.py` skips without a display and
  runs locally with validation as the referee. A small automated-test yield is a property of
  that feature, not neglect.

---

## Verification — open items

- **A golden image per example on lavapipe in CI**, plus the validation layers as an error
  in every test. This is a permanent regression guard and it is still incomplete, because
  the tests assert pixel values directly instead of comparing reference images.
- **Remove the sync-validation skip** when LunarG packages a newer layer for noble (debt #4).
- **The headless fallback** (no windowing extensions) still has no coverage. It is a separate
  path from the API version, which CI does cover since 0.15.
