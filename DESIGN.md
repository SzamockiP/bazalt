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

- **A window mode is one enum and one verb** (0.16). `WindowMode` has four
  exclusive states (`WINDOWED`, `FRAMELESS`, `FULLSCREEN`, `FULLSCREEN_WINDOWED`) and
  `set_mode` applies any of them. A `set_fullscreen` plus a `set_decorated` would spell
  two of the four states twice and leave the caller to invent the rules for combining
  them. It is a verb rather than a settable property for two reasons: the class already
  spells mutation that way (`set_title`, `set_cursor_mode`), and this one can fail, so an
  assignment that raises `WindowError` would read worse than a call that does.

  `FULLSCREEN_WINDOWED` earns its place as a fourth value instead of being composed from
  `FRAMELESS` plus `set_size`: it costs about five lines, and composing it would force
  monitor geometry into the public API. `FULLSCREEN` is **not** exclusive fullscreen —
  see the accepted ceilings.

  The constructor takes `mode=` and then goes through `set_mode` itself, so opening
  fullscreen and switching to fullscreen cannot drift apart, and the requested
  width/height are always what `WINDOWED` returns to.

- **The swapchain learns about a mode change the same way it learns about a resize**
  (0.16). Every mode change resizes the framebuffer, `framebuffer_resize_callback` records
  it, and `present()` consumes the flag. This is why the release needed no Renderer
  change: the 0.14 resize path was already the general case.

- **Per-cycle input rotates on read, driven by the reader** (0.16). `poll_events()`
  increments one process-wide generation counter. Each Window keeps `pending_` (appended
  by the GLFW callbacks), `current_`, and the generation it last saw; any query with a
  stale generation promotes `pending_` to `current_`. One counter serves the key edges,
  the mouse delta and the scroll, because all three are the same problem: state that
  accumulates during a cycle and must read the same twice inside it.

  The callbacks must never rotate. One that did would mark the generation seen halfway
  through a cycle and hide every later event of the same cycle.

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

- **`clear_color=None` means preserve, and it is a third state** (0.16). The parameter is
  an `optional<vector<...>>`: `nullopt` preserves, an empty vector still clears to black,
  a filled one clears to those colours. They have to stay distinct all the way down,
  because `None` used to normalise to the same empty vector as `[]`. Colour and depth
  preserve together — splitting them would put two knobs on the verb for one question,
  and the multi-pass case wants both.

  Three things a naive version gets wrong, and all three are the actual work:

  1. **The entry barrier cannot use `UNDEFINED`.** That is correct precisely *because*
     the pass clears, and with `LOAD_OP_LOAD` it discards what is about to be loaded.
     Preserving takes `oldLayout` from `final_layout()` / `depth_final_layout()` — the
     layout `end_rendering` retires to — and a source stage covering both ways the image
     can have got there: written by an earlier pass, or sampled since.
  2. **The depth store-op became unconditional `STORE`.** It was `DONT_CARE` unless the
     depth would be consumed, which is cheaper and wrong the moment a second pass loads:
     the depth is undefined as soon as the first pass ends, so opaque-then-transparent
     would z-test against garbage. This is the one place the feature costs something for
     everybody.
  3. **MSAA plus preserve is rejected, not supported.** The multisampled image is
     transient and the result lives in the resolve image, which is not what the next pass
     renders into. The guard lives in `main.cpp` for the same reason as the
     cross-Context one: it catches a user error, and the recording methods chain.

  Not guarded, and documented instead: a recording whose *only* pass preserves reads an
  acquired swapchain image, whose contents are undefined at frame start. Same shape as
  the existing replay rule below.

- **`clear_depth` is the clear VALUE, not a second preserve switch** (0.16). Preserving is
  already one question answered by `clear_color=None` for colour and depth together, so
  `clear_depth` only says what a clearing pass clears to. It exists because the fixed 1.0 is
  what made `depth_test(compare=GREATER)` unusable: nothing is ever greater than the far
  plane, so reversed depth needs `clear_depth=0.0`. A knob that only makes sense together
  with another knob added in the same release is a sign the first one was incomplete.

- **A sampler debug name accumulates on the shared object** (0.16). The cache keys on the
  description, so `create_sampler(name="shadow")` and `create_sampler(name="terrain")` with
  the same filtering are one `VkSampler`, and the object ends up called "shadow + terrain".

  Both alternatives are worse, which is why this sat in the deferred list for six releases.
  Naming only the first caller drops a name silently, and it does so exactly when somebody
  is reading validation output. Putting the name in the cache key gives a named sampler its
  own handle, which means **turning debug names on changes what the program allocates** — a
  debug label must never do that. Listing every user is true, and `sampler.name` is readable
  precisely because no single caller can predict it.

- **A new pipeline knob is a kwarg on the verb that already owns the question** (0.16).
  `blend(True, mode=)`, `depth_test(enable, write=, compare=)` and `polygon_mode(mode)`.
  `depth_test` reuses the `CompareOp` the compare samplers introduced in 0.11 rather than
  declaring a second eight-value enum, which is what that enum's own comment predicted.
  `depthWriteEnable` stays gated on `depth_test` as well as on `write`: Vulkan permits
  writing depth with the test off, and `depth_test(False)` has always meant "this pass has
  nothing to do with depth".

- **The pipeline infers sample count and formats from its target** (0.12). No parallel knobs
  on the builder. The target is the single source, so the two cannot disagree.

- **Per-instance data is a second BINDING, so it is a second verb** (0.17).
  `instance_format([...])` sits beside `vertex_format([...])` rather than becoming a kwarg on
  it, because rule 1 is about variants of one thing and these are two different vertex
  bindings. Locations continue across the two (a `vertex_format` of three puts the first
  instance attribute at location 3), which is the same "location is the index in the list"
  rule the vertex side already had — a `mat4` is four `FLOAT4`s and takes four locations.

  The draw verbs went the other way: `draw_indexed_instanced` **disappeared** into
  `draw_indexed(..., instances=)`. A separate method name for one extra argument is exactly
  what rule 1 rejects, `draw` needed the same argument and had none, and pre-1.0 breaks are
  batched (rule 6). This is the release's only break, and it is one line per call site.

- **`Format.DEPTH_STENCIL` is one name resolved per device** (0.17). The spec guarantees only
  that ONE of `D24_UNORM_S8_UINT` / `D32_SFLOAT_S8_UINT` supports a depth-stencil attachment,
  so exposing both would make the caller guess which one their driver has — precisely the
  Vulkan trivia `Features.hpp` exists to hide (rule 3). The Context picks once, preferring the
  float variant where it exists: bazalt's plain depth format is `D32F`, and a `depth_bias`
  tuned against a float buffer means something else entirely on a 24-bit integer one, because
  the bias is scaled in units of the format.

  `format_info()` is `constexpr`, so the entry carries `VK_FORMAT_UNDEFINED` and every call
  site asks `ctx.vk_format(format)`. An `Image` reports its resolved `vk_format()` and its
  `aspect()`, and the RenderTarget, the views and the barriers all read those — the 0.13 rule
  that the view and the barrier come from one source, applied to the aspect.

- **A combined depth/stencil attachment is attachment-only, on purpose** (0.17). Its view
  carries two aspects, and Vulkan forbids sampling through such a view, so the image is
  created without `SAMPLED`/`STORAGE` usage and `read()` refuses with the reason. Keeping the
  usage would make every `DEPTH_STENCIL` view illegal at creation, which is a validation error
  at the target's constructor rather than at the sample that was never going to work. A
  sampleable depth buffer stays `D32F`.

- **The depth layout is derived from the depth format, never named** (0.17).
  `DEPTH_ATTACHMENT_OPTIMAL` covers the depth aspect alone and is illegal for an image that
  also carries stencil, so `depth_final_layout()` and every barrier read
  `has_stencil(depth_format())`. The default lives on the base `RenderTarget`, which is why a
  window with `stencil=True` needs no override — and the bug that made this necessary (a
  windowed stencil target transitioning to a depth-only layout) was caught by RUNNING example
  23, not by a headless test. The window path has no other referee.

- **The stencil test is one verb with eight kwargs, and front == back** (0.17). It is one
  question with several parts: eight separate builder calls would let half of them be set.
  `CompareOp` is reused for the third time (compare samplers, depth test, stencil), so only
  `StencilOp` is new. Separate front and back states are a rare case that would double eight
  parameters on one verb, so they share; the upgrade path is a `face=` kwarg, and it is
  additive.

- **`clear_stencil` is the clear VALUE**, like `clear_depth` before it (0.17). Depth and
  stencil load together — one image, one layout, one load-op — so a pass cannot preserve the
  depth and clear the stencil. `clear_color=None` still answers "what happens to the old
  contents" for all of them at once.

- **A per-attachment blend override is per FIELD, not per attachment** (0.17).
  `blend(attachment=1)` and `color_mask(attachment=1)` each set their own optional fields of
  one `BlendOverride`, so the two compose in either order. Resolving against the
  pipeline-wide state at call time would make the result depend on which line came first,
  which is a rule nobody remembers at 3am. Attachments that actually differ need
  `INDEPENDENT_BLEND`: "core Vulkan" and "needs no feature bit" are different claims, and
  this is the third time that has bitten (after `WIREFRAME` and `WIDE_LINES`).

- **A specialization constant belongs to the pipeline, not to the shader** (0.17). That is
  the point of it — one compiled module, several pipelines that differ by a number — so the
  values live in `GraphicsState` and every hot-reload rebuild re-applies them, following the
  0.16 rule that anything a build depends on and cannot re-derive has to be stored with the
  result. The binding tests `bool` BEFORE `int`, because a Python `bool` IS an `int` and
  `True` would otherwise be baked in as the integer 1 — the same shape as the `str`/`bytes`
  ordering in `compile_shader(source=)`.

- **The pipeline cache is invisible** (0.17). One `VkPipelineCache` per Context, passed to
  every `vkCreate*Pipelines`, with no parameter and no observable difference. A failure to
  create it is not an error: `VK_NULL_HANDLE` means "no cache", which is what the code did
  before. Writing it to disk needs a stable format to be worth anything, so that waits for
  1.0.

- **`copy_image` names where the source is, exactly as `generate_mipmaps` does** (0.17). The
  tracker treats an image's layout at the start of each replay as UNDEFINED (a discard), which
  is right for an image the recording overwrites and wrong for one it reads, so the caller
  says `Access.SHADER_READ` or `Access.SHADER_WRITE`. Both ends finish in `SHADER_READ_ONLY`
  and BOTH are marked on the `Image`, not just the destination: a source that still believed
  it was in `GENERAL` hands a stale `oldLayout` to the next `read()`.

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

### Shader compilation

- **`source=` is one parameter with two types** (0.16). A `str` is text to compile, `bytes`
  is ready SPIR-V. From the caller's side both answer the same question — "here is the
  content instead of the file" — and two mutually exclusive parameters in one signature is
  the shape the 0.15 cross-Context transfer already rejected. Internally it is a
  `std::variant`, and the file-backed `.spv` path and the bytes path share one validator, so
  words from memory get exactly the diagnostics a file gets.

  **The `bytes` test has to come first in the binding.** pybind converts both `str` and
  `bytes` to `std::string`, so an `optional<std::string>` parameter would take a SPIR-V blob
  and report a GLSL syntax error on binary garbage. The length check (a multiple of 4) lives
  in the binding rather than the compiler, because a `bytes` of the wrong length is the wrong
  argument, not a truncated binary.

- **`include_dirs` is a fallback, never a shadow** (0.16). The directory of the including
  file is still tried first, and the search path only answers names that are not there. The
  other order is what a C compiler does, and here it would mean that adding a directory can
  change which file an existing shader includes.

- **`entry_point` is HLSL-only, and says so** (0.16). One file holding VSMain and PSMain is
  ordinary HLSL; a GLSL entry point must be main. So a name on a GLSL compile is rejected
  with a sentence instead of being passed to shaderc for it to fail obscurely. This is the
  first compile knob that exists for one language, which was the reservation about it — the
  answer is that the knob names something only that language has.

- **A shader carries the settings a recompile cannot re-derive** (0.16). `include_dirs` and
  `entry_point` live on the `ShaderModule` next to `path` and `stage`, because the hot-reload
  watcher holds only the module. A recompile that dropped them would resolve a different
  include or compile a different function, and the failure would read as a broken shader
  edit. General form: anything a compile depends on that is not in the file has to be stored
  with the result.

### Streaming, copies and layouts

- **An `Image` tracks its layout per `(layer, mip)`, with a collapse** (0.18). Until then
  it held ONE layout, which was right while every write covered the whole image and
  stopped being right the moment `target.mip(m)` existed (0.13): the whole image was
  marked as rendered, so the next barrier handed the driver an `oldLayout` that was true
  of one level and a lie about the rest. This file called that an accepted ceiling and
  told the caller to render every layer and every mip before sampling — which rules out
  the two things a mip chain is for, a render into one level and a read of another.

  The collapse is what makes it free. Subresources that agree store one value, and a
  split image collapses back the moment the last one catches up, so six cube faces
  rendered one at a time end uniform and the sample that follows costs exactly what it
  did before. Nothing is allocated until something writes a strict subset.

  Three parts, and the third is the one that turns the state into a fix. A pass marks
  only what it wrote (a `SubresourceTarget` and a `MultiviewTarget` name THEIR
  subresource, so the statement is always made by whoever knows what was drawn), and
  then `end_rendering` **evens the image out**: subresources the pass did not write are
  brought to the same final layout. "The layout the result ends in" was always a promise
  about the IMAGE, and a subresource pass used to honour it only for the part it drew.
  Transitioning an untouched subresource out of `UNDEFINED` discards contents that were
  undefined anyway, and a whole-image pass records no extra barrier at all.

  Verified the other way round: four of the six new tests fail on the previous commit
  with `expects VkImage (mipLevel = 2, arrayLayer = 0) to be in layout
  SHADER_READ_ONLY_OPTIMAL -- instead, current layout is UNDEFINED`.

- **`image.update` is asynchronous, and the FIFO order is a promise** (0.18). The case it
  exists for is a video frame at 60 fps, where a blocking update spends the frame budget
  on a memcpy. The cost of async is that "two updates of one image in one frame" needs an
  answer, and the answer is the implementation made explicit: one worker, one queue, so
  the call order IS the GPU order. Writing it down is what stops a future thread pool
  from silently breaking a video decoder.

- **Every user error in `update` is decided in the binding, on the main thread** (0.18).
  The worker cannot raise — it holds no GIL, by the invariant `UploadManager` was built
  on — so a bad dtype, a wrong shape or a strided array has to be caught before the job
  is queued. This is the same rule `create_image` has followed since 0.4, applied to a
  path that now has a thread in the middle of it.

- **`Image`'s layout state belongs to the main thread** (0.18). `set_upload_submitted`
  used to call `mark_has_contents` from the worker. That was a benign word write while
  the layout was one `VkImageLayout`; with a vector behind it, it both races the reader
  and is WRONG for a partial update, because it would claim every layer sampleable when
  one was written. Whoever queues the job records the layout instead. General form: when
  a field grows from a word into a container, re-read every thread that touches it.

- **An update marks the image pending BEFORE queueing** (0.18). Without it, an image
  created synchronously has upload state `None`, so `img.wait()` returns at once and a
  `read()` straight after an update returns the PREVIOUS contents — one update behind,
  every time. A wrong answer, not a slow one, and invisible in a test that only checks
  the final frame.

- **`copy_image` copies every shared mip level** (0.18, replacing the 0.17 ceiling). A
  copy that leaves levels 1..N holding the destination's old pixels is a copy of the
  image's top level, not of the image, and the difference appears the moment anything
  samples with a mip bias. N regions in the same one call. The estimate that made it a
  ceiling ("a case that has not come up") was right until `image.update(mip=)` made
  per-level content ordinary.

- **A cross-Context transfer carries the source's own levels** (0.18, replacing the 0.15
  ceiling). Regenerating them is a silent, plausible wrong answer for anyone who rendered
  their own — a roughness-prefiltered environment map is exactly that. It costs one
  readback and one update per `(layer, mip)`, which is slow; the overload was already
  documented as a setup step that blocks the source queue, and correct data beats fast
  wrong data at setup time.

- **`blit_image` is a second verb, not a kwarg on `copy_image`** (0.18). Rule 1 covers
  variants of one thing, and these are two different Vulkan commands with two different
  contracts: a copy moves texels and requires a match, a blit resamples and requires
  format features. Folding them would make `copy_image(scale=True)` fail for reasons the
  copy never had. `filter=` reuses `bz.Filter` for the third time rather than declaring a
  second two-value enum.

- **`fill_buffer` is 32-bit because `vkCmdFillBuffer` is** (0.18). The offset and the size
  are multiples of 4 and the value is one word repeated. Exposing a byte-wise fill would
  mean emulating it with a dispatch, which is the thing this exists to remove.

- **DYNAMIC buffers carry the transfer usage bits** (0.18). The alternative — refusing
  `copy_buffer`/`fill_buffer` on them — makes "which buffers can the GPU write into" a
  second rule to remember, for no gain: transfer usage costs nothing on memory that is
  already host-visible.

- **A window screenshot takes TWO calls, and that is the design** (0.18). A presentable
  image may only be touched between `vkAcquireNextImageKHR` and `vkQueuePresentKHR`, so
  "read the last frame" is illegal by the spec — the first implementation did exactly
  that and the validation-as-assert fixture caught it with `performs a layout transition
  on presentable VkImage ... but the image has not been acquired`. So
  `present(cmd, capture=True)` records the copy into the frame's OWN submit and
  `read_pixels()` collects it. The frame that asks pays for a full-frame copy; every
  other frame pays nothing.

  `TRANSFER_SRC` is requested only where `supportedUsageFlags` allows it. A compositor
  may refuse, and a swapchain that fails to create takes the window down with it, so the
  capability is recorded and `read_pixels` reports it instead.

- **The channel order is normalised, the extent is not** (0.18). Most compositors hand out
  BGRA, and `out[y, x, 0]` must mean red on every machine, so the swap happens in bazalt.
  The shape comes from the CAPTURE extent rather than the current one, because the window
  may have been resized since.

- **`load_image(bytes)` is an overload declared BEFORE the path one** (0.18). pybind
  converts `str` AND `bytes` to `std::string`, so the path overload would take a PNG and
  report it as a missing file. The same trap `compile_shader(source=)` hit in 0.16, and
  the third time this codebase has had to order overloads by Python type.

### Diagnostics

- **`shader_printf` is a separate switch, not a fifth `ValidationMode`** (0.18). The modes
  are exclusive states of "how hard do the layers check"; printf composes with all of
  them, and you want it *together with* `validation="sync"`, not instead of it. It is off
  by default for two costs: the layer instruments every shader, and every shader in that
  Context is compiled unoptimized.

- **A print is a non-semantic instruction, so the optimizer may delete it** (0.18).
  `debugPrintfEXT` compiles to `OpExtInst` against `NonSemantic.DebugPrintf` and has no
  observable result, so `-O` removes it and the shader silently stops printing. This is
  the one place a debugging switch changes what is compiled, and it is unavoidable.

- **`shader_printf=True` with `validation="off"` is an error, not a silent fix** (0.18).
  The two contradict each other; turning validation on behind the caller's back would be
  a Context that is not the one they asked for. `validation="auto"` on a machine with no
  layers warns instead, because auto exists precisely to keep running.

- **Printf output bypasses `min_severity`** (0.18). The layer reports it at INFO and the
  default floor is Warning, so an obeyed filter makes the feature look broken. Asking for
  the channel by name is the decision the filter would otherwise re-litigate. The layer's
  report boilerplate is peeled off the message, because for a print inside a loop that is
  the difference between a tool and a wall of text; a message matching none of the known
  separators passes through whole, since losing the text is worse than an ugly line.

- **Printf is routed as `Source::SHADER`, matched on the message ID** (0.18). It arrives
  on the same debug-utils messenger as validation findings, so without the routing the
  validation-as-assert fixture would fail any test whose shader says hello. The ID is
  structured data and the text is not; the layer has spelled it `UNASSIGNED-DEBUG-PRINTF`
  and `WARNING-DEBUG-PRINTF` across versions, so the stable part is the suffix. The
  messenger subscribes to INFO only when printf is on, and every other INFO message is
  dropped in the callback — asking for prints must not also subscribe to loader chatter.

- **An occlusion query is not requested as precise** (0.18). Precision needs
  `occlusionQueryPrecise`, and without it the spec allows any non-zero value, so
  `samples` would mean two different things depending on the driver. `samples > 0` is the
  part bazalt can promise everywhere. It refuses outside a rendering scope because Vulkan
  requires the query to begin and end in one render pass, and the alternative is a
  validation message at submit naming neither the call nor the reason.

- **`cmd.label` uses volk's globals, like `set_debug_name`** (0.18). Debug utils is an
  INSTANCE extension: `vkGetInstanceProcAddr` is the sanctioned route and
  `vkGetDeviceProcAddr` may legally return null. The entry points are loader trampolines
  dispatching on the command buffer, so one pointer is right for every Context. An
  unbalanced `end_label` is dropped rather than recorded, because recording one is
  undefined behaviour and the `with` form cannot produce it.

### Asynchronous submits

- **`submit(wait=False)` is paced by the ring, not by a fence per submit** (0.18). The
  hazard is real — with N slots, submit N+2 reuses the command buffer submit N is still
  running — and it is the one the windowed path solves with a fence per slot. Here the
  submission timeline already counts every submit, so remembering which serial last
  occupied a slot is the whole mechanism, and a timeline wait on a value already reached
  costs nothing. A blocking submit pays for none of it: it has already waited.

- **`ctx.wait()` waits on the timeline, not on the device** (0.18). `vkDeviceWaitIdle`
  would stall the upload worker and any other Context sharing the device. The timeline is
  per Context and counts exactly the submits this Context made.

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
   The graphics `storage_image()` declarator arrived in 0.17 WITHOUT waiting for reflection:
   the declarator and the tracking are separate questions, and until then a fragment
   `imageStore` was not merely untracked but unreachable — this file said otherwise, which is
   what a claim with no test behind it does. What reflection still buys is the automatic
   barrier, optional binding declarators, a real diagnostic for the empty-HLSL-entry-point
   ceiling, and — since 0.18 — the ability to compile only the printing shaders
   unoptimized instead of all of them. Target: before 1.0.
4. **The sync-validation test is skipped in CI** — the LunarG layer for noble is still
   1.4.313 and does not report shader hazards. Version 1.4.350 does, but SDK 1.4.350 being
   *released* is not the same thing as a package for noble existing, which is what 0.17 got
   wrong: the workflow was changed to assert the version and the job failed on the assert.
   It now COMPUTES `BAZALT_SYNCVAL_UNSUPPORTED` from `dpkg-query` instead, so the test starts
   running by itself the day the package appears and nobody has to notice. Target: 1.0, and
   it depends on the environment.

   The lesson generalizes past this entry: **a CI gate on someone else's release schedule
   should compute a knob, not assert a version.** An assert turns their timetable into your
   red build, and the thing being gated (one skipped test) is strictly less bad than a job
   that refuses to run at all.

### Ceilings accepted on purpose

These are not debt to pay, they are limits we chose. Each one names its upgrade path.

- **`recreate_swapchain` takes `vkDeviceWaitIdle`** (0.14), so a resize of one window
  stutters the other for a moment. It is correct, it only stutters. Upgrade path: narrow it
  to one swapchain.
- ✅ **A cross-Context transfer regenerates the mip levels above 0** (0.15) — CLOSED in
  0.18. It copies the source's own levels now, one readback and one update per
  (layer, mip). Slower, and deliberately so: the overload is a setup step already.
- **A cross-Context transfer goes through host memory and blocks the source queue** (0.15).
  Without `external_memory` there is no portable alternative, so the documentation calls it
  a setup operation.
- **`FULLSCREEN` is not exclusive fullscreen** (0.16). It takes the monitor and its
  current video mode, but the swapchain stays composited, because exclusive fullscreen
  needs `VK_EXT_full_screen_exclusive`. Upgrade path: a `Feature`, not a fifth
  `WindowMode` — it is a property of the swapchain, not of the window.
- **Fullscreen picks the monitor, and nothing picks the video mode** (0.16). The monitor
  is the one the window overlaps most, which is the difference between "works" and
  "fullscreens on the other screen". A `monitor=` argument and video-mode enumeration
  (resolution and refresh rate) are deferred until somebody asks. Upgrade path: both are
  additive.
- **An HLSL entry point that matches no function compiles to an empty shader** (0.16).
  glslang does not treat it as an error: it synthesizes an entry point under the requested
  name, so the compile succeeds and the shader draws nothing. Detecting it needs SPIR-V
  reflection (debt #3). A test pins the behaviour by SPIR-V size, so a future glslang that
  does complain is noticed and the ceiling can become a real diagnostic.
- **`line_width` other than 1.0 needs `WIDE_LINES`** (0.16), because a driver may support
  exactly one width. Same shape as `polygon_mode` needing `WIREFRAME`: the rasterizer looks
  like free fixed-function state and is not.
- **A preserved second pass re-transitions the attachment** (0.16). Pass 1 retires the
  image to `final_layout()` and pass 2 brings it back, so N passes cost N round trips
  instead of staying in `COLOR_ATTACHMENT_OPTIMAL`. It reuses the existing RenderTarget
  contract rather than inventing a "this pass may continue" state. Upgrade path: a
  recording-wide look-ahead, which is the same machinery the depth store-op would want.
- ✅ **Per-subresource layout tracking does not exist in `Image`** — CLOSED in 0.18. The
  entry called the gain small ("it only removes the loud edge of a partial render"), and
  that was the wrong reading: the loud edge WAS the feature, because it made rendering
  into one mip and then sampling the image impossible. See the decision above.
- **A `DEPTH_STENCIL` attachment cannot be sampled or read back** (0.17). Its view carries
  both aspects. Upgrade path: a second, depth-only view beside the attachment one, the same
  shape as the cubemap's parallel `2D_ARRAY` storage view.
- **The stencil state is the same for front and back faces** (0.17). Upgrade path: a `face=`
  kwarg on `stencil_test`, additive.
- **`primitiveRestartEnable` is always FALSE** (0.17), so a strip is one primitive per draw.
  A restart index would change what the largest index value in an index buffer means, which
  is a decision, not a flag.
- **A specialized compute workgroup size needs Vulkan 1.3** (0.17).
  `layout(local_size_x_id = 0)` compiles to `OpExecutionMode LocalSizeId`, which requires
  `maintenance4`, and the baseline is 1.2. Specialize the numbers the shader reads instead.
  Upgrade path: none needed — it becomes available by itself as the baseline moves.
- ✅ **`copy_image` copies mip 0 only** (0.17) — CLOSED in 0.18. The case came up:
  `image.update(mip=)` makes per-level content ordinary, and a copy that leaves the other
  levels stale is a copy of the top level rather than of the image.
- **A pipeline cache lives and dies with its Context** (0.17). See the decision above: on
  disk it needs a frozen API to be worth writing.
- **Shader printf needs the validation layers** (0.18). It is a layer service, not a
  driver one, so it is unavailable in a release run by construction. Upgrade path: none —
  this is what the feature IS.
- **A printf Context compiles every shader unoptimized** (0.18), not only the ones that
  print. Telling them apart needs SPIR-V reflection (debt #3), and the switch is a
  debugging mode where the optimizer is the smaller loss.
- **An occlusion count is not precise** (0.18). `occlusionQueryPrecise` is a feature bit,
  and without it the spec allows any non-zero value. `samples > 0` is the promise.
  Upgrade path: a `Feature`, additive.
- **`blit_image` covers mip 0 of every shared layer** (0.18). A blit chain between two
  images is a different question, and `generate_mipmaps` answers it on the destination.
- **`image.update` writes one (layer, mip) per call** (0.18). A layered update is N calls;
  they are queued on one worker and land in order, so the result is the same and the API
  stays one verb.
- **A screenshot needs `present(capture=True)` first** (0.18). Not an ergonomic choice: a
  presentable image may only be touched between acquire and present. Upgrade path: none
  that Vulkan permits.
- **A cross-Context transfer of a mipped image is O(layers x mips) readbacks** (0.18).
  Upgrade path: a single readback of every level, which needs a staging layout the host
  side does not currently describe.

---

## Deferred until needed

Additive work with no assigned release. Pull each one in when it really hurts.

- ✅ Async headless submit — DONE in 0.18 (`ctx.submit(wait=False)` plus `ctx.wait()`).
- ✅ A per-level mip copy on a cross-Context transfer — DONE in 0.18.
- **An async `StaticBuffer`.** Still deferred, and now with the reason written down: it
  needs the submit path to track BUFFER residency the way it tracks image residency
  (`require_uploads_resident` walks the images a recording references; there is no
  equivalent list of buffers), and its blocking happens once at setup rather than every
  frame. Rule 4 — plumbing waits until a real need forces it, and nothing has.
- A narrower `recreate_swapchain` (see the accepted ceilings above).

---

## Proposed features

All of them are **additive**, so they break no API and can enter as small additions beside a
big feature. Each one needs a decision. The estimates are whole-feature: C++, the binding
layer, the stub, tests, an example and the docs.

**Ranked first by rule 4, because they make effects:**

- ✅ **Blend modes** — DONE in 0.16 (`blend(enable, mode=)`).
- ✅ **`depth_write` / `depth_compare`** — DONE in 0.16 (`depth_test(enable, write=, compare=)`).
- ✅ **Fragment-shader storage images** — DONE in 0.17. The declarator was the whole gap; the
  automatic barrier still waits for reflection (debt #3), and until 0.17 this file wrongly
  called the feature "reachable today".
- ✅ **Instancing (`VERTEX_INPUT_RATE_INSTANCE`)** — DONE in 0.17.
- ✅ **Stencil** — DONE in 0.17.
- ✅ **CPU image streaming: `image.update(array)`** — DONE in 0.18, with `layer=`, `mip=`
  and `region=`, plus `image.read(layer=, mip=)` as its readback half.
- **Tessellation and geometry stages** (~330 lines). `ShaderStage` has three values, so
  there is no displacement on terrain, no adaptive LOD, no normals drawn as lines, no
  wireframe-on-shaded and no grass or fur grown from a point. Two entries in
  `ShaderStage`, `tesc`/`tese`/`geom` in shaderc, `patch_control_points` and
  `Topology.PATCH_LIST`, and two new `Feature`s — both plain `VkPhysicalDeviceFeatures`
  members, so the table needs no new column. Geometry is slow on modern hardware and
  absent on MoltenVK, which is exactly what rule 3 and `Feature` are for. Ranked first for
  0.19: these make effects.
- **A `RenderTarget` on images you already own** (~200 lines).
  `bz.RenderTarget(ctx, color=[img])`, where the images come from `create_image`. A target
  always allocates its own attachments today, so a graphics ping-pong, drawing into a
  texture brought from another Context, and drawing over a compute-baked texture are all
  unreachable. The usage bits are already there. Open question: it is a fifth way to build
  a target, so check it against rule 1 first — the argument for is that it differs by the
  type of the first argument, exactly like the fourth `create_image` overload.

**Performance:**

- ✅ **Pipeline cache**, in memory — DONE in 0.17. On disk at 1.0, when the frozen API gives
  a stable format.
- ✅ **Specialization constants** — DONE in 0.17.
- **Indirect draw and dispatch** (~250 lines). `draw_indexed_indirect(buffer, offset=,
  count=)` and `dispatch_indirect(buffer)`, plus an `Access.INDIRECT_READ` for the tracker.
  Compute writes the draw arguments, so culling happens on the GPU. It would also make
  `Feature::MULTI_DRAW_INDIRECT` reachable, which is advertised today with no API behind it.
  `What 1.0 means` says to decide this deliberately rather than let it drift. Its
  prerequisite landed early: `cmd.fill_buffer` (0.18) is how the draw-count is zeroed.
- **Bindless / descriptor arrays** (~630 lines). `texture(binding, stage, set, count=N)` and
  `set_image(binding, image, index=)`: one pipeline and one draw for many materials. Needs
  `FeatureInfo` to grow a column first — it maps only plain `VkPhysicalDeviceFeatures`
  members, and descriptor indexing lives in a pNext struct. Already named in the 1.0 list.

**Escape hatches and integration:**

- **Raw-handle interop / `external_memory`** — raw access to the Vulkan handles
  (`VkDevice`, `VkImage`, `VkBuffer`, `VkCommandBuffer`) for C++ ImGui with no copy, CUDA
  interop, a video decoder and OpenXR. Rule 2, and it fits several libraries at once. YAGNI
  until a concrete integration asks for it.
- **ImGui integration** — a companion package or an example on the public primitives, not
  core. See the rejection below for why.

**Small, additive, and waiting for the release that needs them:**

- ✅ **`renderer.read_pixels()`** — DONE in 0.18, as `present(capture=True)` plus
  `read_pixels()`. The estimate assumed one verb; the spec required two.
- ✅ **Async headless submit** — DONE in 0.18.
- ✅ **`buffer.update(data, offset=)`** — DONE in 0.18.
- **Window extras** (~120 lines) — dropped files (drag a texture onto the window and reload
  it), a window icon, setting the cursor position, the clipboard. Queued for 0.19.
- **Gamepad** (~90 lines) — axes and buttons from GLFW. The weakest ratio of value to API
  surface on this list.
- **An async `StaticBuffer`** — see "Deferred until needed" above.

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

- **Text rendering and a debug overlay.** Font rasterization is the ecosystem's
  (freetype, PIL), and the atlas-plus-quads layer is the same argument that kept ImGui out
  of the core: about 90% glue over the public API. A companion package or an example.
- **`image.save(path)`.** No Vulkan glue at all — `read()` returns a numpy array and
  PIL/imageio write the file. It fails the scope test on the second question.
- **Compressed texture loading (BC / KTX2).** Parsing the container is the ecosystem's
  job; accepting already-compressed bytes is a variant of `load_image(bytes)`, not a
  feature of its own.
- **Dual-source blending, blend constants, unnormalized sampler coordinates.** Real Vulkan
  state, but nothing on the proposal list needs them. They stay unlisted until something
  asks.

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
- **Consuming the input state on read** (0.16). It needs no generation counter at all, and
  `was_key_pressed(KEY_F11)` twice in one frame answers True then False. A query whose
  answer depends on how many times you asked is a trap, not an API. We rotate on a
  generation instead, which makes every per-cycle read idempotent.
- **A registry of live windows that `poll_events` clears** (0.16). The obvious way to reset
  per-cycle state, and it needs a global mutable list plus a lock, for work the reader can
  do itself with one comparison. The reader also gets the better semantics for free: a
  frame that never reads keeps its events instead of dropping them.
- **`set_fullscreen(bool)` plus `set_decorated(bool)`** (0.16). Four exclusive states out
  of two bools spells two of them twice and leaves the combination rules to the caller. We
  took one enum and one verb.
- **A settable `window.mode` property** (0.16). The class spells mutation as `set_*`
  already, and this one can fail — an assignment that raises `WindowError` is worse than a
  call that does.
- **A `preserve=` or `load=` bool beside `clear_color`** (0.16). It would let a caller ask
  for both at once, and the answer to "what does this pass do with the old contents" is
  single-valued. `clear_color=None` says it with the parameter that already owns the
  question.
- **`with cmd.compute()`** — a compute dispatch has no teardown, so a context manager would
  be a false symmetry. Chaining covers the aesthetics.
- **String keys anywhere the Vulkan primitive is an index or a handle** — return a handle
  and read the result off it (the 0.9 timers).
- **A separate "debt release"** — rule 5. A debt gets paid by the feature that really needs
  it.
- **A fifth `ValidationMode` for shader printf** (0.18). The modes are exclusive states of
  how hard the layers check; printf composes with every one of them, and making it a mode
  would mean choosing between prints and sync validation. We took a separate switch.
- **Turning validation on implicitly for `shader_printf=True`** (0.18). It would build a
  Context the caller did not ask for. We raise and name the contradiction.
- **Reading the last presented frame in `read_pixels()`** (0.18). Written first, and
  illegal: a presentable image may only be touched between acquire and present. The
  validation-as-assert fixture caught it on the first run, which is the argument for the
  fixture. The copy rides the frame's own submit instead.
- **Coalescing the per-subresource barriers of a split image** (0.18). The callers are
  blocking setup and readback paths, so a handful of extra barriers there is cheaper than
  the bookkeeping to merge runs of equal layouts. The collapse already keeps the common
  case at one barrier.
- **Exposing both combined depth/stencil formats** (0.17). `D24S8` and `D32F_S8` differ by
  which driver has them, which is the one thing the caller cannot know and the library can.
  We took one `DEPTH_STENCIL` name and picked per device.
- **A `stencil=` kwarg on `RenderTarget`** (0.17). The stencil is part of the depth
  attachment's format in Vulkan, so a second bool beside `depth=` would let a caller ask for
  a stencil with no depth attachment at all. `depth=bz.Format.DEPTH_STENCIL` says it once.
  `SwapchainRenderer(stencil=True)` is the exception and earns it: a window has no `depth=`
  parameter to carry the answer, because its depth buffer is scratch the renderer owns.

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

- **A query that reports a change must be idempotent within the frame.** The level/edge
  pair (`is_key_pressed` / `was_key_pressed`) and the mouse delta are all "what happened
  since the last cycle", and the tempting implementation clears on read. Rotate on a
  cycle counter instead: two call sites in one frame is normal code, not misuse.

- **One mechanism for everything that expires at the same moment.** The scroll wheel, the
  key edges and the mouse delta looked like three features and were one: state accumulated
  during a poll cycle. Finding the shared lifetime turned three implementations into one
  counter, and it is why the 0.16 input work was small.

- **A default argument can change the behaviour of an existing call.** `depth_test(enable)`
  derived `depthWriteEnable` from `enable`; adding `write=True` would have made
  `depth_test(False)` write depth with the test off, which is legal Vulkan and nobody's
  intent. The new state is gated on the old one (`depth_test && depth_write`). Check what
  an added parameter's default does to every call that predates it.

- **A capability that looks like fixed-function state usually has a feature bit.** Three
  releases in a row found one: `fillModeNonSolid` (wireframe), `wideLines`, and now
  `independentBlend` for a per-attachment blend state. Before adding a knob that "just fills
  in a struct field", read the VU for the field.

- **When one name has two spellings per device, expose the name and pick the spelling.**
  `Format.DEPTH_STENCIL` resolves to `D24_UNORM_S8_UINT` or `D32_SFLOAT_S8_UINT`. The general
  form is the `Features.hpp` argument: the caller decides WHAT they need, the library decides
  which Vulkan word says it here. The cost is that a `constexpr` table can no longer answer
  everything, so the resolved value has to live on the object (`Image::vk_format()`).

- **A verb that removes a method is worth more than a verb that adds one.**
  `draw_indexed(instances=)` made `draw_indexed_instanced` disappear and gave `draw` the same
  argument it never had. Adding a parameter to an existing verb is nearly always smaller than
  the new name it replaces.

- **A ceiling's stated reason can be wrong, and it gets copied forward every release.**
  Per-subresource layout tracking was dismissed as "it only removes the loud edge of a
  partial render". The loud edge WAS the feature: it is what made rendering into one mip
  and then sampling the image impossible, so the ceiling was not a small missing polish
  but the reason two whole workflows did not exist. Re-derive a ceiling's cost when the
  release around it changes, instead of re-reading the sentence.

- **When the spec forbids the one-call shape, take two calls and say why.** A screenshot
  "should" be `renderer.read_pixels()`. A presentable image may only be touched between
  acquire and present, so that shape cannot be correct, and the honest API is
  `present(capture=True)` plus `read_pixels()`. Bending the API around a spec rule is
  better than an API that reads well and is illegal.

- **An asynchronous operation must mark its resource busy BEFORE queueing it.** An
  `image.update` that queued first left an image created synchronously in upload state
  None, so `wait()` returned immediately and the next `read()` gave the PREVIOUS
  contents — one update behind, forever. A wrong answer, not a slow one, and invisible to
  a test that only checks the final state.

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

- **A presentable image belongs to the compositor outside acquire..present.** Touching it
  after `vkQueuePresentKHR` — including a layout transition for a readback — is a
  validation error naming the swapchain, not the code that did it. Anything that wants a
  copy of a frame records it into that frame's own submit.

- **A non-semantic SPIR-V instruction is one the optimizer is entitled to delete.**
  `debugPrintfEXT` has no observable result, so `-O` removes it and the shader silently
  stops printing. Any feature built on `NonSemantic.*` has to turn the optimizer off, and
  that cost belongs in the decision, not in a bug report six months later.

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

- **A `VkAttachmentLoadOp` and the entry barrier are one decision.** `UNDEFINED` as
  `oldLayout` is not a shortcut, it is a *discard*, and it is correct only while the pass
  clears. Any new load-op has to be read together with the barrier above it, and with the
  store-op of whatever ran before.

- **`VkPipelineShaderStageCreateInfo::pName` must match the name the SPIR-V was compiled
  with.** It was `"main"` everywhere, which is right until `entry_point=` exists: the module
  then declares `VSMain` and pipeline creation fails with `VK_ERROR_UNKNOWN` and no
  validation message worth reading. This is the second reason the module stores its entry
  point, next to the hot reload. Whenever a compile gains a knob, check what downstream still
  assumes the old default.

- **A floating-point depth buffer scales `depthBiasConstantFactor` by roughly 2^-24.** The
  spec scales it by 2^(exponent of the largest depth in the primitive − mantissa bits), so at
  z = 0.9 the factor is about 6e-8, and the 0.001-ish constants every D24_UNORM tutorial
  quotes move the depth by literally nothing. A visible offset on D32F needs five or six
  digits. Anything the spec defines "in units of the depth format" has to be re-derived per
  format, not copied.

- **`VK_POLYGON_MODE_LINE` is not core-mandatory.** It needs `fillModeNonSolid`, which every
  desktop driver has and some mobile ones do not, so the wireframe view is a `Feature`
  (`WIREFRAME`, already in the table since 0.5) and not a free rasterizer flag. This was
  caught by the validation-as-assert fixture on the first run, not by reading the spec —
  which is the argument for the fixture. General form: "it is core Vulkan" and "it needs no
  feature bit" are different claims.

### pybind, C++ and lifetime

- **`raise_error` under `py::gil_scoped_release` is an access violation, not an exception.**
  The bug sat in `require_uploads_resident` since 0.5 on a rare path, and the 0.14
  CommandBuffer guard put it on the path of an ordinary user mistake.
- **Overload order by Python type is now a standing rule, not an incident.** pybind
  converts `str` AND `bytes` to `std::string`, so the `bytes` overload must be DECLARED
  first or the `str` one swallows it. Third occurrence: `compile_shader(source=)` (0.16),
  the specialization-constant `bool` before `int` (0.17), and `load_image(bytes)` (0.18).
  Whenever the Python type carries the meaning, write the narrow overload first and add a
  test that proves the wrong one is not chosen.

- **When a field grows from a word into a container, re-read every thread that touches
  it.** `Image::layout_` was a `VkImageLayout` the upload worker wrote directly — racy by
  the book, benign in practice. Turning it into a vector made the same line a real data
  race AND semantically wrong (it claimed every layer sampleable when one was written).
  The size of a field is part of its threading contract.

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

- **pybind converts `str` AND `bytes` to `std::string`.** A parameter typed
  `std::optional<std::string>` accepts both and cannot tell them apart, so a `bytes` argument
  meant as SPIR-V would be compiled as GLSL. Take `py::object` and test
  `py::isinstance<py::bytes>` FIRST whenever the Python type is the thing that carries the
  meaning.

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

- **OS input cannot be synthesized**, so `test_window_input.py` pins the half that is ours:
  a per-cycle query is repeatable inside one frame, and it starts at rest. The keystrokes
  themselves are covered by running `examples/21_window_modes`. Do not mistake the thin
  input tests for the window-mode tests in the same file, which *are* real: each mode
  renders a frame with validation as the referee.

- **A pixel test for new pipeline state has to be two-sided.** Render the same geometry with
  and without the setting and compare, because a knob that silently does nothing otherwise
  passes: `polygon_mode(LINE)` is checked against the FILL pixel count, and
  `depth_test(write=False)` against the same pair with `write=True`.

- **CI wall clock is one wheel, not four.** The extension is one enormous translation unit
  and compiles in about the same time for every interpreter, so a job that builds cp310
  through cp313 in sequence costs four times the wall clock of one that builds them in
  parallel. One job per interpreter per platform. The runners are free on a public repo, so
  the extra minutes cost nothing and the pipeline went from 12 minutes to about 5. The same
  reasoning splits Linux and Windows into separate jobs: `test_lavapipe` needs only the
  Linux wheel, and as one matrix job it also waited for Windows, which finishes four
  minutes later.

- **The full interpreter matrix runs where it gates something**, not on every push. A pull
  request, a release and a manual run build cp310–cp313. A plain branch push builds only the
  cp312 the lavapipe legs install. Nothing reaches master without the full matrix having
  passed on the pull request first, so the push-time saving costs no coverage.

- **MSBuild stays the Windows generator.** Ninja is genuinely faster — 45s against 63s for
  one wheel, measured — but CMake's Ninja generator needs `cl.exe` on PATH, and neither the
  runner nor scikit-build-core sets up the MSVC environment (scikit-build-core only does it
  for the Visual Studio generator). Buying that 30% costs a `vcvars` step in CI and a
  Developer Command Prompt for every local build. Parallel jobs bought more, for free.
  `/MP` was measured too and is noise (60s against 63s): MSBuild's cost here is per-project
  overhead, not serialized compiles, and one huge translation unit has nothing to
  parallelize.

- **`cmake.args = ["-A", "x64"]` was redundant** and is gone. scikit-build-core already sets
  `CMAKE_GENERATOR_PLATFORM` from the interpreter's own architecture for Visual Studio
  generators, which is more correct than hard-coding x64. Beware the argument *form* if this
  ever comes back: scikit-build-core reads the generator off arguments that start with `-G`,
  so `["-G", "Ninja"]` as two list entries is invisible to it. It then treats the build as
  multi-config, never passes `CMAKE_BUILD_TYPE`, and the debug CRT link-fails against the
  SDK's release `shaderc_combined`. `["-GNinja"]` works.

- **cibuildwheel 4 repairs Windows wheels with delvewheel by default**, which would bundle
  `vulkan-1.dll` into the wheel and shadow the loader the user's driver installed.
  `CIBW_REPAIR_WHEEL_COMMAND_WINDOWS: ""` turns it off. Everything else bazalt links —
  shaderc, glfw, volk, vk-bootstrap — is static, so there is no repair left to do.

---

## Verification — open items

- **A golden image per example on lavapipe in CI**, plus the validation layers as an error
  in every test. This is a permanent regression guard and it is still incomplete, because
  the tests assert pixel values directly instead of comparing reference images.
- **Remove the sync-validation skip** when LunarG packages a newer layer for noble (debt #4).
- **`examples/24_video_texture` is the windowed referee for streaming** (0.18). It ran
  ~1900 frames on a real driver with no validation output and `ctx.memory_stats()` flat at
  1.5 MB, which is the claim the feature makes: streaming into ONE image does not grow.
  A version that built a new Image per frame would climb there.
- **The headless fallback** (no windowing extensions) still has no coverage. It is a separate
  path from the API version, which CI does cover since 0.15.
- **The windowed path is verified by running the examples, and that is load-bearing.** The
  0.17 depth/stencil layout bug existed only for a target whose depth is scratch — a window —
  and every headless test passed with it in place. Run the new example before calling a
  release done; "the suite is green" does not cover the half of the library that needs a
  surface.
