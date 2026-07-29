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

   **0.20 is the exception, and it is written down here so it stays one.** It has no
   feature. It splits the binding layer into eight translation units, fixes four bugs and
   removes one accidental API. What made it legitimate is that the work had no feature to
   attach to: a build-time restructuring is forced by the *size* of the code, not by
   anything a user asked for, and the four bugs were found by auditing the binding layer
   rather than by writing something new. Rule 5's target is a release that collects
   unrelated debt because the debt is old. This one collects work that is all the same
   subject. If a future release wants the same exemption, the test is that question: does
   the work share a subject, or only a mood?
6. **An API break is acceptable before 1.0, but you batch the breaks into one release.**
   This is also why 0.20 is not 0.19.1. `CHANGELOG.md` promises that a patch release never
   breaks the API, and removing `.export_values()` deletes about sixty names that were
   reachable. Undocumented is not the same as absent.

Rules 4 and 5 have a corollary that the 0.9 to 0.15 sequence confirmed: **the most invasive
work goes last.** The volk dispatch-table pass touched every `vk*` call. Done early it would
have taxed every later feature, because each new call site would have to go through a fresh
dispatch layer. Done after the resource types settled, it was one mechanical pass over a
complete, frozen set of call sites.

0.19 confirmed it again and added a second corollary: **the estimate is for the feature, and
the invasive part is usually next to the feature rather than in it.** Tessellation was
estimated at ~330 lines and the stages themselves were about that. What the estimate missed
entirely was that the per-Context shader stage mask had to replace an `inline constexpr`,
which is a pass over some twenty call sites in three headers — bigger than the feature, and
forced by it. The same shape appeared twice more in one release: reflection needed
`bind_pipeline` to start keeping state and `DescriptorSet` to remember its binding indices,
and it needed two new `Feature` rows before the thing it fixes was even legal to do. When
sizing a feature, ask what it makes *unavoidable* elsewhere; that is where the release goes
over.

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

### The binding layer, and why it is eight files

- **The bindings are split by subject, the core stays header-only** (0.20). Until 0.20 the
  whole library was one translation unit: `main.cpp` held every binding and included the
  entire header-only core, so a rebuild used one core of however many the machine has, and
  0.14 had already needed `/bigobj` because the object file passed COFF's 65,536-section
  limit. `src/bindings/` now holds one `.cpp` per subject — enums, resources, pipelines,
  commands, windowing, context, targets — and `main.cpp` is the `PYBIND11_MODULE` plus seven
  calls.

  The core headers stay headers. They are templated and `inline`, so splitting them into
  `.hpp` + `.cpp` would be a rewrite.

  **The build-time result is a trade, not a win, and the honest numbers belong here.**
  Against a same-configuration baseline of 36.6s: a rebuild after editing one binding file
  is 25.8s (30% faster), and a rebuild after editing a core header is 43.9s (20% slower).
  Eight translation units each parse the header-only core where one did before.

  The reason the second number could not be rescued is worth recording, because three
  attempts failed and each looked like it should have worked. `/MP` does nothing here —
  MSBuild parallelizes across projects, not across the sources inside one, and the flag made
  both cases marginally worse. `VS_GLOBAL_UseMultiToolTask` did nothing either. The Ninja
  generator, which parallelizes properly, cannot find `cl.exe` without an MSVC environment,
  so it would require every build and every CI job to run from a Developer Prompt — a real
  cost for a build-time gain. Putting the core headers in the PCH moves the binding-edit case
  to 20.8s and the header-edit case to 45.4s, because the PCH rebuild is then a serial step
  in front of everything: roughly a wash, so the simpler behaviour wins.

  What this leaves: **the header parse dominates a rebuild, and it always did.** The single
  translation unit was never slow because `main.cpp` had 3,500 lines of bindings in it; it
  was slow because it included 15,600 lines of headers. That is why the split cannot beat it
  on a full rebuild — a full rebuild has to parse those headers at least once no matter how
  many files there are — and it is the measurement that would justify splitting the core
  later, if anyone ever wants to pay for it. **The reason to keep the split is the structure,
  and the release note says so rather than claiming a speed-up it does not have.**

- **`bindings/Common.hpp` is `inline`, not an anonymous namespace** (0.20), and this is the
  one part of the split that fails silently rather than loudly. The shared helpers were an
  anonymous namespace in `main.cpp`, which is correct for one translation unit and wrong for
  eight: an anonymous namespace in a header gives every `.cpp` its own private copy, so
  `register_exceptions(m)` would fill in `main.cpp`'s copy of the `exc_*` handles and every
  `raise_error` from any other file would go through a null one. It links, and it crashes at
  runtime. `inline` gives one definition shared by all of them.

  Two definitions escaped the first mechanical pass because they did not start with the
  return type: `[[noreturn]] void raise_error` and a `const char*` function whose line the
  pattern did not match. Those fail at link time, which is the good case.

- **The precompiled header holds third-party headers only** (0.20). No `src/*.hpp` goes in
  it. The bazalt headers are the ones that change during feature work, and a header inside
  the PCH means every edit to it rebuilds the PCH first — a serial step in front of a
  parallel build, for no gain, because a changed bazalt header already rebuilds all eight
  files anyway. Third-party headers never change, so the PCH survives every rebuild.

  Order inside it is load-bearing: pybind11 before volk. `volk.h` includes `<windows.h>`
  when `VK_USE_PLATFORM_WIN32_KHR` is defined, `<windows.h>` defines `min` and `max` as
  macros, and `pybind11/numpy.h` calls `std::min`. `main.cpp` had the right order by luck
  for twenty releases; the PCH has it on purpose, with the reason next to it.

- **Two registration orders are real, and one that looks real is not** (0.20). `py::arg(x) =
  SomeEnum::VALUE` casts the default at `.def()` time, so every enum used as a default
  argument must be registered first — that is why `bind_enums` runs first. `RenderTargetBase`
  must be registered before `OffscreenTarget` and `SwapchainRenderer`, the module's only two
  derived registrations — that is why `bind_targets` runs last, and both live in the same
  file so the constraint cannot be split. `RenderTarget` as a *parameter* type constrains
  nothing: pybind resolves a caster at the first Python call, not at `.def()` time. The call
  order in `main.cpp` carries all three reasons, because the order is the only documentation
  a future binding has.

- **`build-dir` in `pyproject.toml` was the cheapest win in the release** (0.20), and it was
  not in the plan. scikit-build-core builds in a temporary directory and deletes it unless
  told otherwise, so every `pip install .` reconfigured CMake and rebuilt glfw, volk and
  vk-bootstrap from scratch. Two lines of configuration took a rebuild from 66.1s to 36.6s
  before a single line of C++ moved — a bigger improvement than the restructuring it was
  measured in preparation for, and it ended up being the release's largest.

  The lesson is about sequencing: **measure the thing you are about to optimize, because the
  measurement is where you find out that most of it was not the thing you thought.** Of the
  36.6s that remained, 10.2s was pip and CMake overhead no source change can move, which set
  the ceiling on what the split could ever be worth. Had the baseline not been taken first,
  the release would have credited the split with a 66→26s improvement that was mostly a
  configuration default.

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
  `target.layer(i, mip=)` and `target.all_layers()` return a light view object that goes
  straight into `cmd.rendering(...)`. (0.13 also shipped `target.mip(m)`; the 0.18 audit
  removed it as a second spelling of `target.layer(0, m)`.) The verb gets no knobs:
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
     renders into. The guard lives in the binding layer for the same reason as the
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
  the binding layer, not in the headers: `CommandBuffer` methods return `*this` for chaining and
  have no error channel, and this catches a **user error**, not a C++ invariant, so it
  belongs in the binding layer where the GIL is held and `raise_error` is legal. Every class
  got an `owner()` accessor (not `context()`, because `SwapchainRenderer` already has a
  `context()` that returns a `shared_ptr`).

- **`Image` is a future, not a `Future[Image]`** (0.5). The resource is ready to record with
  at once, and only the submit needs residency. `img.wait()` and `ctx.wait()` are a separate
  verb, not a second version of `load_image`.

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
  stopped being right the moment rendering into a single mip existed (0.13): the whole image was
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

### Shader stages, and the mask they broke

- **Geometry is not redundant with tessellation, and mesh shaders do not settle it**
  (0.19). Tessellation subdivides a patch through fixed-function hardware and hands back
  the same kind of primitive, so "triangles in, lines out" is outside what it can express:
  normals as lines, a point becoming a quad, and per-primitive culling have no tessellation
  route. What DID take geometry's most famous job is already here — routing one draw into
  every layer used to need `gl_Layer` from a geometry shader, and `target.all_layers()`
  (multiview, 0.13) does it without one, which is why the stage looks redundant at first
  glance. Mesh shaders supersede both, reach roughly half of reported devices, and give
  back no fixed-function tessellator, so they stay a future additive `Feature` (see the
  rejection of a 2.0). Geometry's marginal cost once the tessellation plumbing exists is
  about forty lines, so skipping it would have saved nothing measurable.

- **The shader stage mask had to stop being a constant, and that was the release's
  invasive pass** (0.19). `kAllShaderStages` was an `inline constexpr` covering vertex,
  fragment and compute. Neither option survived tessellation: wide enough to cover a
  tessellation read, it is illegal on a Context without the feature
  (VUID-vkCmdPipelineBarrier-srcStageMask-04090/-04091, and the validation-as-assert
  fixture turns that into a suite-wide failure); narrow enough to always be legal, it
  silently drops the read a tessellation shader just made. Only the enabled feature set
  answers both, so it became `Context::all_shader_stages()`.

  The constant was **deleted** rather than given a defaulted parameter, and that is the
  verification: a stage mask has no other referee, so a missed call site has to be a
  compile error. Same lesson as `volkLoadInstanceOnly` in 0.15 — make the old road stop
  working. `CommandBuffer::execute` gave a second, independent reason: its replay
  wrap-around barrier read `constexpr … = kAllShaderStages | VERTEX_INPUT_BIT`.

- **A stage's feature is checked at `compile_shader`, not only at `build()`** (0.19).
  SPIR-V for these stages declares `OpCapability Tessellation` / `Geometry`, and
  `vkCreateShaderModule` rejects a capability whose feature is off
  (VUID-VkShaderModuleCreateInfo-pCode-08740). A gate only in the pipeline builder fires
  *after* the invalid module exists, so the diagnostic belongs at the line the user wrote.
  The pipeline gate stays and is not a duplicate: it catches a module compiled on a Context
  that HAS the feature. Both call one function, so they cannot disagree about which feature
  a stage needs or about the wording.

- **A specialization constant is keyed by stage, not sorted into two lists** (0.19).
  `constant(id, bytes, stage)` tested `FRAGMENT` and sent everything else to the vertex
  list, so a `TESS_CONTROL` constant would have been baked into the vertex shader. A switch
  would only have moved that bug into a `default:`; a map keyed by stage has no illegal
  state, because a bucket no module claims is simply never read.

- **`Topology.PATCH_LIST` and tessellation imply each other** (0.19). A patch has nothing
  to subdivide it and a tessellation pipeline has nothing to read without patches, so both
  spellings of the mistake are refused with the same explanation. `patch_control_points` is
  the INPUT patch size, which is why it lives on the pipeline: the control shader's own
  `layout(vertices = N) out` is its OUTPUT count, a different number, and neither can be
  derived from the other.

### Reflection, and what it is allowed to conclude

- **Reads are always assumed; writes are only ever narrowed by a positive proof of
  absence** (0.19). This one sentence is the whole safety argument for
  `src/SpirvReflect.hpp`. Every uncertainty sets `writes_unknown` and `writes()` then
  answers true for everything, so an uncertain module behaves exactly as bazalt did before
  reflection existed. The only way to LOSE correctness is a parser bug that positively
  claims "no write" where there is one, which is the narrow class the opcode tables are
  built to keep small.

- **The atomics are not a detail of the write scan, they are half of it** (0.19). A buffer
  mutated by nothing but `atomicAdd` is what a GPU counter looks like, and `imageAtomicAdd`
  reaches its image through `OpImageTexelPointer` rather than through a load. A scan that
  looked only for `OpStore` would report both as read-only and silently reinstate the exact
  debt this pays off. `OpAtomicLoad` is the one atomic that is a read, and a buffer touched
  by nothing else is genuinely read-only.

- **Provenance is the trust boundary, not capabilities** (0.19). The first version of the
  parser had a capability allowlist to catch instruction sets it had never been taught.
  Wrong mechanism, and instructively so: the numbers are a long enum, a list written from
  memory is wrong in both directions, and being wrong the SAFE way (flagging a capability
  that is fine) silently turns the whole feature off while looking like it works. The real
  question is "did bazalt compile this?" — which the caller knows for certain and the
  parser cannot see at all. `ShaderCompiler` sets `writes_unknown` for `.spv` and
  `source=bytes`, and the parser stays about parsing.

- **Reflection is computed in `compile_parts`, not in `compile`** (0.19). It travels in the
  same call as the words it describes, so the two cannot drift. One level up, every
  hot-reloaded module would keep its original's answers and the tracker would order the
  shader that used to be there — silently, and only after an edit. This is the 0.16 rule
  ("anything a build depends on that it cannot re-derive lives with the result") applied to
  derived data instead of settings.

- **The tracker asks the pipeline at record time, and the pipeline asks its modules**
  (0.19). `Pipeline::shaders()` returns the live list rather than a precomputed write map,
  which is the same trick `desc_.recreate` uses for the handles: a `replace()` is picked up
  with nothing to invalidate. `bind_pipeline` had to start recording the bound pipeline as
  record-time state, because before this a draw had no way to find out which pipeline it
  belonged to. With no pipeline bound the answer is the conservative one — a draw with no
  pipeline is a bug the layers name precisely, and guessing "not written" there would drop
  a real barrier.

- **`track_draw_` and `track_dispatch_` are one function differing by a stage mask**
  (0.19). They used to disagree about the same question: the graphics one called every
  storage buffer a READ and handled storage images not at all, the compute one called both
  READ+WRITE unconditionally. Compute LOSES barriers in the merge, and that is the point:
  `use(..., writes=true)` wipes read state, so two dispatches that only read one input SSBO
  used to get a WAW barrier between them and to switch on the per-replay memory barrier.

- **A graphics shader that writes a descriptor needs a feature bit** (0.19), and the gate
  asks the reflection rather than the declarator, so declaring a storage image and only
  reading it costs nothing. Fourth release running where something that reads like plain
  command recording turns out to have one, after `fillModeNonSolid`, `wideLines` and
  `independentBlend`. The gate reads `written_bindings` and not `writes()`, because a module
  flagged `writes_unknown` claims to write everything and demanding the feature for every
  `.spv` shader would break callers who never write at all.

### Indirect work

- **The indirect verbs are three verbs, and bazalt declares no struct for their arguments**
  (0.19). A `buffer=` kwarg on `draw` would invalidate `vertex_count` and `instances` in
  the same signature, which is the shape 0.15 rejected for the cross-Context transfer. And
  the argument layout is `VkDrawIndirectCommand` — four `uint32`s that numpy already
  writes, byte-identical to a std430 GLSL struct. A dtype declared in bazalt would be a
  type that exists only to be converted, which fails the scope test's second question.

- **`BufferType::STORAGE` carries the indirect usage bit unconditionally** (0.19). The
  whole use case is a compute shader writing the arguments, so the type that carries
  `STORAGE_BUFFER` is the type that carries this. A fifth `BufferType` would fragment "can
  I bind this as an SSBO" and make "which buffers can be indirect" a second rule to
  remember, for a bit that costs nothing — the same reasoning that gave DYNAMIC buffers the
  transfer bits in 0.18. `type_` moved from `DynamicBuffer` to the `Buffer` base so the
  verbs can refuse a non-STORAGE buffer by name instead of leaving the layers to report a
  usage flag.

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
  dispatching on the command buffer, so one pointer is right for every Context. An unbalanced
  end is dropped rather than recorded, because recording one is undefined behaviour —
  and `with cmd.label(...)`, the form to reach for, cannot produce one.

### What blocks and what does not

- **The rule: every write is asynchronous, every read blocks** (0.18.0). This is the
  answer to "which calls stall?", and before 0.18.0 there was none — `load_image(path)`
  was asynchronous, `create_image(array)` right beside it was blocking, and nothing in
  either name said so. Both make an Image out of pixels, so "one obvious way" was
  already broken; the fix was to pick the side that can be one way.

  Writes can, because **the resource is its own future**. `create_buffer` and
  `create_image` return the handle either way, so making them asynchronous changes no
  signature and adds no `Future[T]` type — 0.18's `image.update` proved it by flipping
  from blocking to asynchronous without moving a parameter. Reads cannot: `buf.read(...)`
  returns a numpy array, an asynchronous version has to return a future plus a
  `.result()`, and that is a second shape, not one path. It would also buy nothing — you
  ask for the bytes because the next line uses them.

- **Two different things are called "asynchronous", and they are not one feature**
  (0.18.0). *Async transport* is the upload worker: a thread that decodes a PNG off the
  main thread, so the win is CPU-side and only exists where there is decoding to do.
  *Async submit* is skipping the wait: no thread at all, and the win is that the GPU
  stops idling between round trips. `load_image` needs both. `create_buffer` and
  `create_image(array)` need only the second, because the caller already handed over
  decoded bytes.

  Which is why neither of them goes through `UploadManager`. Routing them there would
  add a queue, a job variant and a failure that arrives on another thread, to move a
  `memcpy` that has to stay on the main thread anyway (it reads a Python object under
  the GIL). One-shot `deferred_submit` instead: the copy is submitted by the calling
  thread — so **every error still raises at the `create_*` call**, which is the property
  `load_image` has to work for with a synchronous header probe — and only the
  `vkQueueWaitIdle` is gone. That wait-idle was the entire cost: 30 meshes meant 30 full
  queue drains at startup.

- **`deferred_submit` is `immediate_submit` minus the wait, and `immediate_submit` is
  now written in terms of it** (0.18.0). Two copies of allocate → begin → record → end →
  submit → signal would drift. What the split names is the real distinction: a readback
  still blocks (see the rule above), an upload keeps its serial. The one-shot command
  buffer retires through the deletion queue rather than being freed inline, because
  without the wait the GPU still holds it — and the deletion queue is keyed on exactly
  that serial already.

- **Buffer residency is a list on the CommandBuffer, not a hook in `track_use_`**
  (0.18.0). The submit path had `require_uploads_resident` walking the images a
  recording references; buffers needed the same walk, and `track_use_` looks like the
  place because every bind already calls it. It is not: it returns early with
  `auto_barriers=False`, and residency is not a barrier the caller can take over —
  turning off automatic barriers must not silently turn off waiting for uploads.
  `record_buffer_use_` is separate and unconditional, and it skips buffers with serial
  0, so a recording of DYNAMIC buffers stores nothing.

  This is the plumbing that 0.18's "deferred until needed" entry said an async
  `StaticBuffer` would need, and the estimate was right about the work — it was wrong
  about the trigger. The trigger was not a mesh loader getting slow. It was the API
  being unpredictable.

- **One wait verb, and one narrowing of it** (0.18.0). `ctx.wait()` waits for
  everything this Context started; `res.wait()` narrows that to one resource. Nothing
  else.

  The release passed through a five-verb version first — `ctx.wait()`,
  `ctx.wait_for_uploads()`, `ctx.wait_idle()`, `res.wait()`, plus the `uploads_done` and
  `upload_progress` predicates — each one true, each one a slightly different width. That
  is the failure mode rule 1 describes: no single addition was wrong, and the sum was a
  ladder the reader has to memorize before they can wait for anything. Three of them
  ended at the *same timeline semaphore*, so the widths were not even real.

  What made them look distinct was bookkeeping, not semantics. One-shot uploads
  (`create_buffer`, `create_image(array)`) never entered the worker's batch, so the
  Context tracked their serial separately and `uploads_done` had to OR two systems
  together. Registering them in the same batch — started and submitted in one step, since
  they have no decode stage — deletes the second system, and with it the reason
  `wait_for_uploads` was a different verb from `wait`. `upload_progress` then covers
  every upload rather than decodes only, so `upload_progress == 1.0` says what
  `uploads_done` used to.

  `ctx.wait_idle()` went the same way. It was added in 0.17, before `ctx.wait()` existed,
  and its docstring reason — "timing a batch of work, making sure nothing is in flight" —
  is `ctx.wait()`'s job. What `vkDeviceWaitIdle` adds over the timeline is stalling *other
  Contexts*, which is a bug in every caller that wanted it. The C++ `Context::wait_idle`
  went with the binding: the destructor and swapchain recreation call `vkDeviceWaitIdle`
  directly, and they are the only places that legitimately want the device.

- **One GPU wait, one one-shot submit, one staging buffer** (0.18.0). Under the API,
  five call sites had hand-rolled "wait for timeline value N" (two `vkWaitSemaphores`
  copies, two `vkQueueWaitIdle`), `UploadManager` had a copy of `deferred_submit`'s
  submit block, and six places built a staging buffer with the same twenty lines. All
  three are now single functions: `Context::wait_for_serial`, `Context::submit_one_shot`,
  and `create_staging_buffer(ctx, size, direction, data)`.

  The two `vkQueueWaitIdle` replacements changed behaviour for the better. Both call
  sites already held the exact serial their submit signalled, so draining the whole queue
  made a readback or a `submit(wait=True)` block on whatever the upload worker happened
  to have in flight. They now wait for their own work only.

  One thing that looks like duplication and is not: the upload worker still frees its own
  command buffers (`retired_`) instead of using the deletion queue. VMA is internally
  synchronized so a staging buffer can retire anywhere, but command pools are externally
  synchronized — the main thread draining the deletion queue while the worker allocates
  from the same pool is a race the validation layers flag. `submit_one_shot` therefore
  covers the submit and leaves the freeing policy to the caller, which is where the two
  threads genuinely differ.

- **`UploadManager` is created with the Context, not on the first `load_image`** (0.18.0).
  Lazy creation cost five `if (!upload_manager())` blocks in the bindings and a null
  branch in every aggregate query. Once the manager became the single place that counts
  uploads, "it might not exist yet" stopped being an acceptable state. The worker thread
  parks on a condition variable, so a Context that never loads an image pays a blocked
  thread and nothing else.

### Asynchronous submits

- **`submit(wait=False)` is paced by the ring, not by a fence per submit** (0.18). The
  hazard is real — with N slots, submit N+2 reuses the command buffer submit N is still
  running — and it is the one the windowed path solves with a fence per slot. Here the
  submission timeline already counts every submit, so remembering which serial last
  occupied a slot is the whole mechanism, and a timeline wait on a value already reached
  costs nothing. A blocking submit pays for none of it: it has already waited.

- **`ctx.wait()` waits on the timeline, not on the device** (0.18). `vkDeviceWaitIdle`
  would stall any other Context sharing the device. The timeline is per Context and
  counts exactly what this Context submitted.

- **`submit(cmd, wait=True)` stays a keyword, not two verbs** (0.18). The audit that
  collapsed the wait verbs left this one alone on purpose, and the reason generalizes:
  a flag that selects *whether to wait for work you just described* is one path, because
  the work is identical either way and only the caller's next line differs. Two names —
  `submit_sync` / `submit_async` — would duplicate every future argument to `submit`.
  Contrast the wait verbs, which were not one function with a flag but five functions
  ending at the same semaphore.

### Errors and the pybind boundary

- **The exception type is the recoverability contract.** `ErrorCode` maps 1:1 onto a Python
  exception class through the `unwrap()` helpers. `ShaderError` must be catchable on its
  own, or hot reload is pointless.

- **Which exception a user error gets** (0.20). `ErrorCode` decides which `BazaltError` a
  *bazalt* failure becomes, and says nothing about the errors the binding layer diagnoses
  itself. Those had drifted into two answers for one question:
  `image.read(layer=9)` raised `ResourceError` from the core and `image.update(pixels,
  layer=9)` raised `ValueError` from the binding lambda. The C-contiguous rule was worse —
  the *same sentence* arrived as `ResourceError` from `Buffer.update` and as `ValueError`
  from `Image.update`, and the type stub documented only the first.

  The line is **what had to be consulted to know it was wrong**:

  - **`ValueError`** — the call is malformed on its own. A value outside a fixed range in
    the signature (`frames_in_flight` is 1..4), a name outside a fixed set
    (`validation="nonsense"`), a sequence of the wrong length (`region=` is four numbers).
    No resource, no data, no device enters the decision, and fixing it means editing a
    literal.
  - **`BazaltError`, usually `ResourceError`** — the call is well formed and wrong for this
    resource, this data or this GPU. A layer or mip the image does not have, a dtype or
    shape its format does not accept, a strided array, a multisampled image being written.
    Deciding it means asking an object.

  Nine sites moved to `ResourceError` and three keep `ValueError`. The reason the
  `ValueError` half is not folded into the hierarchy for tidiness: `except
  bz.InitializationError` is the "fall back to headless" handler, and a typo in a keyword
  argument must not be caught by it. `except bz.BazaltError` must not either. Python's own
  answer for "your argument is nonsense" is already `ValueError`, and pybind raises it next
  door for a failed cast, so this agrees with the language rather than inventing a rule.

  Two things generalize past the fix. **A user error decided in the binding layer needs the
  rule stated, because `unwrap()` does not reach it** — every one of the eleven was written
  in isolation and each was locally defensible. And **the drift was invisible to the tests
  that covered it**: `test_streaming.py` asserted `ValueError` five times and passed, because
  a test written beside the code inherits the code's assumption. What catches this class is a
  test that puts the two spellings of one question side by side
  (`test_read_and_update_agree_on_the_exception_type`).

- **Anything under a released GIL returns `std::expected`** (0.14). A pybind throw
  constructs a Python object, so it needs the GIL, and `raise_error` under
  `py::gil_scoped_release` is an access violation, not an exception. Both submit paths run
  without the GIL, so `unwrap` runs only after the GIL is back.

  0.20 found that `record_frame` had never obeyed this, and the shape of the miss is worth
  keeping. The rule was written down twice, in `require_same_context` ("the GIL is held
  here, which is what makes raising legal at all") and in `present_command_buffer` ("a
  diagnosis has to travel back as an Error and be raised by the caller"), and both comments
  sit within thirty lines of the function that broke it. `record_frame` raised on a failed
  `vkBeginCommandBuffer` or `vkEndCommandBuffer`, and both submit paths reach it with the
  GIL released. **A rule stated in a comment is not enforced by anything.** The audit that
  found it was not looking for it either: it was checking which functions were free of
  pybind so they could move to a header, and `record_frame` came back "not free" for a
  reason that turned out to be the bug. The general form: *when a refactor asks "does this
  code depend on layer X", a surprising yes is worth reading before it is worked around.*

- **A frame that is acquired and never submitted has to be given back** (0.20), and turning
  a crash into an exception is what made this reachable. `acquire()` resets the slot's
  in-flight fence, and only a submit signals it, so `present()` returning an error on a
  failed `record_frame` left a fence nobody would ever signal — and `acquire()` waits on it
  with `UINT64_MAX` three frames later. The 0.20 fix therefore replaced a crash with a
  correct `bz.OutOfMemoryError` **followed by a hang**, which is worse than the crash: the
  exception says the frame failed, and the traceback for the hang points at an unrelated
  line in the next iteration of the loop.

  `abandon_frame_()` submits nothing, waits on the acquire semaphore (the next
  `vkAcquireNextImageKHR` needs it unsignalled) and signals the fence, then recreates the
  swapchain, because Vulkan releases an acquired image on present or on swapchain
  destruction and this frame does neither. The lesson is the general one: **every early
  return between `acquire()` and `end_frame()` owes the slot the same bookkeeping
  `end_frame` does.** Untested, and honestly so — the trigger is an out-of-memory result
  from `vkBeginCommandBuffer`, which no test can provoke without an injected allocator.

  The sibling is **found and not fixed**: `end_frame` logs a failed `vkQueueSubmit` and
  presents anyway, which waits on a render-finished semaphore that submit was going to
  signal. The same fence is stranded, so the hang is identical. It is not the same one-line
  fix, because `advance_submit_serial()` has already reserved a value on the Context
  timeline that the failed submit was going to signal, and anything waiting for that serial
  is stranded too. Fixing it is a decision about whether a reserved serial can be signalled
  by hand or must never be reserved before a submit succeeds — pre-1.0, and it wants its own
  release rather than a patch at the end of this one.

---

## Technical debt register

The numbering is historical and stays. Every entry means "do it when it really hurts", not a
permanent ceiling.

1. ✅ **One live Context per process** — PAID in 0.15. `volkLoadDeviceTable` plus a
   per-Context table, and the device-level globals stay unloaded on purpose.
2. ✅ **The 1.2 path was untested in CI** — PAID in 0.15. A second lavapipe job runs with
   `BAZALT_FORCE_VULKAN_1_2=1`, and the same knob works locally on any driver.
3. ✅ **ResourceTracker: no SPIR-V reflection** — PAID in 0.19. `src/SpirvReflect.hpp`
   walks a module once and reports which `(set, binding)` pairs it writes, which
   execution models it declares, whether it prints, and whether a requested HLSL entry
   point resolved to an empty function. It bought four things: the automatic barrier for
   graphics SSBO and storage-image writes, `STORAGE_IMAGE` being tracked in the graphics
   path *at all*, compute narrowed from conservative to actual, and the two ceilings
   below turning into real behaviour.

   The entry was understated in one way and overstated in another, and both are worth
   keeping. Understated: this was not only "untracked". `set_storage_image` already
   recorded the image as resting in GENERAL while `track_draw_` never transitioned it, so
   the layout the descriptor promised and the layout the image was in disagreed — a
   validation error, not a slow path. Overstated: it listed **optional binding
   declarators** as something reflection buys. That is now rejected on purpose (see
   below).

   It also turned up that the feature was unreachable for a second reason nobody had
   noticed: a graphics shader writing a descriptor needs `fragmentStoresAndAtomics` or
   `vertexPipelineStoresAndAtomics`, so 0.17's declarator produced a write that worked on
   the GPU and a pipeline that failed to build. `Feature::FRAGMENT_STORES` and
   `Feature::VERTEX_STAGE_STORES` landed with the reflection, and the gate asks the
   reflection rather than the declarator: declaring a storage image and only reading it
   costs nothing.
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
5. **`supports_multiview()` is a second way to ask `supports(Feature)`** — found by the
   0.18 audit and deliberately left standing, because the honest fix is not a deletion.
   `FeatureInfo` maps each `Feature` to a plain `VkPhysicalDeviceFeatures` boolean through
   a pointer-to-member, and multiview lives in `VkPhysicalDeviceVulkan11Features`. So
   `Feature.MULTIVIEW` needs the table to grow a pNext column first — which **bindless
   already needs** (see Proposed features), so the two arrive together or not at all.
   Removing the method before then would lower the ceiling: there would be no way left to
   ask. Both `Context.supports_multiview` and `Device.supports_multiview` go when the
   enum entry lands. Target: with bindless, before 1.0.

   0.19 added four plain-feature rows (tessellation, geometry and the two graphics-store
   bits) and needed no column, which is the confirmation that the column really is only
   wanted by the pNext capabilities. It also found a third customer for it:
   `drawIndirectCount`, which is what a GPU-decided draw *count* needs. So the column now
   has three, and they still arrive together or not at all.

### Ceilings accepted on purpose

These are not debt to pay, they are limits we chose. Each one names its upgrade path.

- **`Error.hpp`'s `check()` puts the raw `VkResult` name in the message** (0.20). It is the
  template behind essentially every internal Vulkan failure, so `VK_ERROR_OUT_OF_POOL_MEMORY`
  reaches a Python user who cannot act on it. The 0.20 audit listed it as the highest-impact
  message in the codebase and it was left alone on purpose: for a driver-level failure that
  name is exactly what a bug report needs, and the alternative is either a translation table
  for a long enum (wrong in both directions, and written from memory) or dropping the one
  fact that identifies the failure. Upgrade path: none wanted. The messages worth improving
  are the ones bazalt writes itself, and 0.20 did those.

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
- ✅ **An HLSL entry point that matches no function compiles to an empty shader** (0.16) —
  CLOSED in 0.19. Reflection sees the empty body and no interface, so the typo is refused at
  the compile that caused it. Gated on HLSL with an explicit `entry_point`, because a GLSL
  `main()` that deliberately does nothing is legitimate and has no name to misspell.
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
- ✅ **A printf Context compiles every shader unoptimized** (0.18) — CLOSED in 0.19.
  Reflection reports whether a module imports `NonSemantic.DebugPrintf`. The ORDER is the
  whole trick and is easy to write backwards: `prints` can only be read off unoptimized
  words, because `-O` deletes the print that would prove it. So bazalt compiles at zero,
  reflects, and recompiles at performance when the shader turns out not to print. A
  printing shader is one compile; a quiet one in a printf Context is two, and it is no
  longer taxed for the Context it happens to be in.
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
- **Automatic barriers are computed at RECORD time from reflection** (0.19), so a hot
  reload that makes a shader start writing an already-bound resource does not re-barrier a
  recording that is only replayed. Much narrower than it sounds: `rebuild()` never
  recreates the descriptor layouts, so a reload cannot *add* a binding — the hazard is
  exactly "declared, bound, previously read-only, now written". And every example records
  inside the frame loop; `examples/12_hot_reload` builds a fresh CommandBuffer each frame,
  in the example whose subject IS hot reload. Nothing can close it automatically, because
  re-deriving the barriers means re-running user Python. Upgrade path: a generation counter
  on `ShaderModule` bumped by `replace()`, checked at `execute()` to warn.
- **The write scan fails open** (0.19). A descriptor handed to a function, one appearing in
  a pointer `OpPhi`, a decoration group, a malformed word, or SPIR-V bazalt did not compile
  are all treated as written. The tracker may be pessimistic, never optimistic: an
  uncertain module behaves exactly as bazalt did before reflection existed. Writes through
  `buffer_device_address` stay invisible, and no bazalt API can produce such a buffer.
  Upgrade path: a real def-use pass over call graphs, when a shader measurably pays for it.
- **Reflection is only trusted for SPIR-V bazalt compiled** (0.19). A `.spv` file or
  `compile_shader(source=bytes)` sets `writes_unknown`, so no barrier is narrowed for it.
  The first attempt at this was a capability allowlist, and that was the wrong mechanism:
  the numbers are a long enum, a list written from memory is wrong in both directions, and
  being wrong the SAFE way silently turns the whole feature off while looking like it
  works. Provenance is the thing the parser cannot see and the caller knows for certain.
  Upgrade path: teach the parser more write opcodes, and trust more of them.
- **A borrowed-image `RenderTarget` is single-sample** (0.19), because `create_image` has
  no `samples=`. Upgrade path: that kwarg, additive.
- **The indirect verbs take no `stride=`, and `count` is CPU-side** (0.19). A packed array
  is the one obvious way and `offset=` covers starting later in the buffer. A GPU-decided
  draw COUNT needs `vkCmdDrawIndirectCount` and `drawIndirectCount` in
  `VkPhysicalDeviceVulkan12Features` — the same pNext column debt #5 waits on. The shape to
  use instead is one command whose `instanceCount` a compute pass accumulates atomically,
  plus a compacted instance buffer (`examples/28_gpu_culling`).
- **The tracker orders uses within ONE recording** (0.19, and true since 0.6 — written down
  now because indirect draw made it easy to hit). Two CommandBuffers that share a
  GPU-written buffer are ordered by nothing the tracker can see, so the second recording
  needs a manual `cmd.barrier()`. `examples/28_gpu_culling` does exactly that, because the
  compute pass runs in one window's recording and the other window reads its output.
  Upgrade path: Context-level tracking of which serial last wrote a resource.

---

## Deferred until needed

Additive work with no assigned release. Pull each one in when it really hurts.

- ✅ Async headless submit — DONE in 0.18 (`ctx.submit(wait=False)` plus `ctx.wait()`).
- ✅ A per-level mip copy on a cross-Context transfer — DONE in 0.18.
- ✅ An async `StaticBuffer` — DONE in 0.18.0, together with an async
  `create_image(array)`. The entry here predicted the work correctly (the submit path
  did need buffer residency) and the trigger wrongly: what forced it was not a slow mesh
  loader but the API being unpredictable about which calls block. See "What blocks and
  what does not".
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
- ✅ **Tessellation and geometry stages** — DONE in 0.19. Three `ShaderStage` entries, not
  two: the tessellation pair plus geometry. The estimate missed the invasive part, which was
  not the stages but the *stage mask* — see the decision below.
- ✅ **A `RenderTarget` on images you already own** — DONE in 0.19. The open question
  resolved in favour: it passes rule 1 for the reason the fourth `create_image` overload
  does — it differs by the type of the argument, and "allocate attachments for me" is not a
  variant of "render into these", the same distinction that keeps `blit_image` out of
  `copy_image(scale=True)`. Ownership needed no work at all: `OffscreenTarget` already held
  attachments by `shared_ptr` and its destructor already destroyed only its own views.

**Performance:**

- ✅ **Pipeline cache**, in memory — DONE in 0.17. On disk at 1.0, when the frozen API gives
  a stable format.
- ✅ **Specialization constants** — DONE in 0.17.
- ✅ **Indirect draw and dispatch** — DONE in 0.19, which answers the `What 1.0 means`
  question by shipping rather than deferring. `Feature::MULTI_DRAW_INDIRECT` finally has an
  API behind it after sitting in the table since 0.5.
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
- ✅ **Window extras** — DONE in 0.19. Dropped files went into `PollState` and rotate with
  the key edges, because a drop is the same kind of thing they are: a change that expires
  with the poll cycle. The clipboard became free functions for the reason `poll_events()` is
  one.
- **Gamepad** (~90 lines) — axes and buttons from GLFW. The weakest ratio of value to API
  surface on this list.
- ✅ **An async `StaticBuffer`** — DONE in 0.18.0.

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
  it. 0.20 is the one exception and rule 5 now carries the reason and the test for repeating
  it. Note what 0.20 was NOT: it did not sweep up the open debt register. Debt #4 (the
  skipped sync-validation test) and #5 (`supports_multiview`) are still open, because both
  wait on something outside the release.

- **`.export_values()` on a scoped enum** (0.20). It binds every member a second time as a
  bare module attribute, so `bz.ShaderStage.VERTEX` had a twin `bz.VERTEX`. Fifteen of
  twenty-two enums had it and seven did not, which made even the inconsistency invisible:
  `bz.ALPHA` existed and `bz.RGBA8` did not. Two names collided and the later registration
  silently won — `bz.VERTEX` resolved to `ShaderStage` and `bz.FLOAT` to `VertexFormat`, so
  `BufferType.VERTEX` and `DataType.FLOAT` had no bare spelling at all. This is the "a second
  spelling is not a convenience, it is a fork" lesson arriving through a default nobody
  chose. The switch exists for unscoped C++98 enums; these are `enum class`.

  Adding it to the other seven was the alternative and it is the wrong direction: it doubles
  the fork instead of closing it, and it would put `LINEAR`, `NEVER`, `ALWAYS`, `KEEP` and
  `ZERO` on the top level of the package.
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
- **Optional binding declarators, inferred from SPIR-V** (0.19). Debt #3 listed these as
  something reflection would buy. Rejected once reflection existed, for two reasons. It is
  not merely a second way to declare a layout, it is a second way that *disagrees with the
  first at reload time*: `Pipeline::rebuild()` never recreates `layout_`/`desc_layouts_` on
  purpose, so reflected bindings would let a shader edit invalidate a live
  `VkDescriptorSetLayout` and every `DescriptorSet` allocated from it — leaving a choice
  between refusing reloads that change bindings (a worse ceiling than typing the
  declarator) and making descriptor-set lifetime reload-aware. And a declarator carries a
  `stage`, which reflection of one module cannot supply for a merged layout. So reflection
  reports what a shader WRITES and never what it declares.

- **Validating the reflected bindings against the builder's declarators** (0.19). It reads
  like a free diagnostic and is not: a shader may legally declare a descriptor the pipeline
  did not, as long as nothing dereferences it, so the check would let a hot reload reject an
  edit that runs fine. It is the binding-declarator idea wearing a diagnostic's coat.

- **A capability allowlist in the SPIR-V parser** (0.19). See the decision above: the safe
  direction of being wrong is the one that silently disables the feature, which is the worst
  property a safety mechanism can have. Provenance replaced it.

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
- ✅ Add the `KEY_*` and `MOUSE_*` constants to `__all__` in `bazalt/__init__.py` — DONE, and
  it had been done for several releases while this line still asked for it. Found by the 0.20
  audit. A checklist nothing tests is a checklist that drifts.
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

- **A second spelling is not a convenience, it is a fork** (0.18). The audit before
  0.18 found six, and each had the same origin: an ergonomic shortcut added next to the
  general form, both kept because removing either would break someone. `target.read_pixels()`
  was `target.color[0].read()` with the layer and mip choice removed, and its own binding
  comment said so. `target.mip(m)` was `target.layer(0, m)`, same constructor call.
  `window.should_close()` was `not window.is_open()`. Each was one line of C++ and a
  paragraph of documentation explaining which to prefer — the paragraph is the tell.

  The one that survived the same test is `renderer.read_pixels()`, because a screenshot
  is not a readback with different ergonomics: it needs `present(capture=True)` first,
  it reads a capture-time extent, and it swizzles whatever channel order the compositor
  picked. Different work, so a different verb — and once `target.read_pixels` was gone,
  the shared name stopped being ambiguous too.

- **A `with` block and an explicit begin/end pair are two paths unless the pair reaches
  somewhere the block cannot** (0.18). Both surviving pairs do, and for one reason: a
  `with` block cannot span a function boundary. A recording assembled from helpers —
  `begin_scene(cmd)` opening a target and a matching `end_scene(cmd)` closing it — has no
  block to put around it, and that shape is ordinary in prototyping code. Rule 2: the
  block is the path, the pair is the escape hatch.

  The audit first removed `cmd.begin_label`/`end_label` and kept
  `cmd.begin_rendering`/`end_rendering`, on the reasoning that "a label always opens and
  closes inside one recording". That reasoning was wrong, and instructively so: it is
  equally true of a render pass, so it did not separate the two cases at all. The real
  criterion was always the function boundary, and it applies to both. The labels came
  back the same day.

  **The lesson: a justification that would equally justify the opposite decision is not a
  justification.** Before cutting one of two things that look alike, state the rule and
  check it against the one you are keeping. If the rule convicts both, either cut both or
  cut neither.

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

- **A safety mechanism whose failure mode is "silently does nothing" is worse than none**
  (0.19). The rejected capability allowlist could be wrong in two directions. Wrong the
  dangerous way, it misses a write. Wrong the SAFE way, it flags a capability that is fine,
  every module becomes `writes_unknown`, every barrier stays conservative — and the whole
  feature is off while the tests still pass and the API still answers. Before writing a
  conservative fallback, ask what it looks like when the fallback fires for the wrong
  reason. If the answer is "exactly like working", find a different mechanism.

- **A plausible number is not a verified number** (0.19). `examples/28_gpu_culling` reported
  about 350 survivors out of 20,000 and that was accepted for a whole release, because it
  is the sort of number frustum culling produces. The frustum planes were being built from
  the view-projection matrix's COLUMNS — GLSL indexes a `mat4` by column, so `m[3] + m[0]`
  adds two columns and yields a plane that means nothing. Garbage planes reject almost
  everything, which reads as "culling is working very well". The fix was cheap; finding it
  needed the same test written independently on the CPU, and writing THAT exposed the bug
  twice, because the first version indexed pyglm's `to_list()` as rows and reproduced the
  wrong answer exactly. General form: for anything whose output is a count or a
  measurement, a second implementation is the referee. Eyeballing a plausible magnitude is
  not.

- **Verifying an example by starting it and watching for errors proves almost nothing**
  (0.19). Three real bugs in `27_drop_and_icon` and `28_gpu_culling` survived a clean
  20-second run each: the run never dropped a file, never pasted, and never pressed a key,
  so the paths under test never executed. An example's interactive paths have to be driven
  directly — call the function the key would call, against real data — and its visual claim
  has to be measured (a closed outline is "zero object pixels adjacent to background", not
  "looks right").

- **A comment that overstates a use case gets followed** (0.19). `set_cursor_position`'s
  comment called per-frame recentring "the case this exists for". That is the pattern for a
  HIDDEN cursor; with `CURSOR_DISABLED` the mode already hands out unbounded motion and
  recentres itself, so warping every frame cancels every frame's delta. Two examples were
  then written from that comment and both had a camera that stopped turning while the button
  was held. Documentation that names one pattern as *the* pattern is a design statement, and
  it will be obeyed.

- **Dropping a wait means auditing every reader, not only the one in the ticket.** Making
  `StaticBuffer` asynchronous is a two-line change at the create site and a race
  everywhere else. Two submits on one queue are ordered by nothing but a semaphore or a
  barrier, so `read_bytes` — whose own `vkQueueWaitIdle` looks like it covers everything
  — could copy out a buffer the fill had not written yet, on exactly the drivers that
  overlap submits. The recording path was already safe (the submit path waits on the
  serial); the readback path had to be told. General form: when a resource stops being
  ready on return, grep every function that reads it.

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
  sections, and every pybind lambda is a template instantiation, so this only grows. 0.20
  split the bindings into eight files, which is the actual answer to a growth curve a flag
  can only postpone. The flag stays: any one of the eight can still grow into it.

- **A binding that stores something must say so with `py::keep_alive`, and nothing checks
  that it did** (0.20). `SwapchainRenderer(window, ctx)` called
  `window.get_surface_provider()`, which hands back three lambdas capturing the raw
  `GLFWwindow*` and a pointer to the `Window`'s own resize flag, and the renderer kept them
  for its whole life. `del window` then left both pointers dangling and the next `present()`
  read freed memory. There was no `py::keep_alive` anywhere in the module.

  The general shape is worth more than the fix: **a C++ constructor taking `T&` and storing
  anything derived from it needs `py::keep_alive<1, N>`, and the C++ side gives no hint that
  it does.** `SurfaceProvider` is a value type holding `std::function`s, so it looks like it
  owns its contents. Look for what the lambdas captured, not at what the struct is.

  It is testable deterministically, which is not obvious: pybind's `keep_alive` stores a
  strong reference on the nurse, so `sys.getrefcount(window)` rising by exactly one after
  the renderer is built is the assertion. Dropping the window and presenting is the second
  half, and on its own it is only a probabilistic check — freed memory often still reads.

- **`offset + length > size` on unsigned operands is a bypass, not a bounds check** (0.20).
  Six of them, and the Python boundary hands the offset straight through, so
  `buffer.update(data, offset=2**64 - 10)` wrapped the sum to a small number, passed, and
  reached a `memcpy` through a wild pointer. `fits_within(offset, length, size)` in
  `Error.hpp` subtracts instead (`offset <= size && length <= size - offset`) and all six
  call it, so they cannot drift apart. Test the near-maximum offset, not just the
  one-past-the-end one — the latter passes on the broken form too.

  **Five was the count the first sweep found, and the sixth shipped in the same release
  it was fixed in.** `image.read(layer=)` is `base_layer + layers > array_layers_` on
  `uint32_t`, so it was the identical shape one grep away, and the sweep had gone looking
  for `offset` and `size` by name. The lesson is the grep, not the site: search the
  *arithmetic* (`+` on the left of a comparison against any limit), because the operands
  of this bug are called `base_layer` and `array_layers_` as readily as `offset` and
  `size`. It also shows what the bug costs when it is not a `memcpy`: the read returned
  layer 0's pixels and a validation ERROR, so a silently wrong array is the good case.
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
  `LoadImage`, `CreateSemaphore`, `near` and `far` collide with our names. 0.20 hit the same
  thing from the other side: a precompiled header that put volk before pybind11 broke
  `pybind11/numpy.h`, which calls `std::min` and has no parenthesis guard. **Include pybind11
  first.** `main.cpp` did, by accident, for twenty releases.
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

- **Running an example is not the same as exercising it** (0.19). Every new example in this
  release ran clean on the first try and three of them were still broken, because a
  non-interactive run never drops a file, never pastes, never holds a mouse button. What
  actually caught them: driving the interactive paths directly from a script (call what the
  key calls, against real data), and turning each visual claim into a number that can fail —
  "zero object pixels adjacent to background" for a closed outline, "pixel-identical with
  culling on and off" for the culled view, "the same count as the CPU computes" for the
  culling itself. Add the measurement, not another look.
- **The headless fallback** (no windowing extensions) still has no coverage. It is a separate
  path from the API version, which CI does cover since 0.15.
- **The windowed path is verified by running the examples, and that is load-bearing.** The
  0.17 depth/stencil layout bug existed only for a target whose depth is scratch — a window —
  and every headless test passed with it in place. Run the new example before calling a
  release done; "the suite is green" does not cover the half of the library that needs a
  surface.
