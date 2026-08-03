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

   **0.21 is the second exception, and it is a different one.** It has a feature —
   bindless — and then three more additions that share no subject with it: a
   GPU-decided draw count, gamepads, and MSAA into borrowed images. Rule 5's test
   ("does the work share a subject, or only a mood?") says no, so the exemption
   needs its own reason, and it is this: **0.21 is the release that empties the
   additive proposal list before the 1.0 freeze.** After it, every remaining item in
   this file is a scope rejection, an escape hatch waiting for a concrete
   integration, or 1.0 content itself. That is a one-time position in the sequence,
   not a precedent — the same exemption in 0.23 would be a mood release, because
   the list it clears would be one somebody had refilled.
   
   Two of the three also came free with the feature: `drawIndirectCount` and
   `Feature.MULTIVIEW` were both waiting on the pNext column that bindless forced,
   and the debt register had said so since 0.18.

   **0.20 is the first exception, and it is written down here so it stays one.** It has no
   feature. It splits the binding layer into eight translation units, fixes four bugs and
   removes one accidental API. What made it legitimate is that the work had no feature to
   attach to: a build-time restructuring is forced by the *size* of the code, not by
   anything a user asked for, and the four bugs were found by auditing the binding layer
   rather than by writing something new. Rule 5's target is a release that collects
   unrelated debt because the debt is old. This one collects work that is all the same
   subject. If a future release wants the same exemption, the test is that question: does
   the work share a subject, or only a mood?
   **0.25 IS an exception, it is the third one, and the reason is a decision rather
   than a discovery.** The release has a feature — **the window's input surface
   finishes**: the characters, the pointer shapes, the gamepad edges and the monitor
   enumeration are one subject, and after them no open backlog entry touches
   `Window.hpp`. Then it empties the rest of the additive list: sampleable MSAA,
   primitive restart, `stride=`, precise occlusion, per-face stencil, exclusive
   fullscreen, `ctx.record()` and the short descriptor verbs. Those share no subject
   with the input surface or with each other.

   The plan proposed the feature alone and argued the rest was a mood release. **The
   owner overruled it, and the argument is on the record because it is better than
   the rule as written:** most releases since 0.20 have been shaped this way, none of
   these items is a new idea that needed designing, and 0.25 is the last additive
   release before the freeze — so "wait for a release that shares your subject" means
   "wait past 1.0" for entries that have been priced and ready for three releases.
   Rule 5 exists so a session can execute a plan without filling its context, and
   that test was met: the whole release is one session.

   **What the rule keeps.** The release is still NAMED for its feature, and each
   entry that rode along says so in its own decision rather than being folded into an
   invented subject — the 0.24 precedent, used the way 0.24 asked. The test for the
   next release quoting this one is not "0.25 did it": it is whether the list being
   emptied was priced and waiting, or refilled by somebody who wanted a big diff.

   **0.24 is not an exception, and it is written down because it nearly was.** The plan
   called it "the seams" and argued that a cross-recording barrier, `close()`, and a
   sentinel split share the subject "boundaries bazalt left implicit". That reading is
   thin — rule 5's test asks whether the work shares a subject or only a mood, and
   "boundaries" is broad enough to cover almost anything. What actually holds the release
   together is narrower and worth stating plainly: **the feature is running bazalt without a
   window**, and the rest is the API review the release was asked for. The barrier fix has
   no connection to the notebook at all; it is here because it was bought separately. A
   future release quoting 0.24 as precedent should quote that sentence, not the subject.

6. **An API break is acceptable before 1.0, but you batch the breaks into one release.**
   This is also why 0.20 is not 0.19.1. `CHANGELOG.md` promises that a patch release never
   breaks the API, and removing `.export_values()` deletes about sixty names that were
   reachable. Undocumented is not the same as absent.
7. **A limit is only a limit if something holds it.** Every entry that says "bazalt does not
   do this" carries one of five verdicts — HARD, SCOPE, COST, STALE, UNASKED — defined under
   the ceilings below. The test that assigns them: **an entry with no way out is HARD or
   SCOPE; an entry with a way out and no named price is UNASKED, not a ceiling.** The 0.22
   audit added this rule because the file had been recording all five in one section called
   "accepted on purpose", which lent the authority of a decision to work nobody had done yet.
   A reader cannot tell those apart a year later, and an unlabelled entry forbids as loudly
   as a real limit.

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

- **The keyboard follows the gamepad: renamed, not translated** (0.23, ergonomics #5).
  `Key`, `MouseButton` and `CursorMode` are `enum class`es whose values ARE the GLFW
  ints, exactly as `GamepadButton` has been since 0.21, so the two cannot drift. The
  query methods keep their `int` signatures — a pybind enum converts through its value —
  which is what keeps every bare `KEY_*` int valid with zero aliasing code. Three naming
  decisions worth recording: the top-row digits are `D0..D9` (an identifier cannot start
  with a digit, and `NUM_*` would read as the keypad, which stays `KP_*`); the C++ member
  for the delete key is `DEL` because `<windows.h>` defines `DELETE` as a macro, while
  the Python name stays `DELETE` (the binding names it); and there is no `LAST` member,
  because `KEY_LAST` is GLFW's array-size sentinel, not a key — the int remains for
  anyone sizing an array.

- **A gamepad is a free function returning a snapshot** (0.21). `glfwGetGamepadState`
  takes a joystick id and no window, so a pad belongs to the process — the same
  fact that makes `poll_events()` and the clipboard free functions. `get_gamepad`
  returns a value, not a handle: the state is what the last poll left behind, and a
  snapshot cannot change halfway through the frame reading it. An empty slot is
  `None`, so "is one connected" needs no second verb.

  Three things the raw GLFW state does not decide, and bazalt does, all answering
  the same question — what did the HAND do?

  A trigger is normalized to 0..1, because GLFW reports -1 released and +1 pulled,
  which is the hardware talking; the sticks keep -1..1 because that IS the question
  there.

  A stick pushed UP reads +1, where GLFW reports -1. GLFW is right for its own
  reason: its Y is screen space, and down is positive there because that is what a
  cursor position means. A stick has no screen to agree with. **The cost is real
  and it is the interesting part of the decision:** `window.get_mouse_state().dy`
  and `GamepadAxis.LEFT_Y` now disagree about which way is positive, so a camera
  driven by both negates one of them. It would have had to negate one of them
  either way, and the version where the negation is in bazalt is the one where the
  library's own two answers can each be right about their own device. Found by
  someone holding a pad, which is the only way this class of thing is found.

  And `deadzone=` exists because a real stick reads 0.03 untouched: scaled rather
  than clipped so the value stays continuous across the edge, applied per axis (a
  square dead area, and nothing has asked for the difference), and to the sticks
  only, since a trigger rests at one end of its range.

  **Level state only, and that is a consequence of the 0.16 input decision rather
  than a shortcut.** The edge queries rotate on a per-window generation counter; a
  pad has no window to hang one on, and a process-wide edge would need exactly the
  global registry of live windows that 0.16 rejected.

  **That last paragraph was half right, and 0.25 shipped the edge anyway.** The
  reasoning it gives is about WINDOWS: a per-window counter needs a registry of live
  windows to drive it, and 0.16 refused to keep one. A pad needs no such registry,
  because the pads are a fixed set of sixteen slots — an array, not a list somebody
  has to maintain — and the counter it rotates on has been process-wide since 0.16.
  So `pad.was_button_pressed(...)` is a per-pad previous-state snapshot beside that
  array, rotated the first time each pad is read in a new cycle. The general form,
  and it is why the entry stays: **a limitation inherited from a neighbouring
  decision has to be re-derived, not re-read.** Third time this file has recorded
  that (after per-subresource layouts in 0.18 and MSAA into borrowed images in 0.21).

  **What the edge costs here that a key edge does not.** GLFW records a key press in
  a callback, so no cycle can lose one. It has no gamepad callback, so a pad edge is
  measured between two READINGS: a press and a release inside a cycle nobody read are
  both invisible. Documented rather than fixed, because fixing it means polling every
  pad from `poll_events()` whether or not the program owns a gamepad.

- **A character is not a key, and that is the whole reason `text_input()` exists**
  (0.25). `window.text_input()` returns what the user TYPED during the last poll
  cycle, as text. `is_key_pressed(Key.A)` reports a physical key, and between the two
  sit the keyboard layout, the shift state, AltGr, the dead keys and an IME — so no
  amount of key state reconstructs a character, which is why every text field in
  every toolkit reads a character stream instead. It rides the 0.16 rotation with the
  key edges and the dropped files, because it is the same kind of thing: a change
  that expires with the cycle.

  It returns `str`, not codepoints, and the conversion happens in the CALLBACK. GLFW
  hands out one Unicode codepoint per event and the rest of the library speaks UTF-8,
  so encoding at the only place a codepoint exists means neither the binding layer
  nor a C++ caller has to.

  **This is the one thing an overlay library cannot supply for itself**, which is
  what made it the release's feature rather than a nicety: `ImGuiIO::AddInputCharacter`
  has to be fed from exactly this callback, and until 0.25 a text field in a bazalt
  program was deaf.

- **The pointer's shape is a second question, not a fourth `CursorMode`** (0.25).
  `set_cursor_mode` answers "is the pointer visible, hidden or captured" and
  `set_cursor(shape)` answers "what does it draw" — two questions, two verbs, and no
  rules for combining them. Folding the ten shapes into `CursorMode` would have made
  a thirteen-value enum of which three values are exclusive with each other and ten
  are exclusive with each other, which is the `set_fullscreen` plus `set_decorated`
  mistake `WindowMode` was designed against.

  `Cursor`'s values ARE the GLFW ints, following `Key` and `GamepadButton`. The ten
  cursors are created once and shared by every window, because a `GLFWcursor` belongs
  to the process — the `poll_events()` line again — and the cache is cleared beside
  `glfwTerminate`, which destroys every remaining cursor and would otherwise leave the
  cache holding dangling pointers the moment the last window closes.

  A platform without a given shape gets the default arrow rather than an error. That
  is the `set_icon` contract — a request, not a guarantee — and the alternative is an
  error channel every caller must handle for a cosmetic detail.

- **`Device` is dead data** (0.14). `bz.list_devices()` builds a bare VkInstance,
  enumerates, destroys it and returns `list[Device]`. `Context(device=)` then matches on
  `deviceUUID` inside its own instance. `Device` holds neither `VkPhysicalDevice` nor
  `VkInstance`, so it has no affinity to any instance: it outlives any Context and cannot
  dangle. Instance ownership stays **invisible from Python**, so the inside of
  `list_devices()` can be rewritten without breaking a line of user code. The enumeration
  goes through `vkGetInstanceProcAddr`, not `volkLoadInstance`, because the second one would
  point the instance-level globals of a live Context at an instance that disappears a moment
  later.

- **A `Monitor` is the `Device` shape, and it keeps its handle** (0.25).
  `bz.list_monitors()` reports every display as inert data — name, position in the
  virtual desktop, current mode, physical size, content scale and every video mode —
  and `Window(monitor=)` / `set_mode(mode, monitor=, video_mode=)` take one. The
  backlog entry called this a kwarg and it is not: choosing needs the displays
  VISIBLE from Python, which is a new public type, and that is where the cost is.

  It departs from `Device` in exactly one place, and the reason is what makes the
  departure safe. A `Device` deliberately holds no `VkPhysicalDevice`, because that
  handle dies with the instance `list_devices` destroys a moment later — keeping it
  would be a guaranteed dangle. A `GLFWmonitor*` dies only when somebody unplugs that
  display, and every use re-scans `glfwGetMonitors` for the pointer before touching
  it. So the pointer is a KEY looked up in the live list, never something
  dereferenced on trust, and an unplugged display raises `WindowError` naming itself.

  **Matching by name was the alternative and it does not work.** The machine this was
  written on reports two monitors, both called "Generic PnP Monitor". A name is not
  an identity, which is the same lesson as "no string keys on an API whose primitive
  is a handle" (0.9) seen from the other end.

  **It is the only process-wide query that does NOT need a live Window.** The
  clipboard and the gamepads refuse without one, because GLFW comes up with the first
  Window and goes down with the last. Monitor enumeration cannot follow that rule and
  still be useful: choosing where to open the first window happens before there is
  one. So it initializes GLFW itself, the way `list_devices()` builds its own
  instance rather than borrowing a Context's, and nothing shuts it down again if no
  window is ever created — `glfwTerminate` is process-wide cleanup the OS does at
  exit anyway, and a second reference count beside `window_count_` would be machinery
  for a case nobody has.

  Both extras are refused on a mode that cannot mean them: `monitor=` needs a
  fullscreen mode (a windowed window is placed with `set_position`), and
  `video_mode=` needs `FULLSCREEN`, because `FULLSCREEN_WINDOWED` is DEFINED by
  leaving the display's mode alone. Refused rather than ignored — a call that quietly
  does half of what it says is the thing this file keeps finding a year later. And
  `set_mode` stopped being a no-op when the mode repeats: moving a fullscreen window
  to another display asks for the same mode and has real work to do.

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

- **The volk device table had two layouts, and nothing said so** (0.25). `VolkDeviceTable`
  declares every Win32-only entry point behind `#if defined(VK_USE_PLATFORM_WIN32_KHR)`.
  The macro was defined `PRIVATE` on the `_core` target and NOT on the `volk` target, so
  `volk.c` compiled a struct with fewer members than the one every bazalt file saw. That is
  a one-definition-rule violation: the two translation units disagreed about the offset of
  every field after the guarded ones, and `volkLoadDeviceTable` filled a shorter struct
  than the caller read.

  It had been true since Windows support existed and hurt nothing, because no bazalt code
  had ever named one of the guarded members. Exclusive fullscreen is the first, and it
  found the bug the only way a layout mismatch can be found — by crashing. **Fixed by
  defining the macro `PUBLIC` on the volk target instead**, so it reaches volk's own
  compilation AND everything that links it, and the two cannot drift apart again. The same
  line covers `VK_USE_PLATFORM_METAL_EXT` on macOS.

  **The general form is the 0.15 lesson from the other side.** That entry says the compiler
  cannot catch a missed call site, because the globals still exist as symbols; this one says
  the compiler cannot catch a mismatched struct either, because both spellings compile.
  What catches these is a call that has to go through the thing — which is the argument for
  writing the feature rather than only the column it needed.

  A null entry point is now checked before it is called, in the one place that has a single
  pointer rather than a whole table. A driver can legitimately leave one null when an
  extension is available but was not enabled at device creation, and the difference between
  a crash and a logged warning is one `if`.

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

### Descriptors, and the feature table that had to grow first

- **A compute shader can sample a texture** (0.21). `ComputePipelineBuilder` had
  `storage_image` and no `texture`, so an image reached a compute shader only as
  `imageLoad` at an integer coordinate — no filtering, no mip selection, no address
  mode. Nothing downstream was missing: `set_image` checks the binding type it
  finds in the layout, the descriptor pool counts samplers, and the tracker's
  `COMBINED_IMAGE_SAMPLER` branch takes whatever stage mask it is handed. The gap
  was one declarator, and it survived four releases because the compute examples
  all happened to want a storage image.

  General form: **an absent verb leaves no trace.** A wrong one is a bug report and
  a missing one is silence, so the way to find these is to read a builder's list of
  verbs against its sibling's rather than to wait for someone to miss it.

- **`FeatureInfo` has three columns, one per feature struct** (0.21). It mapped a
  `Feature` to a `VkBool32 VkPhysicalDeviceFeatures::*` and nothing else, so
  multiview, descriptor indexing and `drawIndirectCount` had no row to sit in —
  which is what debt #5 had been waiting for since 0.18. Three optional
  pointer-to-member columns of which exactly one is set, rather than a union or a
  variant: the table stays `constexpr`, stays an aggregate initializer, and
  `feature_info()`'s existing safe fallback still works. A row that sets none
  answers False forever, which is loud the first time anybody asks for it.

  `query_device_features` came with it and is the more valuable half.
  `Context::configure_features_` and `list_devices` each hand-built the same
  `VkPhysicalDeviceFeatures2` chain, which is two chances to ask a different
  question; the helper is one. It clears the `pNext` links on the way out, because
  a `Device` is copied into Python and a chain pointing at the members of a
  temporary is a dangling pointer wearing a struct.

- **A required pNext feature cannot gate device selection, and the reason is
  vk-bootstrap's** (0.21). `set_required_features_11/12` put a feature struct into
  the library's own chain, and `DeviceBuilder::build` then refuses a chain that
  also holds ours
  (`VkPhysicalDeviceFeatures2_in_pNext_chain_while_using_add_required_extension_features`).
  Moving every struct into vkb's chain would rewrite the 1.2/1.3 negotiation for a
  machine that has two GPUs of which only the non-preferred one has descriptor
  indexing. So a required pNext feature is diagnosed by name in
  `configure_features_` instead, and `select_physical_device_` carries the reason.

- **`Feature::BINDLESS` keys on `descriptorIndexing`, and then enables the bits one
  by one** (0.21). The spec says enabling the roll-up "does not imply the other
  minimum descriptor indexing features are also enabled", so `descriptorIndexing`
  is the right thing for *availability* — it is a promise about a set — and the
  wrong thing to stop at. `enable_descriptor_indexing` turns on what the device
  reports, the same enable-what-is-present shape `enable_features_if_present` has.

  This matters because the guarantee is uneven: `descriptorIndexing` promises
  partially-bound, runtime arrays, and non-uniform indexing plus update-after-bind
  for sampled images and storage buffers — but NOT non-uniform indexing for uniform
  buffers or storage images, and not update-after-bind for uniform buffers. So the
  layout code asks which bits actually stuck rather than assuming, and a device
  missing one of them still gets a working texture array.

- **`count=` is a kwarg on the declarator, not a second declarator** (0.21). A
  bindless texture and a plain one differ by how many there are, which is the
  definition of a variant (rule 1). It goes on all four declarators of both
  builders, because it is one parameter through one `add_binding` and leaving it
  off three of them would make "which declarator can be an array" a second rule
  to remember — the reasoning that gave DYNAMIC buffers the transfer bits in 0.18.

- **An array binding carries `PARTIALLY_BOUND`, and `UPDATE_AFTER_BIND` where the
  device has the bit** (0.21). A 500-slot array nobody fills is the normal case,
  and without the first flag reading the SET at all is undefined rather than
  reading an unwritten slot. The second is what makes rewriting a slot while an
  earlier frame still reads it legal
  (VUID-vkUpdateDescriptorSets-None-03047), and swapping a texture at run time is
  the prototyping case, so it has to be legal rather than usually working.

  **The default answers per shape and the caller can override it** (`update_after_bind=`,
  0.21). On for an array, off for a single descriptor, because that is what each is
  FOR: an array exists to be rewritten while the scene runs, and a plain binding is
  written once at setup in nearly every program. Turning it on for everything was
  the first version and it is wrong for the same reason the debug name stayed out
  of the sampler cache key in 0.16 — asking for BINDLESS would then change what an
  unrelated descriptor allocates. Turning it on for nothing but arrays is wrong
  too, and the user who asked for the knob is the evidence: a single texture
  swapped between frames is an ordinary thing to want, and it was unreachable.

  One boolean rather than a verb on the builder, because the question belongs to
  the binding: "may THIS descriptor be rewritten while a frame reads it". A
  pipeline-wide switch would make a bindless user remember to call it, and
  forgetting is undefined behaviour rather than an error.

  **Off is the cheaper side, and how much cheaper is not measured.** An
  update-after-bind descriptor counts against a separate limit
  (`maxPerStageDescriptorUpdateAfterBind*`, usually larger than the ordinary one)
  and some implementations place it differently. That is a reason for a knob and a
  default, not for a claim, and the comment in `binding_flags_for` says so rather
  than inventing a number.

- **`count > 1` requires the feature, which is stricter than the spec** (0.21). A
  fixed-size array indexed by a dynamically uniform expression is core Vulkan. It
  is refused anyway, because without descriptorIndexing an unwritten slot is
  undefined, an index that differs per invocation is undefined, and
  update-after-bind does not exist: the unguarded version is a texture array that
  works here and returns garbage on the next machine. `descriptorIndexing` is on
  every desktop driver since about 2018 and on lavapipe, so rule 3 holds, and the
  escape hatch is one binding per texture exactly as before.

  The gate lives in `build()` because that is the first place with a Context — the
  tessellation gate's shape, minus the `compile_shader` half, since a declarator
  has no SPIR-V to read.

- **Two declarations of one binding merge their stages and refuse to merge their
  counts** (0.21). Declaring a binding twice is how a resource read by two stages
  is spelled. Two different counts are not that: the layout holds one number and
  one of the two is wrong. The builder verbs chain and have no error channel, so
  the diagnosis is recorded and returned by `build()`.

- **A bound descriptor is identified by `(binding, index)`** (0.21).
  `DescriptorSet::bound_images_` and `buffers_` appended on every write, so writing
  one binding twice kept the first image alive for the set's whole life and made
  `track_descriptor_uses_` walk a list that only grew. Invisible while a binding
  held one descriptor, an unbounded leak the moment an array is rewritten per
  frame. The sampler moved into the entry at the same time, for the same reason:
  it belongs to that descriptor, so it is replaced when the descriptor is.

  General form, and it is the third time this file has recorded a version of it:
  **a container that only ever grows is a leak with a delay.** The size of the
  thing being appended is what decides how long the delay is.

### The 0.23 error taxonomy

- **Three recoverable kinds, split by what the caller fixes** (0.23, breaking #3 plus a
  wider move the census forced). `ResourceError` means "this resource or data cannot do
  that" — fix the data. `StateError` means "right call, wrong moment" (a double
  `acquire()`, a barrier inside a rendering scope, a CommandBuffer submitted twice) — fix
  the order. `UnsupportedError` means "this GPU cannot, with any argument" — ask
  `ctx.supports()` or take the escape hatch the message names. Siblings, not subclasses:
  an `except bz.ResourceError` written for data problems must not swallow the other two.
  The 0.20 rule (`ValueError` when the argument is wrong on its own) stays untouched
  underneath.

- **The capability gates moved out of `ShaderError` too** (0.23), which the plan's census
  did not list and the grep found. `polygon_mode` needing WIREFRAME, `count>1` needing
  BINDLESS, a tessellation stage needing its feature — all raised `ShaderError` because
  they happened to be found in a builder or a compile. A missing feature is a fact about
  the Context, not the file, and the compile-time and build-time gates go through one
  function, so both now raise `UnsupportedError` and cannot disagree. Pipeline
  DESCRIPTION errors (a stencil test against a target with no stencil, a missing fragment
  shader) stay `ShaderError`. Hot reload keeps its contract: a recompile failure is still
  `ShaderError`, and a feature gate cannot newly fail on reload because the feature set is
  fixed per Context.

- **A full descriptor pool is `ResourceError`, from either `VkResult`** (0.23).
  `VK_ERROR_OUT_OF_POOL_MEMORY` mapped to `OutOfMemoryError` and
  `VK_ERROR_FRAGMENTED_POOL` fell through to `ResourceError`, so one user mistake arrived
  as two types depending on the driver's mood. The fix for a full pool is a bigger pool
  (or the automatic one), not freeing memory, so `code_from_vk_result` sends both to
  `Resource`.

### Render targets and resources

- **The automatic descriptor pool grows blocks sized from the layout being served**
  (0.23, ergonomics #1). `create_descriptor_pool()` with no arguments is auto mode: a
  vector of `VkDescriptorPool` blocks, a new one whenever the current fills
  (`OUT_OF_POOL_MEMORY` / `FRAGMENTED_POOL`), each sized `max(default, this request)` per
  descriptor type with array `count=` included. Sizing from the layout is what satisfies
  MoltenVK, which refuses a pool that cannot hold a whole bindless array even when only
  some slots are written (0.22) — one mechanism, two reasons. Explicit sizes keep the old
  fixed single block as the escape hatch, byte-for-byte. The trap the plan predicted was
  real: a free must return to the block that allocated the set, and the `shared_ptr` alone
  cannot say which, so every `DescriptorSet` carries its `block_`. The pool kwarg renamed
  `samplers=` to `textures=` in the same pass (ergonomics #2): the builder's `.texture()`,
  the pool's `samplers=` and the set's `set_image()` looked like three names for
  `COMBINED_IMAGE_SAMPLER`, and the audit's answer is that only the first two were — 
  `set_image` names its ARGUMENT, and `test_stubs.py` already asserts `set_texture` is
  dead, so resurrecting that name would undo a recorded decision.

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

  **The one exception is a 3D slice target** (0.23). Vulkan tracks a volume's layout per
  mip with exactly one array layer, so `target.layer(z)` on a 3D attachment feeds `z` only
  into the VIEW (`baseArrayLayer` selects the slice of a `2D_ARRAY_COMPATIBLE` volume) and
  the barrier names layer 0. Feeding `z` into both — the 0.13 rule applied naively — would
  index past the layout state. Consequence, documented rather than fought: rendering one
  slice marks the whole mip, which is the granularity a volume's layout actually has, so a
  partial render followed by a sample is legal.

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

- **MSAA on a borrowed target allocates the multisampled attachments, and the
  borrowed images are the resolve targets** (0.21). `samples=` joins the signature
  that takes images, matching the one that allocates them and `SwapchainRenderer`.
  The images handed in become what `colors_`/`depth_` already are on the allocating
  path — the single-sample, sampleable resolve — so every `*_resolve_*` accessor
  works unchanged.

  **The ceiling's stated upgrade path was `create_image(samples=)`, and it was
  wrong.** Handing in a caller-owned multisampled image needs matching resolve
  images passed alongside it and every resolve accessor taught about them, and it
  puts a second MSAA idiom in the API: one where bazalt owns the multisampled image
  and one where the caller does. Second time a ceiling's recorded reason turned out
  to be worth re-deriving rather than re-reading, after per-subresource layouts in
  0.18. The lesson section already says so; this is the confirmation.

- **The samples are readable, and `keep_samples=` is what they cost** (0.25).
  `target.multisampled_color[i]` binds to a `sampler2DMS` for a custom resolve — the
  driver's resolve averages the samples and that average is the end of the road, so
  per-sample edge detection and a TAA history reject need them before it happens.

  **Three things stood in the way and only the first was the one the ceiling entry
  predicted.** `usage_for_image` stripped `SAMPLED` from every multisampled image;
  the entry called the image transient, and it is an ordinary device-local image, so
  the bit was the whole obstacle. The depth/stencil narrowing now runs FIRST in that
  function, because it is a hard Vulkan limit and the multisample rule must not hand
  `SAMPLED` back to a two-aspect view.

  **The second is the store-op, and it is the real cost.** A resolving pass used
  `STORE_OP_DONT_CARE` on the multisampled attachment, which is what makes MSAA cheap:
  on a tiled GPU — MoltenVK on Apple Silicon, which bazalt supports — the samples
  never leave tile memory. Storing them is the price of reading them, so the TARGET
  asks for it once and pays there. It cannot be derived: the recording that renders
  cannot see the recording that reads, because those are different command buffers
  and usually different frames.

  **The third is the layout, and it is not `final_layout()`.** `end_rendering`
  retired only the resolve image, so the multisampled one stayed in
  `COLOR_ATTACHMENT_OPTIMAL`. A kept attachment now retires to
  `SHADER_READ_ONLY_OPTIMAL` — deliberately not the target's final layout, because a
  swapchain's is `PRESENT_SRC_KHR` and no multisampled image is ever presented.
  `mark_rendered` records the same layout, so the barrier and the tracker cannot
  disagree; that is the 0.13 rule about the view and the barrier reading one source,
  applied to a third reader.

  **`multisampled_color` is empty without `keep_samples`, even though the images
  exist.** Handing out an attachment the pass discarded is handing out undefined
  contents. An empty tuple fails at the line that reads it; a subtly wrong picture
  fails nowhere, which is the failure this API is allowed to prevent cheaply.

  **The estimate said ~300 lines and named a `texture2DMS` declarator and "the
  tracker seeing a descriptor type it has not seen". Both were wrong**, and in the
  useful direction: `VkDescriptorSetLayoutBinding` records nothing about sample
  count, so a `sampler2DMS` binds through the same `COMBINED_IMAGE_SAMPLER` as a
  `sampler2D` and the layout says nothing about which is which — the shader's
  declaration decides. What the estimate missed instead was the store-op and the
  layout, which is the 0.19 corollary again: the invasive part sits next to the
  feature rather than in it.

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

- **A blend is stored as an equation, and a mode is four points in that space** (0.23).
  `BlendState` held a `BlendMode` and `color_blend_attachment_` switched on it at pipeline
  creation, so a preset was the only thing a pipeline could remember. It holds a
  `BlendEquation` now — six factors and two ops — and `blend_equation_for(mode)` is the one
  place a preset means anything. Downstream stops knowing which spelling produced it, which
  is what keeps the two from disagreeing: the test that pins it renders `src=ONE, dst=ONE`
  against `mode=ADDITIVE` and compares the pixels.

  **Three rules make the factor spelling complete rather than a diff against a hidden
  base.** `src=` and `dst=` are required together (half an equation only reads as a guess
  about the other half). `op=` defaults to ADD, because every named mode uses it. The alpha
  channel follows the colour unless `src_alpha=`/`dst_alpha=` spell it out — the
  `glBlendFunc` rule, and the alternative (alpha defaulting to ONE/ZERO) would silently
  destroy destination alpha for anyone writing a colour blend. Mixing `mode=` with any
  factor argument is refused, because both answer one question and the alternative is a
  silent winner between them.

  All three refusals are `ValueError` and live in the binding layer: nothing is consulted to
  know the call is malformed (the 0.20 line), and raising there points the traceback at the
  offending line instead of at `build()`. That is `require_preservable`'s address, not
  0.21's recorded-diagnosis one — the count-mismatch case needed `build()` because the
  mistake spans two calls, and this one has every argument in hand.

  **What stayed out, and why it is not a hole in the hatch:** constant-colour factors need a
  `blend_constants()` verb (and the dynamic state to go with it), and `SRC1_*` needs a
  second output declared in the shader. Each is more API than an enum row, so each remains
  its own proposal. `SRC_ALPHA_SATURATE` is in, because it is only a row.

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

- **A 3D texture is `depth=` on `create_image`, and a numpy volume is ndim 4** (0.23).
  Rule 1 both times. The empty overload takes `depth=n`; the array overload keeps every
  existing meaning — ndim 3 is still `(h, w, channels)` — and a volume is `(d, h, w, c)`,
  with a single-channel volume one `arr[..., None]` away. A `depth=` kwarg on the array
  overload was rejected because the shape already carries the fact, and a validation-only
  argument is the `create_image(array, cube=True)` defect the 0.22 review filed. The two
  error messages that teach the rule are the ndim gate and the unsupported-shape branch,
  because a raw `(64, 512, 512)` volume lands in the second with its depth read as
  channels. A volume has exactly one array layer (Vulkan's rule), so `depth=` refuses
  `layers=`, `cube=` and `samples>1` with the reason.

- **Render-to-slice is the borrowed-images target plus the existing `layer(z)`** (0.23).
  No new verb and no new kwarg: `layer()` already names the slice axis of an attachment,
  and on a volume that axis shrinks with the mip (level 1 of a depth-4 volume has 2
  slices), so the bound follows. The slice view is a 2D view of a 3D image — full Vulkan
  always allows it, MoltenVK does not — so target creation gates on
  `Feature::IMAGE_VIEW_2D_ON_3D` (the fourth portability row, enabled implicitly like the
  other three) and the refusal names the compute escape hatch. Rendering the WHOLE 3D
  target is refused at the binding layer: its main view is a 3D view, which Vulkan does not
  accept as an attachment, and the message says `target.layer(z)`. A depth attachment
  beside a 3D colour is refused as SCOPE — no use of a volume target has asked to z-test
  into a slice, and the guard is one sentence. `all_layers()` on a volume is refused too:
  multiview lights array layers, and a volume has one.

- **A barrier on a volume says `VK_REMAINING_ARRAY_LAYERS`** (0.23). The validation layers
  warn about a layer count of 1 on a `2D_ARRAY_COMPATIBLE` 3D image — with maintenance9
  that count will mean one Z slice — and the first example run printed ten warnings before
  drawing a frame. `Image::barrier_layers()` is the one place the choice lives; copy and
  blit REGIONS keep the real count, because `VkImageSubresourceLayers` does not accept the
  sentinel. `mark_subresource_contents` clamps the sentinel back to a real count, so the
  layout state never sees it.

- **Everything a Context owns comes from a `ctx.create_*` verb, and the two constructors
  that did not are GONE** (0.23, ergonomics #8). `bz.Context` and `bz.Window` are the roots
  — nothing owns them — and every resource under them is made by the Context: buffers,
  images, samplers, pools, pipelines, command buffers, and now render targets
  (`ctx.create_render_target`) and swapchain renderers (`ctx.create_renderer`). Remembering
  which of two conventions a type used was a coin flip; now there is no second convention.

  **The first attempt added the factory and kept the constructor, and that was the bug.**
  An alias is not a migration: it leaves both spellings alive, and this file's own 0.18
  audit says a second spelling of the same call is a fork, not a convenience — same work,
  same arguments, one paragraph of documentation explaining which to prefer, and the
  paragraph is the tell. These would have been the seventh and eighth. The constructors are
  removed instead, which is legal exactly once: 0.23 is the pre-1.0 break batch. The classes
  stay registered as TYPES, because `isinstance` and the annotations still name them.

  No `TargetType` enum: the two target forms differ by their arguments (formats plus a size,
  or images), which is the same overload pattern `create_image` uses, and an enum would
  duplicate what the arguments already say.

  **The renderer's `keep_alive` had to move, and it is the interesting part.** The
  constructor tied the Window to the new instance (`keep_alive<1, 2>`), because the
  `SurfaceProvider` captures the raw `GLFWwindow*` and a pointer to the Window's own resize
  flag — the 0.20 dangling-pointer bug. On a factory the nurse is the RETURN VALUE, so it is
  `keep_alive<0, 2>`: argument 1 is the Context, argument 2 the window. Getting that index
  wrong reintroduces exactly the bug that entry was written about, and nothing in the C++
  would say so.

- **`SubresourceTarget` and `MultiviewTarget` are registered, empty, and that is the whole
  feature** (0.23). The base is polymorphic, so pybind downcasts the existing `layer()` /
  `all_layers()` returns to the registered derived type by itself. Until then both came
  back as opaque `RenderTargetBase` and the stub could say nothing about them. No methods:
  each is a view that exists to be handed to `cmd.rendering(...)`, and the parent keeps
  every knob.

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

- **`compile_shader` is two overloads, and the language is an argument** (0.24). The old
  signature took `path` and `source=` together, and that made one parameter carry two
  unrelated meanings at once. With `source=` the file was never opened, yet its extension
  still chose the parser — so HLSL in a string needed an invented `.hlsl` filename, and a
  `.frag` file holding HLSL had no spelling at all. The project's own test suite wrote
  `compile_shader("bad.hlsl", stage, source=...)` against a file that does not exist,
  which is the tell: when the tests have to lie to the API, the API is asking the wrong
  question.

  `path` was doing four jobs — choose the language, name the file to open, tag
  `ShaderError.path`, anchor relative `#include`. Only the second is really about a path.
  So the split gives each job a parameter that admits to it: `path` (a file), `source=`
  (the code), `language=` (the parser), `name=` (what errors call it). The path form keeps
  extension inference, because a project whose files are named correctly should not have to
  say so, and keeps `#include` anchored to the file's own directory. The source form
  resolves includes through `include_dirs=`, which is the parameter named for exactly that.

  **`source` must be keyword-only.** A path and GLSL text are both `str`, so a positional
  string in the second overload could mean either. This is the same trap as the `bytes`
  ordering below and it gets the same answer: never let one Python type reach two meanings
  by position.

  **`language=` has to reach more than shaderc.** The extension used to feed three
  decisions — the parser, the `entry_point=` gate and the reflection's entry-point name —
  through one `const bool hlsl`. An override that only set the shaderc option would half
  apply, and `language=HLSL` with `entry_point=` on a `.frag` would be refused by the
  gate. The resolution happens once at the binding, and `compile_parts` takes the answer.
  Stored on the `ShaderModule` for the same reason `entry_point` is: a hot reload that
  re-inferred it would parse the file as the other language.

  **No `SPIRV` member on the enum.** SPIR-V is a compiled format, not a language, and it
  already has two spellings — a `.spv` path and bytes in `source=`. A third would be the
  fork rule broken for a value that answers a different question. `language=` on either
  raises `ValueError` rather than being ignored, because an argument that quietly does
  nothing is the failure the whole split exists to remove.

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

- **A shader carries the settings a recompile cannot re-derive** (0.16, extended in 0.24).
  `include_dirs`, `entry_point` and now `language` live on the `ShaderModule` next to `path`
  and `stage`, because the hot-reload watcher holds only the module. A recompile that
  dropped them would resolve a different include, compile a different function or parse the
  file as the other language, and the failure would read as a broken shader edit. General
  form: anything a compile depends on that is not in the file has to be stored with the
  result. `language` is the entry that proves the rule was worth writing down — it was
  added three releases later and the rule said where to put it.

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

  **And writing it down did not make it true** (0.21). One thread submitting in
  sequence gives submission order, not execution order: two submits on one queue may
  overlap unless one waits for the other, and the spec says so plainly. Every upload
  now waits on the previous upload's serial, which costs nothing real — they are
  issued by one thread already, and the work being ordered is a copy.

  **And ordering the submits was only half of it.** The upload state was one enum,
  so the worker submitting the FIRST of several queued updates flipped it to
  Submitted and `wait()` stopped waiting with the rest of the queue untouched. A
  flag cannot say "five more outstanding", so it is a count now, and the condition
  variable waits for zero. This is the same shape as the 0.18 entry two paragraphs
  down — "an asynchronous operation must mark its resource busy BEFORE queueing" —
  and it shows what that entry did not: **marking is not enough when several
  operations can be outstanding at once. The mark has to be a count.**

  It also explains why the six-update test passed for three releases and the CI
  failure looked like a reordering: with two updates the worker is usually only one
  behind, so the wrong answer and the right one differ by one frame. Two hundred
  updates make it 71 versus 200, which fails on any machine — **when a race test
  only sometimes loses, make the queue deeper rather than trusting the driver to
  lose it for you.**

  **And the chain had a hole where the threads meet.** `create_image(array)` has
  nothing to decode, so it submits its copy inline on the calling thread (0.18.0) —
  which the worker's own chain knows nothing about. The first `update` of such an
  image therefore raced the copy that created it, and losing meant the image kept
  its original pixels. A queued upload now waits on the maximum of the worker's
  last serial and the image's own, so "whichever thread submitted it" stops being a
  question. General form: **a chain is only as long as the set of submitters it
  knows about, and an optimization that moves one submit off the worker moves it
  out of the chain.**

  Three more things are worth keeping. **The promise was three copies of a submit block**,
  and only one of them went through `submit_one_shot`; the two inline ones kept
  `waitSemaphoreCount = 0` through two releases that both edited this file. **The
  test was right and passed anyway** for three releases, because the desktop driver
  it ran on serialized what the spec lets it overlap — lavapipe in CI is what failed
  it, which is the argument for running the suite somewhere that takes the freedom
  the spec gives. And it is "dropping a wait means auditing every reader" seen from
  the other end: **a guarantee about ORDER needs a mechanism, and a comment is not
  one.**

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

- **A GPU-decided draw COUNT is a kwarg on the two verbs that already exist**
  (0.21). `count_buffer=` and `count_offset=`, with the existing `count` becoming
  the maximum. Where the count comes from is one argument of a draw that already
  exists, and a `draw_indirect_count` name would need a copy of every future
  argument to `draw_indirect` — the reasoning that made `draw_indexed_instanced`
  disappear in 0.17, applied to a verb that had not been written yet.

  This is the case the 0.19 ceiling deferred, and the deferral was right at the
  time: the entry said it needed `drawIndirectCount` and the pNext column, and
  that is exactly what it waited for. The count buffer takes the same checks the
  argument buffer takes, because it is read by the same command processor at the
  same stage, and it goes through `track_indirect_` so a compute pass that wrote
  it is ordered against the read with no manual barrier.

  `examples/28_gpu_culling` keeps its accumulated `instanceCount` on purpose: one
  command plus a compacted instance buffer needs no feature bit, and a count
  buffer is for survivors that need DIFFERENT commands.

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

### Closing a Context, and who decides what that means (0.24)

- **`close()` stops the threads and does NOT destroy the device**, and the validation layers
  decided that rather than taste.

  The first version destroyed everything `~Context` destroys. The validation-as-assert
  fixture refused it in one line — `vkDestroyDevice(): VkDevice has 3 leaked objects` —
  because a resource the user still holds a name for is a live child of that device. From
  there the spec allows exactly two moves: free the resource out from under a live Python
  reference, or refuse to close while any name is bound. The first is a use-after-free with
  extra steps. The second makes `with` useless in a notebook, where names persist in the
  cell namespace by design — and the notebook is the whole reason for the feature.

  So the device follows the last `shared_ptr`, as it always did, and `close()` owns what the
  Context owns exclusively: the hot-reload watcher, the upload worker, and the GPU work in
  flight. **That contract is more honest than the bigger one, not smaller.** The threads are
  what accumulate across cell re-runs; the memory is freed when you stop holding it, which is
  the only moment at which freeing it could ever have been correct.

  A first attempt guarded `defer_destroy` so a resource outliving a closed Context became a
  no-op. That is the crash-safe version of the wrong semantics: it does not free the handles
  either, it only hides the leak from the destructor. Deleted.

- **A closed Context refuses reads that could still physically run.** Its device is alive, so
  `buffer.read()` after `close()` would work. It raises `StateError` anyway, because a
  "closed" that holds for some verbs and not others is two contracts, and the caller would
  have to learn which. Cached facts — `device_name`, `frames_in_flight`, `headless` — still
  answer: they are copies, and a closed Context that cannot say what it WAS is harder to
  debug rather than safer.

- **The guard sits at the binding layer, one call per verb, not in the headers.** Same
  address and the same argument as `require_same_context`: it catches a user error where the
  GIL is held, instead of threading an error channel through every factory. Resources reach
  it through `owner()`, and they need it — reading a render result outside the `with` block
  that produced it is the notebook's most natural version of this mistake.

- **`Window.close()` is deliberately absent.** The symmetry is tempting and the case is not
  there: a notebook on a remote server has no window, and a `Window` in a local notebook is
  already destroyed deterministically by `del`. The cost is real — some thirty methods that
  would all need a closed-check — and worse, a `SwapchainRenderer` captures the raw
  `GLFWwindow*`, which `py::keep_alive` currently makes impossible to outlive. `close()`
  would open exactly that hole. It comes back if somebody asks.

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

  The sibling had the same fence and one more failure on top of it: `end_frame` logged a
  failed `vkQueueSubmit` and **presented anyway**, so `vkQueuePresentKHR` waited on a
  render-finished semaphore that the failed submit was going to signal. It now skips the
  present and abandons the frame through the same helper.

  **The reason this looked like a bigger decision than it was is worth keeping, because the
  reasoning was wrong.** `advance_submit_serial()` reserves a value on the Context timeline
  under the queue lock, so a failed submit appears to strand it, and stranding a serial
  looks like it needs either a hand-written signal or a redesign that reserves only after
  success. Neither: a timeline signal must merely be **greater** than the current value, not
  the next value, and every wait in bazalt is `>= N`
  (`wait_for_serial`, `completed_submit_serial() >= upload_serial_`). So the next submit
  signals a higher number and satisfies every wait on the skipped one. A monotonic counter
  with `>=` waits absorbs a dropped reservation by construction. **Check what the
  synchronization primitive actually promises before pricing a fix around it** — the answer
  here turned a deferred release into four lines.

  What the failed-submit path deliberately does NOT do is raise. `acquire()` already logs
  and returns `False` on a lost frame, and the windowed path's contract is that a bad frame
  is skipped rather than fatal. A frame lost to a failed submit is logged as an ERROR and
  the loop continues, which is the same answer for the same shape of problem.

### Platforms, and what macOS costs (0.22)

Rule 3 says "run on more than 90% of machines", and until 0.22 the answer to "does bazalt run
on a Mac" was "nobody has tried". Apple Silicon is too large a slice to leave at that before
1.0, so 0.22 is the release that opens the platform. Three decisions carry it.

- **The Vulkan needed one line. The C++ needed a compiler.** `VK_USE_PLATFORM_METAL_EXT` for
  volk, the mirror of the WIN32 define, is the whole graphics-side cost. Everything a reader
  expects to find — `VK_KHR_portability_enumeration`, the
  `VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR` flag, `VK_KHR_portability_subset` on the
  device — **vk-bootstrap already does** (`VkBootstrap.cpp:782-790, 883-886, 1315-1323`), and
  it already picks the Metal surface extension for the same reason `create_instance_` does not
  name the Win32 one. The 1.3-or-1.2 negotiation from 0.15 was written with MoltenVK named in
  its comment and needed no change either.

  What did block the port was **C++23 arriving in AppleClang later than in MSVC and GCC**. The
  first macOS build produced twenty errors and exactly one was bazalt's: libc++ does not
  implement P0154R1, so `std::hardware_destructive_interference_size` does not exist there and
  `MpscQueue`'s padding did not compile. That one is now a per-architecture constant, which is
  a portability fix worth having anyway — the value only has to be at least a cache line.

  The other nineteen were the toolchain being old: `deducing this` carries the entire pipeline
  builder, and `std::jthread`, `std::stop_token`, `std::ranges::contains` and
  `std::ranges::fold_left` are spread over four more files. Rewriting those to a 2020 subset
  would be a large diff against features the library is built on, so the answer is the newest
  Xcode on the image instead. **Where a language feature is load-bearing, raise the compiler
  floor rather than write the feature out** — the alternative here was reimplementing
  `jthread` and un-deducing forty builder methods to support a compiler nobody has to use.

- **The wheel ships no loader and no MoltenVK.** Windows and Linux get a loader from the
  graphics driver; macOS supplies none, so a Mac user installs the LunarG SDK. Bundling
  MoltenVK inside the wheel would make `pip install bazalt` self-sufficient, and it was
  rejected for the same reason `CIBW_REPAIR_WHEEL_COMMAND_WINDOWS` is empty: a bundled loader
  shadows the one the user installed, and here it would also freeze a MoltenVK version inside
  a wheel that has no way to update it. The validation layers come from the SDK regardless, so
  a bundle would buy a working `import` and still not a working development setup. What the
  decision costs is one failure mode, and 0.22 pays it in the message rather than the
  documentation: `volkInitialize()` fails for exactly one reason, and `err_no_vulkan_loader()`
  now names the SDK instead of reporting `VK_ERROR_INITIALIZATION_FAILED`. Upgrade path:
  bundle it, if a user ever asks for a Mac install with no SDK — additive, and reversible.

- **arm64 and macOS 14, both forced rather than chosen.** GitHub retired its x86_64 macOS
  runners, so an Intel wheel could be cross-compiled and never tested, and an untested wheel
  is worse than no wheel. The deployment target is the more interesting one: libc++ annotates
  `<format>` (and the floating-point `std::to_chars` behind it) with availability, so a target
  below 13.3 fails to compile every binding file. `std::format` is in the error path of most
  of the library, so this is a hard floor rather than a preference — 14.0 is that floor with
  room. CMakeLists.txt refuses a lower target with one sentence, because the alternative is a
  page of "'format' is unavailable" landing on somebody who asked for `pip install bazalt`.

**What Metal does not have, and how each one is answered.** The first green run is the only
honest source for this list, and it splits into three kinds, which is the useful part:

| What | Kind | Answer |
| --- | --- | --- |
| Geometry shaders, `shaderFloat64`, wide lines | plain feature bits | already `Feature`, already skipped |
| Comparison samplers, sampler mip LOD bias, multisampled arrays | portability-subset bits | three new `Feature` rows (0.22) |
| `debugPrintfEXT`, timestamp deltas, occlusion of an empty pass, pool room for a whole array | driver behaviour with no bit to read | the tests probe and skip |

The middle row is the one worth the API. These are not exotic capabilities — a comparison
sampler IS shadow mapping — and full Vulkan has no feature bit for them because it never made
them a question. Only a portability driver can take them away, so the rows read the opposite
way from every other: **a device with no `VK_KHR_portability_subset` answers True without a
struct to read.** That inversion is why they could not be folded into the existing columns.

Two details cost more than the enum entries. Chaining
`VkPhysicalDevicePortabilitySubsetFeaturesKHR` at device creation is what makes the difference
between *reporting* the restriction and *lifting* it: an unlisted portability feature defaults
to off, so a driver that supports comparison samplers still refuses them if nobody asked. And
the struct lives in `vulkan_beta.h`, so `VK_ENABLE_BETA_EXTENSIONS` is a compile definition on
every platform, not only Apple — a struct that exists on one platform is how a header compiles
in CI and not on a contributor's machine.

The bottom row has no bit anywhere, so the tests read the driver's own answer: a timing that
is exactly zero on every frame, an occlusion query that counts an empty pass, a pool
allocation that is refused. Each skips with the observed value in the message. **A skip that
names what it saw survives a driver fixing the problem; one that names the driver does not.**

**What the port found in bazalt itself** is the argument for doing it. Once the wheel built,
the whole suite failed on macOS with "no suitable GPU found", because
`select_physical_device_` handed `PhysicalDeviceSelector` the version the *instance*
negotiated. A device may be older than its loader — on macOS it always is, since the LunarG
loader reports 1.4 and MoltenVK's device reports 1.2 — so bazalt refused every Mac before
looking at it. The fallback underneath was already right: `configure_features_` reads
`device_has_1_3` off the device and takes `VK_KHR_dynamic_rendering` when it must. One line
upstream was asking the wrong object, and no Windows or Linux machine could show it, because
there a 1.3 loader arrives with a 1.3 driver. **A version negotiated at one level is not a
fact about the level below it.**

The lesson the release generalizes: **a platform port is mostly the CI to prove it, and the
toolchain to survive it.** The failures came in this order: the installer URL, the archive
format, twenty compiler errors, then one real bug in bazalt. Only the last is a Vulkan
question, and the Vulkan part of the port really was one define.

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
5. ✅ **`supports_multiview()` is a second way to ask `supports(Feature)`** — PAID in
   0.21. `FeatureInfo` grew the pNext column bindless forced, `Feature::MULTIVIEW`
   got its row, and both `Context.supports_multiview` and
   `Device.supports_multiview` are gone. The entry predicted the shape exactly —
   "the two arrive together or not at all", and they did — and it named the third
   customer, `drawIndirectCount`, which shipped in the same release.

   What the entry got wrong is worth keeping: it assumed the column would arrive
   *with* bindless as a shared cost. It arrived **before** it, as the thing that had
   to exist first, which is why phase 1 of the release was a prerequisite nobody
   asked for and phase 2 was the feature. When a debt entry says "these arrive
   together", check whether one of them is actually the other's precondition — the
   ordering is the part that decides how a release is planned.

   The original entry follows, for the reasoning it recorded:

   *found by the 0.18 audit and deliberately left standing, because the honest fix is not a deletion.*
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
6. **Nothing runs the examples** — found by the 0.24 example sweep. 34 directories and
   about 6700 lines of the most-read code in the project, and the only thing CI executes
   is the README through `tests/test_readme.py`. The sweep found what that costs:
   `examples/12_hot_reload` had been raising `StateError` after 120 frames since the
   `gpu_time_ms` contract changed earlier in the same release, and two more examples
   guarded on a `None` that had become an exception. A test caught none of it, because
   no test opens these files.

   The honest fix is not "run every example in CI". Most need a window, several need a
   camera and a person, one needs a 1 GB scene download, and the frame loops do not
   terminate. What is testable is the setup half: every example builds its Context,
   compiles its shaders, builds its pipelines and allocates its descriptors before the
   first `while window.is_open()`. A harness that runs an example to that line on a
   headless Context and stops would have caught the `StateError` only if it read
   `gpu_time_ms`, which is in the loop — so even that is partial.

   Two cheaper things cover most of it and are not done either: import every example
   module under `BAZALT_FORCE_HEADLESS` with the loop guarded, and grep the examples for
   the calls a release changed, as part of the release checklist. The second is what this
   sweep did by hand. Target: before 1.0, because "the examples are the documentation" is
   only true while they run.

   **Estimate: ~300 lines** for the import harness — a test that walks `examples/`, imports
   each module under `BAZALT_FORCE_HEADLESS`, and fails on a raise. The cost is not the
   harness. It is that every example needs its frame loop behind a guard the harness can
   stop at, which is 34 small edits and a convention the next example has to follow.

### Ceilings accepted on purpose

**Audited in 0.22.** The section used to open with "these are not debt to pay, they are
limits we chose", and that sentence was true of about half of what it covered. The rest was
work nobody had needed yet, filed under a heading that made it read as settled. The audit
went through every entry against the code and split them by what actually holds each one:

- **HARD** — the spec, the driver or the platform forbids it. Bazalt has no move.
- **SCOPE** — possible, and deliberately outside the library. It must fail the scope test
  above, and the entry says on which of the two questions.
- **COST** — possible and in scope, not done because the price is real. The entry names the
  price and who pays it. These moved to `### Priced, not forbidden` below.
- **STALE** — the obstacle the entry named is gone, or was never there. Moved to the
  ergonomics backlog, with a one-line trace left here.
- **UNASKED** — possible, in scope, cheap, and the only reason is that nobody hit it. This
  is the honest name for "we don't do that". Also moved to the backlog.

Rule 7 is the test: no way out means HARD or SCOPE; a way out with no named price means
UNASKED. What follows are the HARD and SCOPE entries, plus the traces of what left.

- **HARD. `Error.hpp`'s `check()` puts the raw `VkResult` name in the message** (0.20). It is
  the template behind essentially every internal Vulkan failure, so
  `VK_ERROR_OUT_OF_POOL_MEMORY` reaches a Python user who cannot act on it. The 0.20 audit
  listed it as the highest-impact message in the codebase and it was left alone on purpose:
  for a driver-level failure that name is exactly what a bug report needs, and the
  alternative is either a translation table for a long enum (wrong in both directions, and
  written from memory) or dropping the one fact that identifies the failure. Upgrade path:
  none wanted, which is what makes this HARD rather than UNASKED — every alternative is worse
  than the thing being complained about. The messages worth improving are the ones bazalt
  writes itself, and 0.20 did those.

- ✅ **A cross-Context transfer regenerates the mip levels above 0** (0.15) — CLOSED in
  0.18. It copies the source's own levels now, one readback and one update per
  (layer, mip). Slower, and deliberately so: the overload is a setup step already.
- *Moved in 0.22: `recreate_swapchain` takes `vkDeviceWaitIdle` — COST, see `Priced, not
  forbidden`.*
- *Moved in 0.22: a cross-Context transfer goes through host memory — COST, see `Priced, not
  forbidden`.*
- *Moved in 0.22: `FULLSCREEN` is not exclusive fullscreen — UNASKED, to the backlog. The
  entry said the upgrade path was "a `Feature`, not a fifth `WindowMode`", which is an
  additive change with no price attached, and rule 7 calls that a backlog item. The
  observation itself stands and is documented on `WindowMode`: the swapchain stays
  composited.*
- *Moved in 0.22: fullscreen picks the monitor and nothing picks the video mode — UNASKED,
  to the backlog. The entry ended "deferred until somebody asks", which is the definition of
  the label. `src/Window.hpp:526` already enumerates the monitors and reads their video
  modes to pick the best-overlapping one, so what is missing is the user-facing choice, not
  the machinery.*
- ✅ **An HLSL entry point that matches no function compiles to an empty shader** (0.16) —
  CLOSED in 0.19. Reflection sees the empty body and no interface, so the typo is refused at
  the compile that caused it. Gated on HLSL with an explicit `entry_point`, because a GLSL
  `main()` that deliberately does nothing is legitimate and has no name to misspell.
- **HARD. `line_width` other than 1.0 needs `WIDE_LINES`** (0.16), because a driver may
  support exactly one width. Same shape as `polygon_mode` needing `WIREFRAME`: the rasterizer
  looks like free fixed-function state and is not.
- *Moved in 0.22: a preserved second pass re-transitions the attachment — COST, see `Priced,
  not forbidden`.*
- ✅ **Per-subresource layout tracking does not exist in `Image`** — CLOSED in 0.18. The
  entry called the gain small ("it only removes the loud edge of a partial render"), and
  that was the wrong reading: the loud edge WAS the feature, because it made rendering
  into one mip and then sampling the image impossible. See the decision above.
- *Moved in 0.22: a `DEPTH_STENCIL` attachment cannot be sampled or read back — COST, see
  `Priced, not forbidden`. The Vulkan half of the entry is real (a view carrying both aspects
  cannot be sampled); the conclusion was not, because the parallel-view trick the entry names
  as its upgrade path is already written for cubemaps in the same file.*
- *Moved in 0.22: the stencil state is the same for front and back faces — UNASKED, to the
  backlog. `src/Pipeline.hpp` writes the same struct into `.front` and `.back`, and the
  entry's own upgrade path is one additive kwarg.*
- *Moved in 0.22: `primitiveRestartEnable` is always FALSE — UNASKED, to the backlog. The
  entry's argument was "a restart index would change what the largest index value in an index
  buffer means, which is a decision, not a flag", and that is right about a global switch and
  wrong about the fix: restart is per-pipeline state, so an opt-in kwarg makes the meaning
  change the caller's own. `src/Pipeline.hpp:1644` hardcodes `VK_FALSE`.*
- **HARD. A specialized compute workgroup size needs Vulkan 1.3** (0.17).
  `layout(local_size_x_id = 0)` compiles to `OpExecutionMode LocalSizeId`, which requires
  `maintenance4`, and the baseline is 1.2. Specialize the numbers the shader reads instead.
  Upgrade path: none needed — it becomes available by itself as the baseline moves.
- ✅ **`copy_image` copies mip 0 only** (0.17) — CLOSED in 0.18. The case came up:
  `image.update(mip=)` makes per-level content ordinary, and a copy that leaves the other
  levels stale is a copy of the top level rather than of the image.
- *Moved in 0.22: a pipeline cache lives and dies with its Context — COST, see `Priced, not
  forbidden`.*
- **HARD. Shader printf needs the validation layers** (0.18). It is a layer service, not a
  driver one, so it is unavailable in a release run by construction. Upgrade path: none —
  this is what the feature IS.
- ✅ **A printf Context compiles every shader unoptimized** (0.18) — CLOSED in 0.19.
  Reflection reports whether a module imports `NonSemantic.DebugPrintf`. The ORDER is the
  whole trick and is easy to write backwards: `prints` can only be read off unoptimized
  words, because `-O` deletes the print that would prove it. So bazalt compiles at zero,
  reflects, and recompiles at performance when the shader turns out not to print. A
  printing shader is one compile; a quiet one in a printf Context is two, and it is no
  longer taxed for the Context it happens to be in.
- *Moved in 0.22: an occlusion count is not precise — UNASKED, to the backlog. The code
  comment at `src/CommandBuffer.hpp:1363` gives the real argument, which the entry did not:
  asking for a count bazalt cannot promise on every device would make `samples` mean two
  different things depending on the driver. That argument is exactly what `Feature` exists to
  answer, and 0.22 added three rows to it in one release, so "a `Feature`, additive" is a
  backlog item rather than a limit.*
- **SCOPE. `blit_image` covers mip 0 of every shared layer** (0.18). A blit chain between two
  images is a different question, and `generate_mipmaps` answers it on the destination. The
  capability is reachable, so what this entry defends is rule 1 rather than a limit: the
  price of lifting it is a second idiom for something that already has one.
- **SCOPE. `image.update` writes one (layer, mip) per call** (0.18). A layered update is N
  calls; they are queued on one worker and land in order, so the result is the same and the
  API stays one verb. Same shape as the entry above — reachable, and the cost of a shortcut
  is a second spelling.
- **HARD. A screenshot needs `present(capture=True)` first** (0.18). Not an ergonomic choice:
  a presentable image may only be touched between acquire and present. Upgrade path: none
  that Vulkan permits.
- *Moved in 0.22: a cross-Context transfer of a mipped image is O(layers x mips) readbacks —
  COST, see `Priced, not forbidden`.*
- **HARD. Automatic barriers are computed at RECORD time from reflection** (0.19), so a hot
  reload that makes a shader start writing an already-bound resource does not re-barrier a
  recording that is only replayed. Much narrower than it sounds: `rebuild()` never
  recreates the descriptor layouts, so a reload cannot *add* a binding — the hazard is
  exactly "declared, bound, previously read-only, now written". And every example records
  inside the frame loop; `examples/12_hot_reload` builds a fresh CommandBuffer each frame,
  in the example whose subject IS hot reload. Nothing can close it automatically, because
  re-deriving the barriers means re-running user Python. Upgrade path: a generation counter
  on `ShaderModule` bumped by `replace()`, checked at `execute()` to warn.
- **HARD. The write scan fails open** (0.19), and the direction is the point rather than the
  precision. A descriptor handed to a function, one appearing in
  a pointer `OpPhi`, a decoration group, a malformed word, or SPIR-V bazalt did not compile
  are all treated as written. The tracker may be pessimistic, never optimistic: an
  uncertain module behaves exactly as bazalt did before reflection existed. Writes through
  `buffer_device_address` stay invisible, and no bazalt API can produce such a buffer.
  Upgrade path: a real def-use pass over call graphs, when a shader measurably pays for it.
- **HARD. Reflection is only trusted for SPIR-V bazalt compiled** (0.19). A `.spv` file or
  `compile_shader(source=bytes)` sets `writes_unknown`, so no barrier is narrowed for it.
  The first attempt at this was a capability allowlist, and that was the wrong mechanism:
  the numbers are a long enum, a list written from memory is wrong in both directions, and
  being wrong the SAFE way silently turns the whole feature off while looking like it
  works. Provenance is the thing the parser cannot see and the caller knows for certain.
  Upgrade path: teach the parser more write opcodes, and trust more of them.
- ✅ **A borrowed-image `RenderTarget` is single-sample** (0.19) — CLOSED in 0.21,
  and not by the `create_image(samples=)` the entry named. See the decision above:
  `samples=` on the borrowed signature reuses the MSAA machinery that already
  existed, and the entry's upgrade path would have added a second MSAA idiom.
- *Moved in 0.22: a multisampled attachment cannot be sampled — UNASKED, to the backlog. The
  entry ended "additive, and nothing asks for it yet", which is the label's definition.
  `src/Image.hpp:576` drops `SAMPLED` for `samples > 1` and the comment beside it says why:
  "SAMPLED/TRANSFER are dead weight (you sample the resolve)". True of render-and-resolve
  MSAA and not of `sampler2DMS`, which is legal Vulkan needing only that bit — so the holder
  here is a default, not the API.*
- ✅ **The indirect verbs' `count` is CPU-side** (0.19) — CLOSED in 0.21 by
  `count_buffer=`, once the pNext column made `drawIndirectCount` nameable. The
  entry's own prediction held: it waited on exactly the thing it said it waited on.
- *Moved in 0.22: the indirect verbs take no `stride=` — UNASKED, to the backlog. "A packed
  array is the one obvious way" holds for the common case and stops holding the moment the
  draw arguments are interleaved with per-draw data, which is ordinary GPU-driven practice.
  `src/CommandBuffer.hpp:699` passes `sizeof(VkDrawIndirectCommand)` as a constant.*
- *Moved in 0.22: the record-time descriptor walk is per descriptor — COST, see `Priced, not
  forbidden`.*
- **HARD. A descriptor is rewritable mid-flight only where somebody said so** (0.21).
  The default is on for an array and off for a single descriptor, and
  `update_after_bind=` overrides either. Rewriting a descriptor that has neither
  while a submit reads it is undefined, and the layers report it. Not a ceiling to
  raise: the alternative is the flag everywhere, which spends the separate limit
  budget on descriptors nobody rewrites.
- *Moved in 0.22: a gamepad has no edge queries — **STALE**, to the backlog. The entry said
  the rotation "needs somewhere process-wide to keep the generation" and pointed at the 0.16
  rejection of a registry of live windows. That obstacle does not exist and did not exist
  when the entry was written: `poll_generation_` is a `static inline std::atomic<uint64_t>`
  on `Window` (`src/Window.hpp:80`), which is process-wide already and is the exact counter
  the entry went looking for. What 0.16 rejected was a registry of window objects, which is a
  different thing. This is the entry that motivated the audit.*
- *Moved in 0.22: the tracker orders uses within ONE recording — COST, see `Priced, not
  forbidden`.*
- **HARD. macOS needs the Vulkan SDK, and the wheel does not carry it** (0.22). See the
  decision above. The platform ships no loader, and bundling one is refused for a stated
  reason rather than an unexamined one: a bundled loader hides the one the user installed.
  Upgrade path: bundle MoltenVK and a loader, which is additive and reversible, if somebody
  asks for a Mac install with no SDK.
- **HARD. Shader printf needs a driver whose compiler implements it** (0.22), a second
  requirement beside "needs the validation layers" from 0.18. MoltenVK advertises
  `VK_KHR_shader_non_semantic_info`, accepts the SPIR-V, and then fails in the Metal compiler
  with "use of undeclared identifier 'debugPrintfEXT'". Nothing can be asked in advance —
  there is no bit for "my shader compiler implements this instruction" — so the failure is a
  pipeline build error and the tests probe for it. Upgrade path: none available to bazalt.
- **HARD. A GPU timestamp may be advertised and useless** (0.22). Bazalt already checks
  `timestampPeriod` and `timestampValidBits` before offering `gpu_time_ms` at all, and
  MoltenVK on a paravirtual device passes both and then writes the same value twice. There is
  nothing left to ask, so `gpu_time_ms` returns 0.0 rather than None. Upgrade path: none — a
  library cannot tell "fast" from "not measured" without a third fact the driver does not
  supply.
- **HARD. The macOS wheels are arm64 and macOS 14** (0.22). Intel because GitHub retired the
  x86_64 runners and an untested wheel is worse than none. 14.0 because libc++ carries
  `<format>` behind an availability annotation and the binding layer formats everywhere.
  Upgrade path for the first: a cross-compiled wheel plus somebody with the hardware to
  test it. For the second: none wanted — it moves by itself as the floor rises.
- **HARD. API coverage counts a symbol per NAME, not per (class, name), for the half it
  cannot call** (0.22). Methods, properties and functions are measured exactly, by wrapping
  them.
  Enum members and the key constants are read rather than called, so nothing can wrap them
  and the report matches their bare name against the identifiers in the test sources. Two
  enums that share a member name share an answer. Upgrade path: none worth it — the report
  exists to say what to test next, and it is exact for everything that is called.
- **HARD. A symbol used only by an example counts as untouched** (0.22). Deliberate: the
  examples are not run by CI, so they prove nothing about a symbol still working. It is also
  why the untouched list is longer than it looks — 124 of the 127 key constants are in it,
  and `examples/21_window_modes` presses a good many of them.

### Priced, not forbidden

The COST half of the 0.22 audit. Every entry here is possible and in scope, and each one
names what it costs and who pays. That is the difference between this section and the one
above: these are for sale, and the price is why nobody has bought yet. An entry that loses
its price stops belonging here and becomes backlog.

- **`recreate_swapchain` takes `vkDeviceWaitIdle`** (0.14), so a resize of one window
  stutters the other for a moment. It is correct, it only stutters. **Price of lifting it:**
  narrowing the wait to one swapchain means tracking which submits touched which swapchain,
  which is per-window state the Renderer does not keep today. **Paid by:** anyone with two
  windows on one Context, at every resize. **Estimate: ~350 lines**, most of it the new
  per-window submit bookkeeping rather than the resize path itself.
- **A cross-Context transfer goes through host memory and blocks the source queue** (0.15).
  **Price:** `external_memory` — `VK_KHR_external_memory_win32` and `_fd` are two spellings
  of one idea plus a platform split in the build, and the entry in `Proposed features` files
  the same work under raw-handle interop. So the honest statement is not "no portable
  alternative" but "one escape hatch buys this and three other integrations, and nobody has
  asked for any of them yet". **Paid by:** multi-GPU setups, once, at setup time.
  **Estimate: ~600-900 lines**, the largest single item left anywhere in this file — two
  platform spellings, a split in the build, and both sides of the handle. It is also the item
  with four other customers waiting on it (see raw-handle interop), so it is the one whose
  cost is worth paying least often and covering most.
- **A preserved second pass re-transitions the attachment** (0.16). Pass 1 retires the image
  to `final_layout()` and pass 2 brings it back, so N passes cost N round trips instead of
  staying in `COLOR_ATTACHMENT_OPTIMAL`. **Price:** a recording-wide look-ahead — the same
  machinery a depth store-op wants — against the current design where each pass is decided in
  isolation. **Paid by:** any multi-pass recording on one target, per frame.
  **Estimate: ~500 lines, and the spread is design risk rather than typing.** Every other
  entry here adds machinery beside what exists; this one changes an invariant — that a pass
  decides its own transitions with no knowledge of the next — which is the assumption the
  RenderTarget contract is written on.
- **A `DEPTH_STENCIL` attachment cannot be sampled or read back** (0.17). Its view carries
  both aspects, and Vulkan forbids sampling through such a view — that half is HARD. The
  conclusion is not: **price** is a second, depth-only view beside the attachment one, plus
  deciding which view `set_image` hands out. The shape already exists in the same file — a
  cubemap keeps a parallel `2D_ARRAY` storage view beside its `CUBE` sampling view
  (`src/Image.hpp:738`) — so this is a known pattern applied twice, not new machinery.
  **Paid by:** anyone wanting a stencil-carrying shadow map, who today needs two targets.
  **Estimate: ~300 lines**, and the number is low precisely because the cubemap already
  answered the hard question — which view a given verb hands out.
- **A pipeline cache lives and dies with its Context** (0.17). **Price:** an on-disk format is
  only worth writing against a frozen API, so this one is genuinely waiting on 1.0 rather
  than on effort. **Paid by:** startup time, every run. **Estimate: ~300 lines**, including
  the header validation a cache blob needs before it is trusted — vendor and device UUID,
  driver version — because a stale blob from another GPU is the failure mode this feature
  has.
- **A cross-Context transfer of a mipped image is O(layers x mips) readbacks** (0.18).
  **Price:** a single readback of every level needs a staging layout the host side does not
  describe today. **Paid by:** setup time on a mipped cross-GPU transfer.
  **Estimate: ~250 lines.**
- **The record-time descriptor walk is per descriptor, not per binding** (0.21). A draw asks
  the tracker about every bound descriptor, so a 500-texture array is 500 hash lookups — at
  RECORD time only, since replay costs nothing, and a sampled image the tracker never saw
  stops at `tracker_.tracks()`. **Price:** skipping the walk above some declared count means
  requiring a manual barrier there, which is a worse API. **Paid by:** nobody measured yet,
  which is the entry's real status. **Estimate: zero lines of feature code**, and it is the
  only entry in this file where that is the answer. The work is a benchmark: record a draw
  against a large descriptor array, and see whether the walk shows up at all. Until that
  number exists there is nothing to design, and an optimization written before it would be
  guessing at both the problem and the fix.
- ✅ **The tracker orders uses within ONE recording** (0.19, and true since 0.6) — CLOSED in
  0.24, and **the price this entry named was wrong in both directions.** It is worth keeping
  for that, because the error is the reusable part.

  The entry said the fix costs "Context-level tracking of which serial last wrote each
  resource". It costs no Context state at all. Order independence comes from the floor being
  unconditional: if the first READ of a buffer in a recording always synchronizes against
  everything that could have written it, no record-time knowledge of the other recording is
  needed, and record order does not have to match submit order. `use_image` had been doing
  exactly this since 0.6 and `use` never did — the whole bug was that asymmetry.

  It was also wider than the entry claimed. Writes were already covered: `execute()` emits a
  replay wrap-around barrier at the top of any recording where `tracked_writes_` is set. What
  nobody noticed is that the flag is per RECORDING, not per buffer, so the uncovered case is
  a recording that writes *nothing the tracker sees* — an ordinary draw, whose only writes
  are attachments. Reads, in other words. So the floor fires on reads only, and a writing
  recording pays nothing new.

  **Two lessons, and the second is the sharper one.** A price nobody has tried to pay drifts:
  this one was written from the shape of the problem rather than from the code, and the code
  already had the mechanism. And **the first test written for it passed before the fix** —
  it used two writing recordings, which the wrap-around barrier already ordered. A test that
  passes against the unfixed build is not a weak test, it is a test of something else; the
  fix only became real when the test failed first. See `run_cross_recording_case` in
  `tests/test_barriers.py`, whose negative control exists for the same reason.

  Cost of the floor: one `vkCmdPipelineBarrier` per distinct buffer READ first in a
  recording, and the source mask is narrowed by `BufferType`. Every type carries
  `TRANSFER_DST`, so `copy_buffer` and `fill_buffer` reach any of them — the narrowing that
  looked obvious, gating the floor on `STORAGE`, is unsound for exactly that reason — but
  only `STORAGE` carries `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`. So a vertex, index or uniform
  buffer waits on `TRANSFER` alone, which is idle in almost every frame.

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
- ✅ **Bindless / descriptor arrays** — DONE in 0.21, in the shape the entry
  predicted (`count=` on the declarator, `index=` on `set_image`) and needing the
  column it predicted. The estimate of ~630 lines was close for the feature and
  missed two things beside it, which is the pattern 0.19 warned about: the column
  was a prerequisite rather than a shared cost, and the descriptor set's own
  bookkeeping had a leak that only an array makes visible.
- ✅ **A GPU-decided draw count** — DONE in 0.21 (`count_buffer=`), the third
  customer of the same column.

**Escape hatches and integration:**

- **Raw-handle interop / `external_memory`** — raw access to the Vulkan handles
  (`VkDevice`, `VkImage`, `VkBuffer`, `VkCommandBuffer`) for C++ ImGui with no copy, CUDA
  interop, a video decoder and OpenXR. Rule 2, and it fits several libraries at once. YAGNI
  until a concrete integration asks for it.

  **A fourth customer, and it changes the shape** (0.25). A vendor upscaler — DLSS through
  NGX or Streamline, XeSS — is the same request as the other three plus one thing none of
  them needs: the SDK computes which instance and device extensions it requires
  (`NVSDK_NGX_VULKAN_GetFeature*ExtensionRequirements`) and the answer must reach
  `create_instance_` and `create_device_` *before* either runs. So the entry is not only
  "hand out the handles after the fact". Half of it already exists —
  `Context(raw_extensions=)` is exactly the hook, and it was written as an escape hatch for
  this class of caller. What is missing is the handles themselves. The upscaler is rejected
  as a bazalt feature for reasons of its own (see `Rejected, and why`); it is listed here
  because it is the customer that says what the interop entry has to cover.
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
- ✅ **Gamepad** — DONE in 0.21. The entry called it the weakest ratio of value to
  API surface on the list and the estimate of ~90 lines was right, but the two
  decisions worth having are ones the estimate did not see: normalizing the
  triggers and where a deadzone may be applied. A thin wrapper is not the same as a
  wrapper with no decisions in it.
- ✅ **An async `StaticBuffer`** — DONE in 0.18.0.

---

## The ergonomics backlog

Two inputs, one list. The first is an API review done in 0.22 from the outside — someone
reading the stub and the examples as a graphics programmer who has not seen the internals.
The second is the audit above: every ceiling that turned out to be STALE or UNASKED lands
here, because that is what those labels mean.

This is deliberately **not** part of `Proposed features`, which opens by promising that
everything in it is additive. Three entries here are not.

Each entry says: what you write today, what you should write, whether it breaks the API and
how it avoids breaking it if it can, where the code is, and why. The last part is the point —
rule 7 exists because a list of complaints with no reasons decays into a list of orders.

Nothing here is assigned to a release. The one scheduling fact is the split below: **three
entries are breaking, and only those three are bounded by the 1.0 freeze.** Everything else
is additive and can land before or after it. The COST entries in `Priced, not forbidden`
above are also work, and are listed there rather than here because their price is the
interesting part.

**Every open entry carries a line estimate (0.25).** They are measured rather than guessed,
against what comparable features actually cost: the gamepad wrapper 519, 3D textures 618, the
blend escape hatch 530, the `Key`/`MouseButton` enums 491, `ctx.close()` 352, render-to-slice
229, and `BlendMode.MULTIPLY` plus the integer vertex formats 118 for the pair. An estimate
covers C++, the binding layer, the stub and the tests — **not** the example and not the docs,
for a reason the same measurement gives: 0.23's features sum to 2647 lines and the release
diff is 4090, so **about 1400 lines of every release are examples, changelog and docs before
any feature is written.** Against release diffs of 2806 (0.21), 4090 (0.23) and 4017 (0.24),
that leaves roughly 1800-2600 lines of feature code per minor. Anyone planning a release
should subtract the 1400 first; the number is remarkably stable and it is the thing that
makes a backlog look shorter than it is.

The estimates below total roughly 2500 lines for the additive features and 450 for the
ergonomics entries, plus 2300-2600 for the COST entries above and about 700 for the two open
1.0 items (the example harness, debt #6, and splitting the enum row). So the open backlog is
on the order of **three to four releases**, and the spread on the COST half is design risk
rather than typing.

### Breaking, and therefore before the freeze

**The first three landed in 0.23**, which was called the pre-1.0 break batch. Entries 4 and
5 landed afterwards, which is the honest record: the batch was whatever had been noticed by
0.23, not everything that needed breaking. The entries stay for their reasoning; each
carries its outcome.

1. ✅ **`DescriptorSet.set_image` takes `index` positionally.** `set_image(binding, image,
   sampler=None, index=0)` — everywhere else in the API the extras are keyword-only, so
   `set_image(0, img, 3)` reads as "index 3" and passes 3 as a sampler. Make it
   `*, index`. `bazalt/_core.pyi:609`. Shipped in 0.21, so the exposure is one release.
   DONE in 0.23, on all three set verbs — one rule, not one exception.
2. ✅ **`create_buffer` and `Buffer.update` name their first parameter `list`, `array` or
   `size_in_bytes`.** These are the names an IDE shows, and two of them shadow builtins.
   `data` in all of them. Breaks only a caller who passed by keyword, which nothing in the
   examples or the tests does. `bazalt/_core.pyi:401` and `:1706`. DONE in 0.23 — one name
   across the overloads is also what makes the keyword spelling work whichever body the
   argument picks.
3. ✅ **`ResourceError` is the bag for everything.** It covers a strided numpy view, a resource
   from another Context, a double `acquire()`, `set_present_mode` while an image is acquired,
   an unsupported blit, a query outside a pass, a missing file and an exhausted pool. The
   `BazaltError` docstring divides errors into "bazalt had to ask an object" and "the argument
   is wrong on its own", and a double `acquire()` is neither — it is a sequencing error, a
   third category with no name. Proposal: `StateError` beside it. **Counter-argument, kept on
   purpose:** one class fewer is one thing fewer to learn, and the split has to be right the
   first time because `except` clauses freeze at 1.0. Decide before the freeze either way;
   this is the only entry in the file whose cost goes UP if it is deferred.

   DONE in 0.23, and wider than proposed: the census found a THIRD kind hiding in
   `ShaderError` too (capability gates in the builders and the compiler), so the split is
   `StateError` plus `UnsupportedError`. See "The 0.23 error taxonomy" above for the full
   reasoning and what stayed where.

4. ✅ **`target.layer()` takes `mip` positionally.** Found by the 0.23 release review, after
   entry 1 above was called done. `layer(0, 2)` is the identical trap to
   `set_image(0, img, 3)` — two adjacent ints of which the second selects a different axis —
   and the 3D work in the same release made it worse, because on a target over a volume the
   FIRST argument is a Z coordinate, so both ints read as position. DONE in 0.23: three test
   call sites, no example, one `py::kw_only()`.

   **The lesson is about entry 1, not about this signature.** Entry 1 named three verbs and
   was implemented as those three verbs. The rule it was written from — extras are
   keyword-only, "one rule, not one exception" — has no such list. A backlog entry that
   states a rule and then enumerates its known instances gets implemented as the
   enumeration, and the instance nobody wrote down survives into the freeze. When an entry
   generalizes, **grep for the shape** and put the result in the entry before it is worked.

5. ✅ **`compile_shader` takes a path it never opens.** Found by the 0.24 example sweep, from
   a user question that no amount of reading the stub would have answered: with `source=`,
   what is `path` for? The honest answer was four things at once, one of which — choosing
   the parser — nothing in the name suggests. DONE in 0.24 as two overloads plus
   `bz.ShaderLanguage`; see "Shader compilation" above for the full reasoning.

   **The entry is worth keeping for how it was found.** The API review in 0.22 read this
   signature and passed it, because every parameter is individually defensible and the
   docstring explains the rule. What exposed it was somebody asking why, out loud, and
   being unsatisfied with a correct answer. **A rule that needs a paragraph to justify is a
   design smell even when the paragraph is true** — the next review should treat "the
   documentation has to explain this" as evidence rather than as mitigation.

   It also broke the "all three landed in 0.23" claim above, which said the pre-1.0 break
   batch was closed. Two entries have landed since. The batch was never a batch; it was
   whatever had been noticed by 0.23.

### Ergonomics, additive

Ordered by how often the friction shows up, not by effort.

1. ✅ **The descriptor pool makes the user do Vulkan's arithmetic.** DONE in 0.23 — the
   automatic pool, exactly the fix this entry proposed. See the decision above.
   `create_descriptor_pool(max_sets=4, uniform_buffers=8, samplers=4)` asks for per-type
   descriptor counts, and `allocate_frame_set` then consumes `frames_in_flight` sets and
   `frames_in_flight × N` descriptors (`src/DescriptorSet.hpp:499`). So the correct number
   depends on `ctx.frames_in_flight`, which does not appear in the call — `examples/09_shadow_map`
   guesses with headroom, which is the only available strategy. Getting it wrong yields
   `VK_ERROR_OUT_OF_POOL_MEMORY` with the raw name, by the HARD entry above. Fix: make the
   arguments optional and allocate another `VkDescriptorPool` block when one fills; explicit
   sizes stay as the escape hatch.

   **This is not the idea rejected in 0.19** ("Optional binding declarators, inferred from
   SPIR-V"). Layouts stay hand-declared, with the stage they carry and the reload behaviour
   that decision protects. What goes away is only the arithmetic, which the declarators
   already know at build time. Anybody re-reading both entries should not conflate them.
2. ✅ **Three names for one descriptor type.** The builder says `.texture()`, the pool says
   `samplers=`, the set says `set_image()`, and all three mean
   `VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER`. Rule 1 says one name per thing. Fold the
   rename into entry 1, since that call site is being changed anyway. DONE in 0.23:
   `samplers=` became `textures=`. `set_image` stays — it names its argument (an Image),
   not the descriptor type, and `test_stubs.py` asserts `set_texture` is dead. Two of the
   three were the same name problem; the third was a different question.
3. ✅ **`bind_descriptor_set` and `push_constants` take a pipeline that is already known.**
   DONE in 0.25 as short overloads beside the long ones, which stay for a recording split
   across functions and for binding a set against a different compatible pipeline. The entry
   was right that both arguments are derivable and wrong about where from: a `DescriptorSet`
   did NOT record its set index or its bind point, so it had to start. Push constants use the
   LAST pipeline bound whatever its bind point, because a push-constant range belongs to a
   layout rather than to a bind point.
   `cmd.bind_descriptor_set(scene_set, scene_pipe, set=0)` passes three arguments of which two
   are derivable: `bound_graphics_pipeline_` and `bound_compute_pipeline_` are already kept as
   record-time state (`src/CommandBuffer.hpp:539`, added by 0.19's reflection work), and a
   `DescriptorSet` already knows the layout and set index it was allocated from. Add the short
   overload, keep the long one for a recording split across functions. **~150 lines.**
4. ✅ **`cmd.begin()` has no `cmd.end()`.** DONE in 0.25 as `with ctx.record() as cmd:`,
   with `begin()` kept. **`__exit__` deliberately does not submit**, which the entry did not
   say and is the only decision in it: who submits — `ctx.submit` or `renderer.present` — is
   the caller's, and a block that guessed would be wrong half the time. The scope brackets
   the recording, which is exactly what `begin()` starts. Every other pair in the API is symmetric —
   `begin_rendering`/`end_rendering`, `begin_label`/`end_label` — and this one means "reset and
   start recording" while being named like half of a pair. Add `with ctx.record() as cmd:`;
   `begin()` stays. **~120 lines**, and the pattern is already written twice — `Timer` and
   `OcclusionQuery` are both usable as context managers.
5. ✅ **The keyboard is bare ints while the gamepad is an enum.** DONE in 0.23, in the
   backward-compatible shape this entry predicted. See the decision beside the gamepad's. `bz.KEY_W: int`,
   `set_cursor_mode(mode: int)` and `is_mouse_button_pressed(button: int)` sit next to
   `bz.GamepadButton.A`, which got an `IntEnum` in 0.21. Nothing type-checks, nothing
   autocompletes as a group, and `is_key_pressed(bz.MOUSE_BUTTON_LEFT)` compiles. `Key`,
   `MouseButton` and `CursorMode` as `IntEnum`s are backward-compatible by construction; the
   `bz.KEY_W` names stay as aliases. The api-coverage report's "124 of 127 key constants
   untouched" is the same fact seen from the test side.
6. **A binding read by two stages is declared twice.** A camera UBO read in vertex and
   fragment is `.uniform_buffer(0, VERTEX, set=0).uniform_buffer(0, FRAGMENT, set=0)`. The
   merge is real and deliberate (`src/Pipeline.hpp:591`) but appears nowhere in the stub, so a
   user cannot know it is allowed. Accept a sequence, and document the merge either way — the
   documentation half is worth more than the sugar.

   **The documentation half shipped in 0.24**, so what is left is the sequence. **~180
   lines**, and the cost is spread rather than deep: every declarator on both builders takes
   the stage, so accepting `ShaderStage | Sequence[ShaderStage]` is one small change repeated
   across all of them plus the stub for each.
7. ✅ **`set` has no default on the graphics builder and defaults to 0 on the compute one.**
   Same declarator, two contracts. DONE in 0.24: the graphics declarators default to 0
   as well. The entry was written from the audit; what settled it was
   `examples/34_showcase`, where a reader asked why every line said `set=0` — a
   pipeline with one set now says nothing about it, and the demo's two-set pipelines
   still name `set=1` where the distinction is real, which is the only place it reads as
   information. Purely additive: every existing positional and keyword call is unchanged.

   **The first pass fixed half of it**, and the example sweep later in 0.24 found the
   other half: `allocate_set`, `allocate_frame_set` and `bind_descriptor_set` still
   REQUIRED `set`, so a one-set program stopped writing `set=0` on the declarator and
   went on writing it twice per descriptor set anyway. Defaulting only the declarator is
   the same "same question, two contracts" the entry opened with, moved one call along.
   All four default to 0 now. The lesson for the next such change: a default belongs on
   every call that asks for the value, not on the one the audit happened to be reading.
8. ✅ **Two factory conventions.** Buffers, images, samplers, pools, command buffers and
   pipelines come from the Context; `RenderTarget` and `SwapchainRenderer` are top-level
   constructors taking the Context as their first argument. Either is fine, both is a coin
   flip to memorize. `ctx.create_render_target()` as an alias; the class stays. DONE in
   0.23 for BOTH types — and **the entry's own proposal was wrong about how**. "As an
   alias" would have kept two spellings alive, which the 0.18 audit calls a fork; the
   constructors are gone instead. The entry also let `SwapchainRenderer` off on the
   grounds that its first argument is the Window rather than the Context, and that is a
   distinction about the argument list, not about ownership: the renderer is a Context
   resource like the rest, so it is `ctx.create_renderer(window)`. See the decision above.
9. ✅ **No `close()` and no context manager on `Context` or `Window`.** DONE in 0.24 for
   `Context`, and **the entry's own list of what leaks was half wrong**, which is what the
   implementation was decided by. See the decision above. `Window` is deliberately not in
   it: the notebook case never reaches a window, and closing one whose surface a
   `SwapchainRenderer` still holds would dangle where `py::keep_alive` makes that impossible
   today. It comes back as its own entry if somebody asks.

### Features, additive

Ordered by rule 4 — what makes pictures goes first.

1. ✅ **3D textures.** No `image3D` or `sampler3D` anywhere, which rules out 3D LUT colour
   grading, volumetric noise and every volume renderer. The descriptor type does not change
   (still combined-image-sampler or storage-image), so the work is image creation and a view
   type: `create_image(w, h, depth=n)`. Best ratio of picture to code on this list. DONE in
   0.23, the release's headline, render-to-slice included — see the decisions above and
   `examples/31_volume_raymarch` / `32_lut_grading`.
2. ✅ **A blend escape hatch.** Three preset modes and no way past them. `MULTIPLY`
   — an AO or shadow overlay — cannot be spelled at all, nor can a min/max blend op. Rule 2
   says leave the hatch: `BlendMode.MULTIPLY` first, `blend(enable, src=, dst=, op=)`
   underneath it. DONE in 0.23, both halves, in the order the entry gave. See the decision
   above.
3. ✅ **`VertexFormat` stops at `UINT`.** There is no `UINT2/3/4`, so skinning joint indices
   (`uvec4`) cannot be delivered — `UBYTE4_NORM` carries the weights and nothing carries the
   indices. Add `UINT2/3/4` and `UBYTE4_UINT`. DONE in 0.23.
4. ✅ **`bz.wait_events(timeout=None)`.** DONE in 0.24, in the shape the entry gave. One
   thing the entry did not see, and it is in the docstring: a hot-reload edit does NOT wake
   it, because the watcher runs on its own thread and its result is applied from
   `begin_frame()`. A program that only ever waits for input sees the reload at the next
   click, so pass a timeout there.
5. ✅ **`window.text_input()`.** DONE in 0.25, and it became the release's feature. Dropped
   files and the clipboard both shipped in 0.19, but there was no character callback, so
   typing a filename inside an application was impossible. Same per-cycle rotation as the
   key edges. **~180 lines**, and the estimate was right — the decision the estimate did
   not see is that it returns `str` and encodes in the callback. See the decision above.

   **It is not a small sibling of the key edges, and 0.25 is where that became clear.**
   `glfwSetCharCallback` reports Unicode *codepoints*; `is_key_pressed(Key.A)` reports a
   physical key. The layout, shift, AltGr, dead keys and an IME all live between the two, so
   no amount of key state reconstructs the character — which is why every text field in every
   toolkit reads the character stream instead. This is also **the one thing an ImGui backend
   cannot supply for itself**: `ImGuiIO::AddInputCharacter` has to be fed from exactly this
   callback, so the entry below depends on this one.
6. ❌ **Y-flip through a negative-height viewport.** REJECTED in 0.24 — see
   "Rejected, and why". The entry named the catch correctly (it flips the winding) and drew
   the wrong conclusion from it: a Context-level opt-in does not contain that cost, it gives
   `FrontFace` and `CullMode` two meanings. And the line it wanted to remove belongs to GLM
   rather than to Vulkan, which nobody had checked. The escape hatch already exists
   (`cmd.set_viewport(0, h, w, -h)`), so what shipped is the explanation.

### Moved in from the ceilings, by the 0.22 audit

These arrived with their reasoning already written; see the traces above for what each one
used to claim.

- ✅ **Gamepad edge queries** (was STALE) — DONE in 0.25, and the entry read the shape
  correctly: a per-pad previous-state snapshot beside a counter that already existed.
  **~150 lines** was close. What neither the entry nor the estimate saw is that the LIMIT
  it inherited from 0.21 was reasoning about windows rather than about pads, and that GLFW
  having no gamepad callback makes this edge weaker than a key edge. Both are in the
  decision above.
- ✅ **A sampleable multisampled image** (was UNASKED) — DONE in 0.25 as
  `keep_samples=True` plus `target.multisampled_color`. The estimate of ~300 lines named
  two costs that do not exist (no declarator, no new descriptor type) and missed the two
  that do (the attachment store-op and the layout the pass retires to). See the decision
  above, and `examples/36_msaa_resolve`.
- ✅ **`stride=` on the indirect verbs** (was UNASKED) — DONE in 0.25. Two verbs, not three:
  `dispatch_indirect` issues one command, so there is nothing for a stride to step over, and
  the code says so where the argument is missing. What the estimate did not see is that
  `check_indirect_` had to stop multiplying — the LAST command needs only its own struct, not
  a whole stride, so a buffer sized exactly for the data was being refused. A stride below
  the struct size, or not a multiple of 4, is refused with the reason.

  **It is the one 0.25 feature with no example, and the reason is worth recording rather
  than fixing badly.** `examples/28_gpu_culling` looked like its home and is not: it draws
  ONE command whose instanceCount an atomic accumulates, so there is no second command for a
  stride to step over. `examples/34_showcase` does multi-draw, and already solves the
  per-draw-data problem the other way — `firstInstance` carries the material slot, which
  costs no stride at all. So a demo would have to invent a use, and an example that invents
  its own reason teaches the argument rather than the technique. The test
  (`test_indirect_stride_steps_over_per_draw_data`) renders the same two stripes the packed
  multi-draw produces, which is the honest check; the example waits for a program that
  actually wants interleaved arguments.
- ✅ **A precise occlusion count** (was UNASKED) — DONE in 0.25 as `Feature.PRECISE_OCCLUSION`
  plus `VK_QUERY_CONTROL_PRECISE_BIT` where it is on. The code comment's worry — that
  `samples` would mean two things depending on the driver — is exactly what a `Feature`
  answers: the caller can now ask which of the two they are being given. Read once per
  recording rather than per query, because the feature set is fixed for the Context's life.
- ✅ **Per-face stencil state** (was UNASKED) — DONE in 0.25 as `face=` plus `bz.Face`, the
  upgrade path the 0.17 entry named. Two calls spell a two-sided test, which is the
  shadow-volume shape. **`enable` is deliberately NOT per face**: Vulkan has one
  `stencilTestEnable` and two op-states, so any call sets the bit and the last one wins.
  Pretending otherwise would invent a state the hardware does not have.
- ✅ **Primitive restart** (was UNASKED) — DONE in 0.25, in the shape the entry gave, and it
  was the cheapest entry in the file. A kwarg on `topology()` rather than a verb of its own,
  because it means nothing without a strip. Refused on a list topology with the reason,
  rather than left to VUID-VkPipelineInputAssemblyStateCreateInfo-topology-06252.
- ✅ **`monitor=` and video-mode enumeration for fullscreen** (was UNASKED) — DONE in 0.25.
  **~300 lines, and the estimate was the interesting part and it was right:** this reads
  like a kwarg and is not one. Choosing needs the monitors and their modes *visible* from
  Python, which is a new public type of inert data — the `Device` shape, which cost more
  than the kwarg it enables. See the decision above for the one place it departs from
  `Device`.
- **Exclusive fullscreen** (was UNASKED). A `Feature` over `VK_EXT_full_screen_exclusive`, not
  a fifth `WindowMode` — it is a property of the swapchain. **~250 lines**, including the
  acquire/release pair the extension requires and a Windows-only path in the tests.

  ✅ DONE in 0.25 as `renderer.set_fullscreen_exclusive(enable)` plus the read-back
  `renderer.fullscreen_exclusive`. **The estimate of ~250 was wrong and the shape is worth
  recording**, because every part of the overrun was next to the feature rather than in it —
  the 0.19 corollary, for the fourth time.

  `FeatureInfo` had no column for a capability spelled as an EXTENSION rather than a feature
  bit. The table's own comment had been asking for that fourth column since 0.21, and it
  drags `DeviceFeatures` (which now carries the device's extension list) and
  `feature_available` (no longer `constexpr`, because it compares strings) along with it.
  `VK_KHR_get_surface_capabilities2` is an INSTANCE extension that the device extension
  requires, so `create_instance_` has to know about a device capability before any device
  exists. The Win32 half needs an `HMONITOR`, which is a new field on `SurfaceProvider`
  (`void*`, so the header stays free of `<windows.h>`) filled from `glfwGetWin32Window`.
  And the acquire has to follow EVERY swapchain creation, not only the first: the mode
  belongs to the swapchain, so a resize or a present-mode change would lose it.

  **Asking is not getting, and the API says so.** `set_fullscreen_exclusive(True)` succeeds
  and `fullscreen_exclusive` reports what the driver actually gave, because a refusal is a
  normal outcome — another application can hold the display, and the machine this was
  written on refuses with `VK_ERROR_INITIALIZATION_FAILED` while an overlay layer is loaded.
  A warning names the VkResult rather than leaving a silent nothing.
  `APPLICATION_CONTROLLED` rather than `ALLOWED`: the caller said WHEN, so a driver-decided
  mode would make the verb a suggestion.

  **It found a real bug, and that is the best thing about it.** See "The volk device table
  had two layouts" below.
- **Compressed texture formats** (was a STALE rejection). BC and ASTC rows in `Format` plus an
  upload path that does not decode. Container parsing (KTX2) stays out of scope, which is what
  the original rejection got right. **~500 lines, the most expensive additive entry.** Not
  because of the `Format` rows: `UploadManager` decodes through `stbi` to RGBA8 on every path
  it has (`src/UploadManager.hpp:475`), so "do not decode" is a new path rather than a branch,
  and every size computation that assumes one byte per texel — mips, regions, `image.update`
  — becomes block arithmetic. Worth it at four to eight times the VRAM of an uncompressed
  asset.
- ✅ **An ImGui overlay example** (was a SCOPE rejection whose promise nobody kept) — DONE
  in 0.25 as `examples/35_imgui_overlay`, on the public primitives exactly as the rejection
  said. **~350 lines** was close, and the prediction that `window.text_input()` was its only
  missing prerequisite was WRONG by one: `set_cursor` is the other, because ImGui asks for
  an I-beam over a text field and had no way to be answered.

  **The example paid for itself the way an example is supposed to.** It found that
  `cull_mode(CullMode.NONE)` could not be spelled — `front_face` had no default, so culling
  nothing still demanded a winding that means nothing. Fixed in the same release, and it is
  the 0.24 rule that a default belongs on every call that asks for the value.

  **The entry used to say "a text or ImGui overlay example", and 0.25 split it.** Text
  rendering stays rejected and stays rejected for the reason already written — the rasterizer
  is freetype's or PIL's, and the atlas-and-quads layer is glue over an API that already has
  every primitive. Nothing about that changed. ImGui is the half worth building, and a check
  of what it needs found the gap is one call: `ImDrawData` is vertices, indices, a scissor and
  an atlas id, and `DynamicBuffer.update`, `graphics_pipeline`, `push_constants`,
  `draw_indexed`, `set_scissor`, `create_image` and a sampler already cover all of it. Mouse
  position, buttons, `scroll_dx/dy`, key edges and the clipboard shipped in 0.19 and 0.21. The
  character stream did not, so an `InputText` field in a bazalt program is deaf today. **Two
  entries that looked independent are one feature and its prerequisite**, which is the kind of
  thing a backlog hides until somebody prices it.

### Small, and looked at

Defects small enough that each is a line, plus one thing the review raised that turned out to
be fine — recorded so it does not get re-raised.

- ✅ **`cull_mode(CullMode.NONE)` could not be spelled** — FIXED in 0.25. `front_face` had
  no default, so a pipeline that culls nothing still had to name a winding that means
  nothing when nothing is culled. It defaults to `COUNTER_CLOCKWISE`, which is the value the
  builder already started with. Found by writing `examples/35_imgui_overlay`, because ImGui
  does not wind its triangles consistently — the fifth defect an example has found that no
  test looked for.
- ✅ `bazalt/_core.pyi` — the `begin_frame` docstring names `DynamicBuffer`, which is not a
  public symbol. It should say "a DYNAMIC buffer". Internal vocabulary leaking into the file
  users read. FIXED in 0.24. (In this file `DynamicBuffer` is correct: this is the
  developer's document.)
- ✅ The stub does not say that declaring a binding twice merges the stages. FIXED in 0.24 —
  the documentation half of ergonomics 6, which the entry called the more valuable half.
- ✅ `Timer.ms` and `OcclusionQuery.samples` return `None` for four different reasons — no
  timestamp support, a re-recorded command buffer, an unfinished submit, no query pool. The
  caller cannot tell "wait longer" from "this GPU cannot". FIXED in 0.24, and the 0.23
  taxonomy is what made it expressible: `UnsupportedError` for the device, `StateError` for
  the stale handle, `None` for the one remaining meaning. Breaking, in the narrow sense that
  two `None`s became raises.
- ✅ `renderer.gpu_time_ms` returns `0.0` on MoltenVK and `None` elsewhere for the same
  question. FIXED in 0.24 by the same split, and the entry's framing was off: `0.0` is not a
  sentinel, it is MoltenVK measuring and answering zero, which nothing can distinguish from
  a fast frame (the HARD ceiling stands). The real defect was the same one above — `None`
  meant "off", "cannot" and "not yet". `Disabled` joined `QueryStatus` for the first of
  those, because `gpu_timing=False` is a decision the caller made rather than a limit.
- **Looked at again in 0.24, and correct as it stands:** `Image.samples`. The entry called it
  a property with one possible value, which is true, and the docstring has said exactly that
  since it was written. Deleting it would have to be undone the day the sampleable-MSAA
  backlog item lands, and until then it answers the question honestly.
- **Looked at, and kept for the reason it was filed against:** `create_image(array,
  cube=True)` exists only to raise. The removal was tried in 0.24. Without the parameter the
  call matches no overload and pybind answers with its own dump of all four signatures,
  which is where the caller started. An argument that exists to produce one sentence beats no
  argument that produces forty lines.
- **Looked at, and correct as it stands:** `create_sampler(anisotropy=True)`. The review filed
  it as a default that silently does nothing without `optional=[ANISOTROPIC_FILTERING]`. It is
  not: `ANISOTROPIC_FILTERING` is on the implicit list (`src/Context.hpp:1350`) and is enabled
  wherever the device has it, the same way `MULTIVIEW` is. The default is honest.

## Rejected, and why

Scope rejections and implementation alternatives we looked at and turned down. Each entry
says what we did instead.

**Out of scope.** The 0.22 audit went through these too, and each one now says which of the
scope test's two questions it fails. An "out of scope" that cannot name the question is an
UNASKED wearing a costume.

- **SCOPE (fails both). `load_model`** — no Vulkan glue and trimesh already solves it. Bazalt
  does not know about file paths for geometry.
- **SCOPE (fails both). Audio and physics** — no Vulkan glue, and the ecosystem covers them.
  Do not mix them into a graphics library.
- **SCOPE (fails neither question — it is in scope, and lives beside the library).
  ImGui inside core.** ImGui makes no pixels: every frame it emits `ImDrawData`, which is
  vertices, indices, draw commands, a scissor and an atlas id. Bazalt already has every
  primitive for that (a DYNAMIC buffer's `update`, `graphics_pipeline`, an atlas through
  `create_image` and a sampler, `draw_indexed` with `set_scissor`, `push_constants`). The
  backend is about 90% glue over the public API plus input forwarding, so it belongs beside
  bazalt, not inside it.

  **0.22 note.** The scope table at the top of this file already says the same thing —
  "in scope, but a companion package" — so this is not a rejection of the capability, only of
  its address. What neither entry says is that the companion package does not exist, which
  makes the practical answer to "how do I put a slider on screen" read as "write an ImGui
  backend yourself". That is a real gap in the story, and it is filed in the backlog rather
  than here, because the fix is a package to write, not a decision to revisit.
- **A 2.0 for new hardware capabilities.** Ray tracing and mesh shaders become new entries
  in `Feature` (rule 3). Nothing about them needs a major break.

- **`Context(y_up=True)`** (0.24). It sat in the backlog under "Features, additive" as the
  biggest single ergonomic left, because `proj[1][1] *= -1` appears in eight examples and
  will appear in every user program. Rejected once the question "why does Vulkan do this?"
  got an answer, which is the order the rejection is worth remembering for.

  Vulkan's framebuffer origin is the upper-left corner and NDC +y points down. That agrees
  with the image memory layout (row 0 is the top row), with texture coordinates,
  `gl_FragCoord`, `set_scissor`, the offsets `copy_image` and `blit_image` take, and the
  raster order of a display. OpenGL's lower-left origin agrees with none of them, which is
  why GL code carries vertical flips everywhere; D3D and Metal are also top-left. Vulkan
  followed the memory and the majority.

  So `proj[1][1] *= -1` does not correct Vulkan — **it corrects a GLM matrix.**
  `glm::perspective` emits the OpenGL projection (y-up NDC, z in [-1,1]). Two of its three
  mismatches go away with `GLM_FORCE_DEPTH_ZERO_TO_ONE`; the third is that sign, and a
  projection written for Vulkan carries it already.

  A Context flag would therefore compensate in the rasterizer for somebody else's matrix,
  and charge three things. A negative viewport height mirrors the geometry, so it flips the
  winding: `FrontFace` and `CullMode` would mean the opposite thing depending on a keyword
  argument, which is rule 1 broken by a flag. The rasterizer would be y-up while
  `image.read()`, `copy_image`, `blit_image` and `set_scissor` stay top-left — one Context,
  two conventions. And every example and shader has to be read once.

  Rule 2 is not at stake, which is what makes this a rejection rather than a ceiling:
  `cmd.set_viewport(0, h, w, -h)` passes a negative height straight into `VkViewport`
  (`src/CommandBuffer.hpp`), so a caller who wants the flip already has it. What shipped
  instead is the explanation — a "Vulkan clip space" section in the README saying whose line
  the flip is.

- **Inline notebook display (`Image._repr_png_`)** (0.24). **SCOPE, fails the second
  question.** `img.read()` returns a numpy array and `PIL.Image.fromarray(...)` displays it
  in any notebook, local or browser-based. No Vulkan glue is involved anywhere in it.

  It is also the same argument that already rejects `image.save(path)` two entries above,
  and taking one while refusing the other would be a fork in the reasoning rather than in
  the API. The gamepad precedent does not reach it either: a thin wrapper earns its place
  when it has decisions in it, and every decision here — which layer, which mip, what a
  depth or R32F image looks like as a picture — already belongs to the caller through
  `read(layer=, mip=)`.

  What the notebook actually needed was `close()`, coverage of the headless instance, and an
  example. Those shipped in 0.24; see the decision above.

- **SCOPE (fails the second question, for the rasterizer half only). Text rendering and a
  debug overlay.** Font rasterization is the ecosystem's (freetype, PIL), and the
  atlas-plus-quads layer is the same argument that kept ImGui out of the core: about 90% glue
  over the public API. A companion package or an example.

  **0.22 note, and it cuts against the entry.** The identical argument would have rejected
  the gamepad wrapper, which shipped in 0.21 — and the entry written for it says "a thin
  wrapper is not the same as a wrapper with no decisions in it". An overlay has decisions in
  it too (atlas residency, one dynamic buffer or many, whether it owns a pipeline). Rule 4
  also ranks it high: for a prototyping library, putting a number on screen is the second
  thing a user wants after a triangle. The verdict stays SCOPE because the rasterizer really
  is freetype's job, but the honest reading is that "a companion package or an example" is a
  promise nobody has kept, and an unkept promise reads to a user as a refusal. Filed in the
  backlog as an example to write.
- **SCOPE (fails the second question). `image.save(path)`.** No Vulkan glue at all —
  `read()` returns a numpy array and PIL/imageio write the file.
- **STALE, corrected in 0.22. Compressed texture loading (BC / KTX2).** The entry said
  "parsing the container is the ecosystem's job; accepting already-compressed bytes is a
  variant of `load_image(bytes)`, not a feature of its own". The first half stands. The second
  half describes something that does not exist: `load_image(bytes)` runs
  `stbi_load_from_memory` (`src/UploadManager.hpp:475`) and produces RGBA8, and `Format`
  carries no BC or ASTC row at all, so there is no way to hand bazalt a compressed block —
  the variant the entry points at is not a variant of anything. What the rejection actually
  rejected, without saying so, is compressed textures altogether, at a cost of four to eight
  times the VRAM for every real asset. The container half stays out of scope; the format
  rows and an upload path that does not decode are in the backlog.
- **Temporal upscaling and frame generation — DLSS, XeSS, FSR** (0.25). Rejected as a bazalt
  feature, and the interesting part is that the scope test does not settle it. Question one
  says yes, loudly: an upscaler wants `VkInstance`, `VkPhysicalDevice`, `VkDevice`,
  `VkCommandBuffer` and each input image with its view, format and subresource, and it wants
  the extension list decided before the device exists. Question two also says yes for a
  Python caller — nothing in the ecosystem hands you DLSS from Python. Both answers point in,
  and it still stays out, on the two rules the scope test does not cover.

  **Rule 4 is what decides it.** Bazalt ranks what makes pictures first, and an upscaler
  makes no picture. It makes an existing picture cheaper, and the user of a prototyping
  library is bounded by their own edit-run loop rather than by fill rate. That is the
  opposite end of the ordering from "put a number on the screen", which is still unbuilt.

  **Rule 3 rejects the vendor half separately.** DLSS is NVIDIA RTX only, so it is under the
  90% line by construction — and a `Feature` row is not the answer here, because `Feature`
  negotiates a capability the driver already has, not a closed-source blob
  (`nvngx_dlss`) that a wheel would have to carry under a separate licence, per platform.
  XeSS has the same shape.

  **FSR is the path that stays open, and it is open outside bazalt.** FSR 2 and 3 are open
  source and are ordinary compute passes, so they run on any GPU and need no blob, no new
  extension and no handle: they are the ImGui argument again — glue over the public API,
  written beside the library. Rule 3 does not block them; only rule 4 does, which is a
  statement about bazalt's ordering rather than about the technique.

  **What bazalt already gives anyone who tries.** Motion vectors are a second colour
  attachment, and `RenderTarget` has taken several since it existed. Rendering small and
  presenting large is an `OffscreenTarget` plus a draw, today. The one piece of sampler state
  every temporal upscaler forces, a negative LOD bias, shipped as `create_sampler(mip_lod_bias=)`
  for unrelated reasons. Jitter is a projection matrix, which bazalt deliberately does not
  own — the same line the `Context(y_up=True)` rejection above draws.

  **The one piece that could not live outside.** Frame *generation* interposes on present and
  on frame pacing, and `SwapchainRenderer` owns both. If this ever comes back, the question
  to answer is "a hook on present", not "DLSS" — and it should be filed as that.
- **Dual-source blending, blend constants, unnormalized sampler coordinates.** Real Vulkan
  state, but nothing on the proposal list needs them. They stay unlisted until something
  asks. **0.22:** something asks, and it is smaller than any of these three — `MULTIPLY` and
  a min/max blend op, which the three preset `BlendMode` values cannot spell. That is in the
  backlog. Dual-source and unnormalized coordinates stay here, unasked and unlisted.

  **0.23 shipped that smaller thing** (`BlendMode.MULTIPLY`, then `BlendFactor`/`BlendOp`),
  and building it sharpened why the other two stay: the hatch names every factor that is
  only an enum row, and stops exactly where a factor needs machinery beside it. Blend
  constants need a verb to set them and the dynamic state to change them per draw;
  dual-source needs a second output in the shader and a `Feature` row for
  `dualSrcBlend`. Neither is a hole in the hatch — each is its own proposal, still unasked.

**Implementation alternatives.** The 0.22 audit left these unlabelled on purpose. Rule 7
sorts entries that say "bazalt does not do this"; these say "bazalt does this the other way",
which is a choice between two implementations of the same reachable behaviour. There is no
ceiling to raise, so there is nothing for the five verdicts to grade.

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

  **What the freeze actually blocks** (0.22). The ergonomics backlog has twenty-odd entries
  and exactly **three** of them break the API: keyword-only `index` on `set_image`, the
  `data` parameter rename on `create_buffer` / `Buffer.update`, and splitting `StateError`
  out of `ResourceError`. Those three are bounded by this release and nothing else is —
  every other entry, including the enums, the short `bind_descriptor_set`, 3D textures and
  the whole ceiling backlog, is additive and can land after 1.0 without a deprecation cycle.
  So the freeze is not a deadline for the backlog; it is a deadline for three lines of
  signature. Deciding how many releases the rest takes is a separate question from deciding
  when 1.0 ships.

  **0.24 re-opened it, narrowly, and the reason is worth the entry.** 0.23's census looked
  at SIGNATURES, and these are RETURN CONTRACTS: `Timer.ms`, `OcclusionQuery.samples` and
  `renderer.gpu_time_ms` each turned two of their `None`s into exceptions. No signature
  changed, so nothing the 0.23 sweep looked for would have caught them. **The lesson is the
  0.23 one applied one level up:** an entry that generalizes gets implemented as its
  enumeration, and "grep for the shape" has to include shapes that are not in the signature.
  A return type of `Optional[T]` is a shape.

  Two of the three are one release of exposure. Nothing else breaking is open, so 1.0 still
  waits on no signature.

  **0.23 paid all three**, plus the break the census found hiding in `ShaderError`
  (capability gates), the `samplers=` → `textures=` rename that rode along, and a FOURTH
  signature the release review turned up after the first three were called done:
  `target.layer(index, mip)` was the same positional-extra trap as `set_image` (backlog
  entry 4 says why one entry produced two rounds of work). Nothing breaking remains on the
  backlog, so 1.0 no longer waits on any signature.
- **Every public symbol from `_core.pyi` is touched by a test.** An unexercised binding is
  an unimplemented binding. 0.22 made this measurable rather than aspirational:
  `pytest --api-coverage` writes `api_coverage.md`, and the answer on the day it was
  written was **304 of 497** — 129 of 156 methods, 63 of 63 properties, 97 of 139 enum
  members and 3 of 127 key constants. The untouched list is the 1.0 test plan, and
  `tests/api_coverage_baseline.txt` is what stops it growing in the meantime.

  Two things the measurement says that the old wording did not. Properties are already at
  100%, so the gap is not where "unexercised binding" suggested. And the constants are
  most of the arithmetic: they are read-only integers, so a test that touches all 127
  proves nothing except that they exist — which `test_stubs.py` already checks. Reading
  the number as one percentage would put 1.0 behind a pointless test.

  **0.23 acted on that paragraph instead of only writing it down, and then the number
  answered.** The census stopped counting two things it was double-counting or
  fabricating, and both changes are asserted by `tests/test_api_coverage.py` so neither
  can quietly come back:

  - the 127 `KEY_*` / `MOUSE_*` / `CURSOR_*` integers, which since 0.23 are the pre-enum
    spelling of `Key`, `MouseButton` and `CursorMode` — enums the census already counts.
    Keeping both reported the keyboard TWICE (116 untouched constants beside 99 untouched
    `Key` members) and made it the largest number in the file for no fact.
  - `__init__` on a class with no `py::init`. pybind leaves a slot wrapper there that
    exists to raise `TypeError`, and 0.23 made two more classes unconstructible on
    purpose. Counting it asks for a test that constructs what cannot be constructed —
    **23 of the 26 untouched methods were this**, which is why the methods row looked
    like the problem and was not.

  With the noise gone the real answer is **347 of 506**: methods 136/136, properties
  64/64, functions 5/5, exceptions 9/9, and enum members 133/292. So the goal is now one
  row, not five, and the four methods that were genuinely untested
  (`ComputePipelineBuilder.name`, `OcclusionQuery.stop`, `Window.is_open`,
  `Window.set_title`) got tests in 0.23 — all four trivial, none previously noticed, which
  is the argument for the census in one line.

  The remaining 159 are enum members, and they are NOT one job. A `Format` row nobody
  passed to `create_image` is an untested code path; a `Key` member is a read-only integer
  that `test_stubs.py` already proves exists — the same argument the constants lost on.
  Split the row before treating it as the 1.0 test plan.

  **Estimate: ~400 lines**, and the split is most of the value rather than a preliminary. The
  members that name a code path — `Format`, `BlendFactor`, `BlendOp`, `Access`, `AddressMode`
  — get a test that passes them to the verb they belong to, which is a parametrized test per
  enum rather than a test per member. The members that are read-only integers get counted
  separately and left alone, exactly as the 127 key constants were in 0.23. Doing this
  without the split first is how a percentage turns into busywork.
- ✅ Add the `KEY_*` and `MOUSE_*` constants to `__all__` in `bazalt/__init__.py` — DONE, and
  it had been done for several releases while this line still asked for it. Found by the 0.20
  audit. A checklist nothing tests is a checklist that drifts.
- Performance: a pipeline cache on disk. Descriptor indexing shipped in 0.21.
- ✅ Indirect draw / GPU-driven work: **ship them, or defer them with an explicit
  note.** Shipped — the verbs in 0.19 and the GPU-decided count in 0.21. Multi-submit
  is the half still open, and it is a question rather than a plan: nothing has asked
  for a second queue.
- Close debt #4, or write it down as an accepted ceiling. Debt #3 was paid in 0.19
  and debt #5 in 0.21, so #4 is the only entry left, and it waits on someone else's
  package.
- **New capabilities after 1.0 are additive** (rule 3).

---

## Lessons and traps

Lasting engineering conclusions, distilled from the retrospectives. Do not repeat them.

### API design

- **When the exception type is the contract, the docstring that names it is the contract
  too** (0.23). The release split `StateError` and `UnsupportedError` out of `ResourceError`
  and swept the C++ call sites, the tests and the CHANGELOG. It did not sweep the docstrings
  in `_core.pyi`, so six of them still promised `ResourceError` for a double `acquire()`, a
  blit this GPU cannot do, an occlusion query outside a pass. A user writes `except` from
  the file their editor shows them, so the stub was handing out a contract the library no
  longer honoured — a silent one, because nothing runs a docstring. **Any change to an error
  taxonomy is a grep of the prose, not only of the code**, and the count is the tell: six
  wrong lines against about forty correct ones is drift, not a typo.
- **A second spelling is not a convenience, it is a fork** (0.18), **and the rule catches
  additions as readily as legacy** (0.23). `ctx.create_render_target()` was added beside
  `bz.RenderTarget(ctx, ...)` as "an alias; the class stays", which is how all six of the
  0.18 forks began — an ergonomic shortcut next to the general form, both kept because
  removing either would break someone. The tell showed up immediately: the examples split
  into two dialects, the new ones using the factory and the old ones the constructor. The
  fix was to finish the migration rather than document a preference. **When a backlog entry
  proposes "add X as an alias for Y", read it as "fork Y" and price the removal instead.**

  The audit before
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

- **A test branch that only a portability driver reaches is a branch no local run checks**
  (0.23). The 0.22 capability tests are written as `if ctx.supports(X): assert it works;
  else: assert it is refused`, which is the right shape — and the `else` half executes on
  MoltenVK and nowhere else. So the 0.23 error split moved those refusals to
  `UnsupportedError`, the whole suite passed twice locally and on lavapipe, and macOS CI
  went red on three tests whose expectation had rotted under them. The census that found
  every OTHER site was "run the suite and read the failures", and that method is blind here
  by construction. **When a release changes what an error IS, grep the tests for the type by
  name; do not trust a green run to have visited every branch.**

- **A test may only assert what bazalt promises, not what a driver happens to do** (0.23).
  `test_a_full_fixed_pool_raises_one_type` asserted that a pool sized for one set refuses
  the second. Vulkan permits an implementation to serve more than the pool was sized for,
  and lavapipe does, because its descriptors are host memory — so the test failed on the
  one machine the suite gates on. What bazalt actually promises is the TYPE: if the driver
  refuses, it is `ResourceError` and never `OutOfMemoryError`. The test now probes, asserts
  the type when a refusal comes, and skips naming what it saw when none does — the same
  shape as the 0.22 MoltenVK skips, for the same reason.

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

  0.21 found the strongest form of this yet: `29_bindless` **drew nothing at all**,
  and a clean 20-second run with the validation layers on did not say so. Its quad
  was wound the obvious way — right along the top first — which is back-facing under
  the pipeline default, so every triangle was culled. Culling is not an error, a
  window nobody looked at reports nothing, and the layers have nothing to complain
  about. What caught it was rendering the example's own shaders into an offscreen
  target and asserting per quad. Both `fullscreen.vert` and `stripe.vert` already
  carry a comment about this exact trap, which is the second lesson: **a warning
  written in one file does not reach the file that needed it.** For a new example
  the first measurement should be "is anything on the screen", because that is the
  failure a run hides best.

  The measurement also has to survive the pipeline it is measuring. The first
  version compared each quad's centre TEXEL against the source texture and failed on
  8 of 48, because a 64-pixel texture minified into a 43-pixel quad lands wherever
  the filter puts it. Comparing the mean colour of the quad's interior is the same
  claim without a dependence on sampling: pick the statistic the effect cannot move.

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

- **A capability with a roll-up boolean is two questions, not one** (0.21).
  `VkPhysicalDeviceVulkan12Features::descriptorIndexing` reads like the switch for
  descriptor indexing and is not: the spec says enabling it "does not imply the
  other minimum descriptor indexing features are also enabled". It answers "is this
  set of capabilities available", and every bit still has to be turned on by name.
  A feature that ships on the roll-up alone compiles, validates, and does nothing.
  Whenever a feature struct has a member with the same name as the extension, read
  what the spec says it *indicates*.

- **A container that only ever grows is a leak with a delay** (0.21). Every
  `set_image` appended to the descriptor set's list of bound images, which is
  correct-looking and was fine while a binding held one descriptor: rewriting one
  was rare, and the extra `shared_ptr` was one image. An array rewritten every frame
  turns the same line into unbounded growth plus a record-time walk that gets slower
  forever. The fix is to give the entry the identity it always had — here
  `(binding, index)` — and replace rather than append. Look for this wherever a
  "bind" or "set" verb records what it was handed.

- **The same question answered in two places will be answered differently**
  (0.21). `StaticBuffer::create` switched on `BufferType` to pick usage flags;
  `DynamicBuffer::create` asked "is it STORAGE?" and called everything else a
  uniform buffer. So a DYNAMIC vertex buffer — geometry rebuilt every frame, which
  is the whole point of DYNAMIC — had no `VERTEX_BUFFER` bit, and binding one was a
  validation error with a draw reading undefined data behind it. Neither half looks
  wrong on its own, which is why it survived: the bug is the pair.

  What found it was running `examples/28_gpu_culling`, the only code in the repo
  that ever made one. Nothing in the suite did, and the parametrized test that now
  covers every (type, memory usage) pair is four lines. **When two constructors
  answer one question, the cheap test is the cross product, not another case of the
  path you were looking at.**

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

  **0.23 moved the construction to `ctx.create_renderer(window)` and the annotation had to
  move with it**: on a factory the nurse is the return value, so `keep_alive<1, 2>` becomes
  `keep_alive<0, 2>`. The indices are positional and silent — write the constructor's pair
  on a factory and you have re-created this bug, with the same test still passing if it only
  checks the refcount of the wrong object. **When a binding changes shape, re-derive its
  keep_alive indices from what the nurse now is; do not port the numbers.**

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
- **`test_stubs.py` must pass with no GPU, and something has to say so** (0.23). It is the
  file `CIBW_TEST_COMMAND` runs on every wheel, and those runners have no Vulkan driver, so
  one test there that asks for `ctx` fails at SETUP with `InitializationError` and takes all
  three wheel jobs down — which also cancels the lavapipe and MoltenVK jobs that depend on
  them. 0.23 shipped exactly that for one commit, and the failure reads like a broken wheel
  rather than a misplaced test, which is the expensive part. `test_this_module_needs_no_gpu`
  now asserts that no test in the file takes any parameter at all. The blanket form is
  deliberate: "not `ctx`" needs a list of which fixtures are safe, and this needs nothing.

  Generally: **when a file is a CI entry point, the constraint that makes it one belongs
  inside the file as an assertion**, not in a comment and not in the workflow. The person
  adding a test there is reading the tests, not `build.yml`.
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

- **Where the build happens decides what can be cached** (0.22). The obvious cache is the
  CMake build tree, and on Linux it is worthless: cibuildwheel copies the project INTO the
  container and copies only the wheels back out, so nothing written to `build/` there ever
  reaches the host to be saved. Linux caches the SDK tarball instead, downloaded on the host
  and unpacked in the container. Windows and macOS build natively and cache both.

  What the build-tree cache actually saves is worth knowing before anyone measures it and is
  disappointed: `actions/checkout` stamps every source file with the current time, so bazalt's
  own translation units recompile on every run regardless. `_deps` arrives with its cached
  timestamps, so glfw, volk, vma, vk-bootstrap and stb neither re-clone nor rebuild. The key
  hashes `CMakeLists.txt`, so a pin bump busts it by construction.

- **A format gate belongs in CI, not in a habit** (0.22). Every release through 0.21 ended
  with a `style(0.N): clang-format over the release's own code` commit, which is the shape of
  a rule that is enforced by remembering. The job pins the formatter version from PyPI rather
  than taking apt's, because clang-format changes its output between major versions — an
  unpinned formatter turns somebody else's release into a red build here, which is tech debt
  #4's lesson arriving from a different direction.

- **The macOS SDK install is a composite action** (`.github/actions/install-vulkan-sdk-macos`,
  0.22) because two jobs need it and it is twenty lines of "find the thing, do not spell it".
  Both halves of that were learned the hard way: the download is a `.zip` and the first
  attempt guessed `.dmg` (a 404), and the installer app's name and depth inside the archive
  have moved between SDK releases, so it is found with `find` rather than named. LunarG
  publishes `https://vulkan.lunarg.com/sdk/files.json`, which answers both questions for any
  version — check it before bumping `VULKAN_SDK_VERSION`.

- **macOS strips `DYLD_*` when it starts a SIP-protected binary**, and `/bin/bash` is one, so
  a `DYLD_LIBRARY_PATH` written to `$GITHUB_ENV` does not survive into the next step. The
  loader is opened with `dlopen` — by volk, and again by GLFW — so it has to be findable
  without that variable. A symlink into `/usr/local/lib` is what the SDK's own system install
  writes, and it survives everything.

- **`GIT_SHALLOW` with a raw commit SHA works only while that SHA is the branch tip** (0.24).
  A shallow clone fetches the tip of the default branch and nothing else, so CMake's
  clone-then-checkout fails with `unable to read tree` for any other commit. stb is pinned to
  a SHA because it publishes no tags, and the pin was master's tip when it was written, so
  the combination looked correct for four releases. It broke the day upstream pushed past it,
  in a build where nothing local had changed — the exact failure the pin was added to
  prevent, re-entering through the fetch mode.

  Two things generalize. **A configuration that is only correct by coincidence gives no
  warning while the coincidence holds**, so "it has always worked" is not evidence about a
  build pin. And the fix is not to move the pin forward: that re-arms the same trap for the
  next upstream push. The four tagged dependencies keep `GIT_SHALLOW`, because a tag is a
  name the server resolves and no history is needed to reach it. Only stb pays a full clone,
  which is 6 MB and one second.

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
- **A connected gamepad is unverified** (0.21). No test can plug one in, and the
  machine the release was built on had none, so `examples/30_gamepad` was checked by
  substituting a fake reading and measuring the frame it produced — the colour
  follows the sticks, in the direction they moved. What that does NOT cover is
  whether GLFW's mapping reports a real pad the way the enums claim: the axis
  ordering, the trigger range, and whether a stick at rest reads near zero. Run the
  example with a pad before trusting any of it.

- ✅ **The headless fallback** (no windowing extensions) still has no coverage. COVERED in
  0.24 by `BAZALT_FORCE_HEADLESS=1`, and it found a bug on the first run: the device enabled
  `VK_KHR_swapchain` on a headless instance, so every Context on a display-less machine
  emitted `VUID-vkCreateDevice-ppEnabledExtensionNames-01387`. The knob rides inside the
  ordinary suite through a fixture, so CI needed no leg of its own.
- **The windowed path is verified by running the examples, and that is load-bearing.** The
  0.17 depth/stencil layout bug existed only for a target whose depth is scratch — a window —
  and every headless test passed with it in place. Run the new example before calling a
  release done; "the suite is green" does not cover the half of the library that needs a
  surface.
