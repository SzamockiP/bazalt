# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**bazalt** — a Python library for rapid GPU/shader prototyping on Vulkan. A C++23 core in
`src/` compiled into a single pybind11 extension (`bazalt._core`), plus a thin Python
package in `bazalt/` that only re-exports it. Built with scikit-build-core + CMake;
glfw/volk/VMA/vk-bootstrap/stb come in via `FetchContent`.

The core was header-only until 0.27 and is now `.hpp` + `.cpp` pairs — a header holds the
class, the short accessors, the templates and the API comments; the `.cpp` holds the bodies
and the comments that explain them. **A new `.cpp` in `src/` must be added to the
`add_library(_core MODULE ...)` list in `CMakeLists.txt`**, or its symbols link-fail with no
other warning. Small headers (`Format`, `Error`, `Device`, `Sampler`, `ScopeGuard`,
`ResourceTracker`, `SpirvReflect`, `Logger`, `Features`, `ImmediateSubmit`, `HotReload`)
stay header-only on purpose. See DESIGN.md for why the split happened at all.

Note: the working directory is still named `lumapy` and `origin` still points at
`lumapy.git` (redirects); the project and the GitHub repo are `SzamockiP/bazalt`.

## Build & test

Editing anything in `src/` requires a rebuild. `out/build/` is a plain CMake configure and
does **not** produce a usable `.pyd` — always go through scikit-build:

```bash
venv/Scripts/python.exe -m pip install . --no-build-isolation
```

Then copy the fresh binary over the in-tree one, because pytest runs from the repo root and
`import bazalt` resolves to `./bazalt/`, shadowing site-packages:

```bash
cp venv/Lib/site-packages/bazalt/_core.cp310-win_amd64.pyd bazalt/
```

Run the suite (headless, ~35s, needs a real Vulkan driver):

```bash
venv/Scripts/python.exe -m pytest -q
```

Single test / file:

```bash
venv/Scripts/python.exe -m pytest tests/test_render_to_layer.py -q -k multiview
```

C++ formatting: `.clang-format` at the root (LLVM base, Allman, 120 col,
`SortIncludes: Never`); clang-format 20 on PATH. Since 0.22 CI **gates** on it, so run it
before pushing — the same command the `format` job runs:

```bash
clang-format --dry-run -Werror src/*.hpp src/*.cpp src/bindings/*.hpp src/bindings/*.cpp
```

Since 0.27 two more gates run in CI. **clang-tidy** (`.clang-tidy` at the root, pinned to
20.1) analyses the first-party TUs; the house rule is that a check and the code disagreeing
means **the code changes** (the `py::class_` registrations got names rather than a `NOLINT`),
that a check which only PARTLY misfires gets narrowed rather than switched off, and that
every disabled or narrowed entry in the config names its reason. Do not disable a check
before you have seen it fire — the 0.27 plan wanted `modernize-make-shared` off and the check
turned out to have nothing to say about a private constructor behind a factory.

**Its `--fix` output is a suggestion, not a patch.** The 0.27 run produced an extra
parenthesis in three `modernize-use-integer-sign-comparison` rewrites and a variable named
`give_me_a_name` four times. Build and run the suite after every `--fix`, and read the diff.
Running `--fix` twice over the same file also duplicates the includes it inserts.

A local Windows run needs the MSVC environment, or every file fails to parse with
`'array' file not found`: the compile database is clang-cl's and the STL include paths come
from `vcvars64.bat`. Run it through a batch file that calls vcvars first, or the 5000
warnings you get are all from a broken parse.

Such a run reports findings CI does not: MSVC deprecates `getenv` (glibc does
not), and the three call sites in `Context.cpp` are the negotiation knobs. `std::getenv` is
the portable spelling and stays; `_dupenv_s` is a Microsoft extension.
**ruff** checks the Python side with an explicit rule set (its defaults move between
versions). Locally:

```bash
venv/Scripts/python.exe -m ruff check .
```

clang-tidy needs a compile database, which the normal scikit-build build does not leave on
disk. The CI recipe is in `.github/workflows/build.yml` (`tidy` job); locally it needs a
Ninja configure with `-DSKBUILD_PROJECT_NAME=bazalt` (the project is named from a
scikit-build variable) and `-DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON` (clang-tidy cannot read
the MSVC/GCC PCH flags).

Which public symbols no test touches (0.22, the input to the 1.0 test push). Fails on an
untouched symbol that is not in `tests/api_coverage_baseline.txt`, which is what catches a
binding shipped without a test. Since 0.27 it counts CALLABLES only — enum members and
exception classes could only ever be matched by a name scan, which counted a mention in a
comment as a use.
It needs a run with **no skips** — a skipped test is a symbol nobody called on this machine,
not a symbol nobody tested — so the gate is live on a developer GPU and silent in CI.
Regenerate the baseline with `BAZALT_WRITE_API_BASELINE=1`:

```bash
venv/Scripts/python.exe -m pytest -q --api-coverage
```

CI (`.github/workflows/build.yml`) has eight jobs: `format`, `sdist`, wheels for cp310–cp314
on manylinux + Windows + macOS arm64, the full suite on Mesa **lavapipe** (Ubuntu) and on
**MoltenVK** (macos-14, real Apple Silicon), and `publish`. Every wheel runs `test_stubs.py`
through `CIBW_TEST_COMMAND`; only the two test jobs render anything.

lavapipe runs twice, once with `BAZALT_FORCE_VULKAN_1_2=1`, because lavapipe reports 1.3 and
the 1.2 + KHR-alias path would otherwise go untested. That env var is a test knob, not
public API: it makes `Context` negotiate 1.2 wherever 1.3 exists, so the same path is
reproducible locally on any driver:

```bash
BAZALT_FORCE_VULKAN_1_2=1 venv/Scripts/python.exe -m pytest -q
```

`BAZALT_FORCE_SINGLE_QUEUE=1` is the same kind of knob and 0.29 added it for the same reason:
CI's drivers report ONE queue family, so the aliased path is what runs there and the
separate-queue path is what runs on a developer GPU. This forces the alias on hardware that
has more families, so both halves are reachable in one place. Since 0.30 it forces the
TRANSFER alias too. `list_devices()` ignores it, because that one reports the hardware.

```bash
BAZALT_FORCE_SINGLE_QUEUE=1 venv/Scripts/python.exe -m pytest -q
```

The macOS SDK install is a composite action (`.github/actions/install-vulkan-sdk-macos`)
because both macOS jobs need it. macOS has no system Vulkan, so nothing there works without
the LunarG SDK — that is the platform's contract, not a CI detail.

## Architecture

The layering exists so nothing below `Renderer.hpp` knows swapchains exist.

- **`Context.hpp`** — owns the device and everything per-Context: the **device dispatch
  table** (`ctx.vk()`), VMA allocator, the frame ring (a monotonic *serial*, not a wrapping
  index), the serial-keyed deferred destruction queue, the sampler cache, and debug-name
  plumbing. Since 0.28 a queue is a **`QueueRuntime`** — handle, family and its flags, the
  legal barrier stages of that family, timeline semaphore, reserved and submitted serial
  counters, a mutex POINTER and a command pool. **Three are constructed since 0.30**
  (graphics, compute, transfer), reached through `runtimes_[queue_index(kind)]`. Where the
  device has a compute-only or transfer-only family that runtime is a real queue; where it
  has neither, the runtime aliases the graphics `VkQueue` and its mutex (external
  synchronization is about the queue, so a second lock over one handle would be a race) and
  keeps its own timeline and pool — so the cross-queue path runs on every driver, including
  CI's single-family ones. The transfer runtime never borrows the COMPUTE family: a staging
  copy is DMA, not shader work. `Feature::ASYNC_COMPUTE` / `ASYNC_TRANSFER` report which you
  have; `BAZALT_FORCE_SINGLE_QUEUE=1` forces both aliases. One timeline per runtime because
  two queues cannot keep one counter strictly
  increasing without waiting on each other — NOT because the spec forbids co-signalling, which
  is the binary rule and what 0.28's comment wrongly quoted. The deletion-queue key and the
  ring's slot serials are per queue (`retire_key()`, `note_slot_submit`), and `lock_queues()`
  takes every non-aliasing mutex for whoever idles the device. **A per-queue sentinel array is
  written `per_queue(value)`, never a brace list** — a short brace list value-initializes the
  rest, and `kNoBatch` becoming 0 in the third slot is a silent wait no test would catch. Both windowed and headless submits advance
  the same ring, and both record into it. Since 0.15 any
  number of Contexts may be alive: **every device-level `vk*` call goes through
  `ctx.vk()`**, and `create_instance_` calls `volkLoadInstanceOnly`, so the device-level
  globals stay null and a call site that skipped the table crashes instead of silently
  using another Context's device. Instance-level calls stay on the globals on purpose
  (loader trampolines, dispatched on the handle). Recorded lambdas get the table from
  `FrameContext::vk`.
- **`RenderTarget.hpp`** — abstract "anything drawable": which colour attachments, which
  depth attachment, extent, and the layout the result must end in. `OffscreenTarget`,
  `SubresourceTarget` (`target.layer(i, mip=)`), `MultiviewTarget`
  (`target.all_layers()`) and `SwapchainRenderer` all implement it, which is why the same
  pipeline and pass work against a window, an offscreen image, or one cube face. A render
  pass **infers** subresource/multiview/viewport from the target — no knobs on the verb.
- **`Graph.hpp`** — THE way to describe work since 0.28. A `Graph` holds `Pass` objects in
  add order (never reordered), groups consecutive passes on one queue into **batches** (0.29),
  owns one `VkCommandBuffer` per batch per ring slot from that batch's queue pool, and
  compiles the barriers for the whole frame at the first submit after anything changed. A
  dependency that crosses a batch on another queue is a timeline wait rather than a barrier —
  `Context::submit_batches` emits one `vkQueueSubmit` per batch and threads the waits. `Graph::compile_`
  folds every enabled pass's `UseEvent`s through ONE `ResourceTracker`: a use inside the
  graph names its real predecessor and gets an exact edge, a render pass's barriers all go
  into its entry batch (one merged `vkCmdPipelineBarrier`), and the **look-ahead** elides
  the attachment retire/re-enter between two consecutive passes on one target when the
  second preserves. `Graph::execute` replays it. A `Pass` is a handle: `with` seals it,
  `p.enabled` and `g.remove(p)` recompile without re-recording.
- **`CommandBuffer.hpp`** — the engine *behind* a Pass, and nothing else since 0.28: it
  records *lambdas taking a `FrameContext`*, reports what they touch to the graph's event
  sink, and owns the query pools. It computes no barrier — it cannot, because whatever
  wrote what this pass reads was recorded by a different recorder.
- **`ResourceTracker.hpp`** — the hazard state machine the graph's fold drives (it used to
  run per recording). Image state is per `(layer, mip)` since 0.30, in the
  `SubresourceLayouts` shape: one state while the image is uniform, a split on the first
  narrowed use (`set_image(layer=, mip=)`), a collapse when every entry agrees again — so an
  image nobody narrows takes the same path it always did, and a pyramid pass can sample mip
  N-1 while it writes mip N. `tracks()` answers per RANGE for the same reason. Its first-use floors now answer for writers OUTSIDE the graph —
  another graph, or the previous frame — which is the 0.24 argument one level up.
  `add_pass(auto_barriers=False)` hands one pass's hazards to `p.barrier()`, and those
  barriers still feed the fold. Attachment layout transitions are the compile's job and
  stay automatic regardless.
- **`Features.hpp`** — optional GPU capabilities addressed by what they *do*, never by
  version or extension name. Vulkan 1.2 baseline; on 1.2 devices the dynamic-rendering
  entry points are loaded under KHR names and aliased onto the core symbols
  (`Context::alias_dynamic_rendering_entry_points`), so call sites only use core names.
- **`UploadManager.hpp`** — one worker thread decodes images and submits the staging copy on
  the **transfer** runtime since 0.30; each submit signals that queue's timeline, so frames
  wait GPU-side. A MIPPED upload is two submits: the copy on transfer, then the blit cascade
  on graphics waiting the copy's serial, because `vkCmdBlitImage` needs a graphics family.
  So `upload_serial_` is a `QueueSerials` on both `Buffer` and `Image`, and
  `require_uploads_resident` returns one. A reload or an `image.update` also waits the
  graphics queue's newest submitted serial — the frames still sampling the image used to be
  ordered by a fragment-shader source scope, which a transfer-only family may not name — and
  marks the image PENDING, so nothing reads between the two halves of a split upload.
- **`HotReload.hpp`** — watches the shaders (plus `#include`s) and images bazalt itself
  loaded, recompiling/re-uploading in place; a bad edit logs and keeps the last good
  version. Drained on the main thread from `begin_frame` / `ctx.submit`.
- **`Error.hpp` + `bindings/Common.hpp`** — every fallible C++ operation returns
  `std::expected<T, Error>`; `ErrorCode` maps 1:1 onto a Python exception class at the
  pybind boundary via the `unwrap()` helpers. The exception *type* is the recoverability
  contract (`ShaderError` must be catchable alone or hot reload is pointless).

Since 0.20 the binding layer is `src/bindings/`, one file per subject
(`Enums`/`Resources`/`Pipelines`/`Commands`/`Graphs`/`Windowing`/`ContextBind`/`Targets`),
plus `Common.hpp` for what they share and `Pch.hpp` for the third-party headers.
`Commands.cpp` binds `Pass` (the verbs) and `Graphs.cpp` binds `Graph`/`Queue`/`Serial`.
`main.cpp` is only the `PYBIND11_MODULE` and the eight calls, **in an order that is
load-bearing** — the comments there say why, and `bind_commands` must precede
`bind_graphs` because `add_pass` returns the class the first one registers. A new binding goes in the file that owns its subject; a new shared
helper goes in `Common.hpp` and must be `inline`, never an anonymous namespace (each TU
would get its own copy of the `exc_*` handles and `raise_error` would go through a null
one — it links and crashes at runtime).

## Conventions that bite if ignored

- **Validation-as-assert.** The `ctx` fixture in `tests/conftest.py` is session-scoped
  (every resource holds its Context alive, so a per-test Context is a leak trap) and fails
  any test where the validation layers emitted an ERROR from `Source.VALIDATION`. That,
  not eyeballed pixels, is the referee for barriers and layouts. A test needing a Context
  of its own (multi-context, `validation="sync"`) asks the `extra_context` factory, which
  applies the same referee.
- **`bazalt/_core.pyi` is hand-written.** Any binding added, renamed or removed in
  `src/bindings/` also needs the stub, plus `__all__` and the explicit re-export list in
  `bazalt/__init__.py`. Every `.def()` parameter needs a `py::arg()` too, or only a
  positional call works while the stub advertises a keyword one. `tests/test_stubs.py` is what catches the drift (and it's the only
  test CI runs on each wheel).
- **README code is executed** by `tests/test_readme.py` — edit the snippets and the test
  together.
- **User-facing prose is Simplified Technical English** (ASD-STE100, "STE-flavored" mode):
  active voice, one idea per sentence, no semicolons, short common words, no marketing
  adjectives, one name per thing. This covers `README.md`, `CHANGELOG.md`, exception message
  text, docstrings and PR descriptions. The `ste-writing` skill applies it, but the rule is
  the convention, not the tool — the skill is a user-level plugin and a fresh clone will not
  have it. Code, identifiers and the comments inside snippets are out of scope. So are this
  file and `DESIGN.md`: they argue and justify, and STE flattens exactly that.
- `CHANGELOG.md` is the "what" (per release, chronological); `DESIGN.md` is the "why"
  (per subject, no timeline) — design rules, the scope test, the decisions that still bind
  the code, the debt register, the accepted ceilings, the proposed features, and the
  rejected ideas with their reasons. Both are tracked. Read `DESIGN.md` before planning a
  release or arguing about API shape, and put every new decision there **with** its
  justification. A decision without a reason gets undone later.
- **API design rules** (from `DESIGN.md`, they settle most arguments): one obvious way per
  thing — a new resource variant differing by one parameter is a kwarg on the existing
  function, not a new name (`create_image(..., cube=True)`); never cap the ceiling (leave an
  escape hatch like a manual pass with `p.barrier()`, `submit(after=)`, `raw_extensions`);
  must run on >90% of machines
  (1.2 baseline + additive `Feature` negotiation); when a Vulkan primitive is an
  index/handle, return a handle and read results off it — no string keys.
- Releases go on a `release/X.Y.Z` branch, one minor = one big feature + small related
  additions. Claude pushes the branch; the user does the PR, merge and tag.
