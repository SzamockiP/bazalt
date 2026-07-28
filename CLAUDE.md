# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**bazalt** — a Python library for rapid GPU/shader prototyping on Vulkan. A header-only
C++23 core in `src/` compiled into a single pybind11 extension (`bazalt._core`), plus a
thin Python package in `bazalt/` that only re-exports it. Built with scikit-build-core +
CMake; glfw/volk/VMA/vk-bootstrap/stb come in via `FetchContent`.

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
`SortIncludes: Never`); clang-format 20 on PATH.

CI (`.github/workflows/build.yml`) builds wheels for cp310–cp313 on Windows + manylinux,
runs only `test_stubs.py` on each wheel, and runs the full suite on Ubuntu against Mesa
**lavapipe** — twice, once with `BAZALT_FORCE_VULKAN_1_2=1`, because lavapipe reports 1.3
and the 1.2 + KHR-alias path would otherwise go untested. That env var is a test knob, not
public API: it makes `Context` negotiate 1.2 wherever 1.3 exists, so the same path is
reproducible locally on any driver:

```bash
BAZALT_FORCE_VULKAN_1_2=1 venv/Scripts/python.exe -m pytest -q
```

## Architecture

The layering exists so nothing below `Renderer.hpp` knows swapchains exist.

- **`Context.hpp`** — owns the device and everything per-Context: the **device dispatch
  table** (`ctx.vk()`), VMA allocator, command pool, the frame ring (a monotonic *serial*,
  not a wrapping index), one **timeline semaphore counting every submit on the graphics
  queue**, the serial-keyed deferred destruction queue, the sampler cache, and debug-name
  plumbing. Both windowed and headless submits advance the same ring. Since 0.15 any
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
  pipeline and command buffer work against a window, an offscreen image, or one cube face.
  `begin_rendering` **infers** subresource/multiview/viewport from the target — no knobs on
  the verb.
- **`CommandBuffer.hpp`** — records *lambdas taking a `FrameContext`* and replays them on
  every submit, so one recording serves any target and any frame slot.
- **`ResourceTracker.hpp`** — record-time hazard tracking that inserts barriers
  automatically (`Context(auto_barriers=False)` hands it all to `cmd.barrier()`). Attachment
  layout transitions are the RenderTarget contract and stay automatic regardless. Known
  ceiling: no SPIR-V reflection, so SSBO/`imageStore` writes from *graphics* shaders are
  untracked (debt #3).
- **`Features.hpp`** — optional GPU capabilities addressed by what they *do*, never by
  version or extension name. Vulkan 1.2 baseline; on 1.2 devices the dynamic-rendering
  entry points are loaded under KHR names and aliased onto the core symbols
  (`Context::alias_dynamic_rendering_entry_points`), so call sites only use core names.
- **`UploadManager.hpp`** — one worker thread decodes images and submits copies/mipgen on
  the graphics queue; each submit signals the Context timeline, so frames wait GPU-side.
- **`HotReload.hpp`** — watches the shaders (plus `#include`s) and images bazalt itself
  loaded, recompiling/re-uploading in place; a bad edit logs and keeps the last good
  version. Drained on the main thread from `begin_frame` / `ctx.submit`.
- **`Error.hpp` + `main.cpp`** — every fallible C++ operation returns
  `std::expected<T, Error>`; `ErrorCode` maps 1:1 onto a Python exception class at the
  pybind boundary via the `unwrap()` helpers. The exception *type* is the recoverability
  contract (`ShaderError` must be catchable alone or hot reload is pointless).

`main.cpp` is the entire binding layer (one `PYBIND11_MODULE`), so it's the file that grows
with every API addition.

## Conventions that bite if ignored

- **Validation-as-assert.** The `ctx` fixture in `tests/conftest.py` is session-scoped
  (every resource holds its Context alive, so a per-test Context is a leak trap) and fails
  any test where the validation layers emitted an ERROR from `Source.VALIDATION`. That,
  not eyeballed pixels, is the referee for barriers and layouts. A test needing a Context
  of its own (multi-context, `validation="sync"`) asks the `extra_context` factory, which
  applies the same referee.
- **`bazalt/_core.pyi` is hand-written.** Any binding added, renamed or removed in
  `main.cpp` also needs the stub, plus `__all__` and the explicit re-export list in
  `bazalt/__init__.py`. `tests/test_stubs.py` is what catches the drift (and it's the only
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
  escape hatch like `cmd.barrier()` / `raw_extensions`); must run on >90% of machines
  (1.2 baseline + additive `Feature` negotiation); when a Vulkan primitive is an
  index/handle, return a handle and read results off it — no string keys.
- Releases go on a `release/X.Y.Z` branch, one minor = one big feature + small related
  additions. Claude pushes the branch; the user does the PR, merge and tag.
