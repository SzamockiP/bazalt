# Changelog

All notable changes to **bazalt** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/) (pre-1.0: minor versions may break the API,
patch versions never do).

## [Unreleased] — 0.4.1

A source-quality release: bug fixes, refactoring, and C++23 adoption.
No public Python API changes.

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

### Changed
- CI now builds `release/**` branches and smoke-tests every built wheel
  (`import bazalt` + stub consistency); the GPU test suite remains local.

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
