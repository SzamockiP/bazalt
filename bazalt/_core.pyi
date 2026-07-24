from __future__ import annotations

from enum import IntEnum
from typing import Any, Callable, Optional, Sequence

import numpy as np

# ── Errors ─────────────────────────────────────────────────────────────
#
# The exception type is the contract for whether you can carry on: a shader typo
# is recoverable, a lost device is not.

class BazaltError(Exception):
    """Base class for every error bazalt raises."""
    ...

class InitializationError(BazaltError):
    """No Vulkan, no suitable GPU, or a required Feature is missing. Fatal."""
    ...

class DeviceLostError(BazaltError):
    """VK_ERROR_DEVICE_LOST. The Context is unusable afterwards."""
    vk_result: str

class OutOfMemoryError(BazaltError):
    """Host or device memory exhausted. Sometimes recoverable."""
    vk_result: str

class ShaderError(BazaltError):
    """Compilation or pipeline creation failed. Recoverable."""
    path: str
    line: int
    """1-based line number, or -1 when it could not be determined."""

class WindowError(BazaltError):
    """Window or surface creation failed; carries the platform's own message."""
    ...

class ResourceError(BazaltError):
    """Missing file, bad format, or an exhausted pool. Recoverable."""
    ...

# ── Logging ────────────────────────────────────────────────────────────

class Severity(IntEnum):
    """Ordered, so `msg.severity >= Severity.WARNING` works."""
    INFO = 0
    WARNING = 1
    ERROR = 2

class Source(IntEnum):
    """Which subsystem produced a message, so callbacks can route without
    pattern-matching on the text."""
    GENERAL = 0
    VALIDATION = 1
    WINDOW = 2
    SHADER = 3
    UPLOAD = 4
    DEVICE = 5

class LogMessage:
    """A log message with its severity as data rather than a string prefix."""

    @property
    def severity(self) -> Severity: ...
    @property
    def source(self) -> Source: ...
    @property
    def text(self) -> str: ...

class Logger:
    def __init__(self, min_severity: Severity = Severity.WARNING) -> None: ...

    def on_message(self, callback: Callable[[LogMessage], None]) -> Callable[[LogMessage], None]:
        """Register a callback. Returns it, so this works as a decorator."""
        ...

    def log(self, text: str, severity: Severity = Severity.INFO,
            source: Source = Source.GENERAL) -> None: ...

    def flush(self) -> None:
        """Block until every queued message has reached its callbacks.

        Delivery is asynchronous, so without this, asserting that nothing was
        logged only asserts that nothing had arrived yet.
        """
        ...

    min_severity: Severity

# ── Capabilities ───────────────────────────────────────────────────────

class Feature(IntEnum):
    """Optional GPU capabilities, named by what they do.

    Vulkan promotes extensions into core versions, so the same capability is
    spelled differently per driver. Which spelling to use is bazalt's problem.
    """
    ANISOTROPIC_FILTERING = 0
    WIREFRAME = 1
    WIDE_LINES = 2
    DEPTH_CLAMP = 3
    SAMPLE_RATE_SHADING = 4
    MULTI_DRAW_INDIRECT = 5
    SHADER_FLOAT64 = 6

# ── Enums ──────────────────────────────────────────────────────────────

class BufferType(IntEnum):
    VERTEX = 0
    INDEX = 1
    UNIFORM = 2
    STORAGE = 3

class DataType(IntEnum):
    FLOAT = 0
    UINT32 = 1
    UINT16 = 2
    INT32 = 3

class ShaderStage(IntEnum):
    VERTEX = 0
    FRAGMENT = 1
    COMPUTE = 2

class VertexFormat(IntEnum):
    """Vertex attribute layout. Renamed from `Format`, which is reserved for
    pixel formats."""
    FLOAT2 = 0
    FLOAT3 = 1
    FLOAT4 = 2

class Topology(IntEnum):
    """Primitive topology for graphics pipelines. TRIANGLE_LIST is the default."""
    TRIANGLE_LIST = 0
    POINT_LIST = 1
    LINE_LIST = 2

class Access(IntEnum):
    """What a command does to a buffer — the vocabulary of cmd.barrier()
    in manual mode (auto_barriers=False)."""
    SHADER_READ = 0
    SHADER_WRITE = 1
    VERTEX_READ = 2
    INDEX_READ = 3
    UNIFORM_READ = 4

class Format(IntEnum):
    """Pixel formats.

    RGBA8 is data (UNORM, what arrays and render targets default to);
    RGBA8_SRGB is pictures (what load_image decodes into).
    """
    RGBA8 = 0
    RGBA8_SRGB = 1
    BGRA8 = 2
    R8 = 3
    RG8 = 4
    R16F = 5
    RGBA16F = 6
    R32F = 7
    RGBA32F = 8
    D32F = 9

class Filter(IntEnum):
    LINEAR = 0
    NEAREST = 1

class AddressMode(IntEnum):
    REPEAT = 0
    CLAMP = 1
    MIRROR = 2

class CompareOp(IntEnum):
    """Comparison for compare samplers (sampler2DShadow), 1:1 with VkCompareOp.

    The sampler returns the comparison RESULT (1.0 = pass): with LESS, 1.0
    where the reference depth is closer than the stored texel. LINEAR filtering
    on a compare sampler averages four results — hardware PCF."""
    NEVER = 0
    LESS = 1
    EQUAL = 2
    LESS_OR_EQUAL = 3
    GREATER = 4
    NOT_EQUAL = 5
    GREATER_OR_EQUAL = 6
    ALWAYS = 7

class PresentMode(IntEnum):
    """How presentation paces the frame loop.

    FIFO is vsync and the only mode Vulkan guarantees; MAILBOX (the default
    preference) and IMMEDIATE fall back to FIFO with an Info log when the
    surface cannot do them."""
    FIFO = 0
    MAILBOX = 1
    IMMEDIATE = 2

class CullMode(IntEnum):
    NONE = 0
    BACK = 1
    FRONT = 2
    FRONT_AND_BACK = 3

class FrontFace(IntEnum):
    CLOCKWISE = 0
    COUNTER_CLOCKWISE = 1

class MemoryUsage(IntEnum):
    STATIC = 0
    DYNAMIC = 1

class MouseState:
    @property
    def dx(self) -> float: ...
    @property
    def dy(self) -> float: ...

# ── Resources ──────────────────────────────────────────────────────────

class Buffer:
    def update(self, data: bytes) -> None: ...
    def update(self, array: Any) -> None:
        """Upload from any C-contiguous buffer-protocol object.

        Raises ResourceError for a strided view (`arr.T`, `arr[::2]`): copying
        silently would hide an allocation on every upload. Pass
        `numpy.ascontiguousarray(arr)` to be explicit.
        """
        ...
    def update(self, list: list, data_type: Optional[DataType] = None) -> None: ...

    def read(self, dtype: Any) -> Any:
        """Copy the buffer back to host memory as a 1-D numpy array.

        dtype is mandatory — buffers carry no format, so the caller says how
        to interpret the bytes (e.g. `ssbo.read(np.float32)`). STATIC buffers
        take a blocking GPU round trip; DYNAMIC ones return what update()
        last wrote into the current frame's copy.
        """
        ...

class ShaderModule:
    """A compiled (or loaded) shader. See Context.compile_shader."""
    @property
    def path(self) -> str:
        """The source path — or the virtual name for in-memory sources."""
        ...
    @property
    def includes(self) -> list[str]:
        """Files pulled in via #include, absolute and normalized.

        Empty for .spv modules and include-free sources. Together with `path`
        this is the full file set the shader was built from."""
        ...
    @property
    def spirv(self) -> bytes:
        """The SPIR-V words. `open(p, "wb").write(shader.spirv)` produces a
        file that compile_shader("*.spv", stage) loads back."""
        ...

class Image:
    """A GPU image: pixels + format. May be 2D, a texture array (array_layers >
    1) or a cubemap (is_cube). The sampler it used to be fused with is a
    separate (cached) object — see Context.create_sampler."""

    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def format(self) -> Format: ...
    @property
    def mip_levels(self) -> int: ...
    @property
    def array_layers(self) -> int:
        """Number of layers: 1 for a plain 2D image, N for a texture array,
        6 for a cubemap."""
        ...
    @property
    def is_cube(self) -> bool:
        """True for a cubemap (sampled through a CUBE view as samplerCube)."""
        ...
    @property
    def samples(self) -> int:
        """MSAA sample count (1/2/4/…). >1 only for the multisampled attachment a
        RenderTarget owns internally; the images it exposes (target.color/depth)
        are the resolved single-sample ones and always report 1."""
        ...
    @property
    def ready(self) -> bool:
        """Non-blocking: is the pixel data on the GPU?

        False while a load_image decode/copy is still in flight. You never
        have to poll this — a submit that uses the image waits automatically;
        it exists for loading screens and explicit control.
        """
        ...

    def wait(self) -> None:
        """Block until this image's upload has finished.

        A failed decode (corrupt file) surfaces here as ResourceError.
        """
        ...

    def read(self) -> Any:
        """Copy mip 0 back to host memory as a numpy array (layer 0 for a
        texture array / cubemap — sample the other layers to inspect them).

        Shape is (height, width, channels) — or (height, width) for
        single-channel formats — and the dtype follows the format (uint8,
        float16 or float32). Blocking; a debugging and test path.

        Raises ResourceError if the image has no contents yet.
        """
        ...

class Sampler:
    """How to read texels. Cached on the Context: identical descriptions are
    the identical object."""
    ...

class Pipeline: ...

class DescriptorSet:
    def set_image(self, binding: int, image: Image,
                  sampler: Optional[Sampler] = None) -> None:
        """Bind an image (+ sampler; None means linear/repeat/anisotropic)."""
        ...
    def set_storage_image(self, binding: int, image: Image) -> None:
        """Bind a storage image (no sampler) to a binding declared with
        .storage_image(). The image is accessed in GENERAL layout; the tracker
        adds the transition and any barrier around the dispatch automatically."""
        ...
    def set_buffer(self, binding: int, buffer: Buffer) -> None: ...

class DescriptorPool:
    def allocate_set(self, pipeline: Pipeline, set: int) -> DescriptorSet: ...
    def allocate_frame_set(self, pipeline: Pipeline, set: int) -> DescriptorSet: ...

# ── Render targets ─────────────────────────────────────────────────────

class RenderTargetBase:
    """Anything that can be drawn into.

    Both `RenderTarget` and `SwapchainRenderer` are one of these: presenting to
    a window is one way to consume a rendered image, not the definition of
    rendering.
    """
    ...

class RenderTarget(RenderTargetBase):
    """An offscreen target backed by its own Images. No window required.

    The attachments are ordinary Images: `target.color[0]` and `target.depth`
    go straight into DescriptorSet.set_image — that is the whole
    render-to-texture and shadow-map API.
    """

    def __init__(self, context: Context, width: int, height: int,
                 color: Optional[Format | Sequence[Format]] = Format.RGBA8,
                 depth: Optional[Format] = None,
                 samples: int = 1, name: str = "") -> None:
        """color=None with depth=D32F makes a depth-only (shadow) target;
        a list of formats makes an MRT target. At least one attachment is
        required.

        samples>1 turns on MSAA: the target renders into a multisampled image and
        resolves into target.color/target.depth (which stay single-sample and
        sampleable — depth resolves too, via SAMPLE_ZERO). Must be a power of two
        <= ctx.max_samples(). name labels the attachments in validation messages.
        """
        ...

    @property
    def color(self) -> tuple[Image, ...]: ...
    @property
    def depth(self) -> Optional[Image]: ...

    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...

    def read_pixels(self) -> np.ndarray:
        """Copy the colour attachment back to host memory as (height, width, 4) uint8.

        Blocking, and it stalls the GPU — intended for tests and debugging.
        """
        ...

class GraphicsPipelineBuilder:
    def vertex_shader(self, shader: ShaderModule) -> GraphicsPipelineBuilder: ...
    def fragment_shader(self, shader: ShaderModule) -> GraphicsPipelineBuilder: ...
    def vertex_format(self, formats: list[VertexFormat]) -> GraphicsPipelineBuilder: ...
    def depth_test(self, enable: bool) -> GraphicsPipelineBuilder: ...
    def cull_mode(self, mode: CullMode, front_face: FrontFace) -> GraphicsPipelineBuilder: ...
    def blend(self, enable: bool) -> GraphicsPipelineBuilder: ...
    def topology(self, topology: Topology) -> GraphicsPipelineBuilder: ...
    def sample_shading(self, enable: bool = True, min_fraction: float = 1.0) -> GraphicsPipelineBuilder:
        """Per-sample fragment shading on an MSAA target: the fragment shader runs
        once per sample instead of once per pixel, cleaning up interior/specular
        aliasing plain MSAA leaves. Requires the SAMPLE_RATE_SHADING feature on the
        Context (build() raises ShaderError otherwise). The sample count itself
        comes from the target build() is called with — there is no samples knob
        here."""
        ...
    def push_constant(self, size: int, stage: ShaderStage) -> GraphicsPipelineBuilder: ...
    def uniform_buffer(self, binding: int, stage: ShaderStage, set: int) -> GraphicsPipelineBuilder: ...
    def storage_buffer(self, binding: int, stage: ShaderStage, set: int) -> GraphicsPipelineBuilder: ...
    def texture(self, binding: int, stage: ShaderStage, set: int) -> GraphicsPipelineBuilder: ...
    def name(self, name: str) -> GraphicsPipelineBuilder:
        """Debug name for the VkPipeline (validation diagnostics). No-op without
        VK_EXT_debug_utils, i.e. when validation is off."""
        ...

    def build(self, target: RenderTargetBase) -> Pipeline:
        """Build against any render target — a window or an offscreen image.

        The built Pipeline keeps its ShaderModules alive so hot reload can
        rebuild it in place."""
        ...

class ComputePipelineBuilder:
    """No stage arguments anywhere: compute has exactly one stage."""

    def shader(self, shader: ShaderModule) -> ComputePipelineBuilder: ...
    def uniform_buffer(self, binding: int, set: int = 0) -> ComputePipelineBuilder: ...
    def storage_buffer(self, binding: int, set: int = 0) -> ComputePipelineBuilder: ...
    def storage_image(self, binding: int, set: int = 0) -> ComputePipelineBuilder:
        """A read/write image the compute shader accesses by coordinate
        (imageLoad/imageStore). Bind one with DescriptorSet.set_storage_image;
        the auto-barrier tracker transitions it to GENERAL before the dispatch
        and to SHADER_READ_ONLY before a later graphics sample."""
        ...
    def push_constant(self, size: int) -> ComputePipelineBuilder: ...
    def name(self, name: str) -> ComputePipelineBuilder:
        """Debug name for the VkPipeline; no-op without VK_EXT_debug_utils."""
        ...

    def build(self) -> Pipeline:
        """No target — compute has no attachments."""
        ...

class CommandBuffer:
    """Records commands once; they are replayed on every submit.

    Every recording method returns the command buffer itself, so calls chain:

        cmd.begin_rendering(target).bind_pipeline(p).draw(3).end_rendering(target)

    The statement-per-line style works identically — the return value is the
    same object and ignoring it costs nothing.
    """

    def begin(self) -> CommandBuffer: ...

    def begin_rendering(self, target: RenderTargetBase,
                        clear_color: Sequence[float] | Sequence[Sequence[float]] = (0.0, 0.0, 0.0, 1.0)
                        ) -> CommandBuffer:
        """Start rendering into `target`.

        clear_color is either a single [r, g, b, a] applied to every attachment
        (the common case) or, for MRT, a list of them ([[r,g,b,a], …]) clearing
        each attachment independently.

        Also emits a viewport and scissor covering the whole target, so the
        common case needs no further calls.
        """
        ...

    def end_rendering(self, target: RenderTargetBase) -> CommandBuffer: ...

    def rendering(self, target: RenderTargetBase,
                  clear_color: Sequence[float] | Sequence[Sequence[float]] = (0.0, 0.0, 0.0, 1.0)
                  ) -> RenderingScope:
        """The begin/end pair as a context manager:

            with cmd.rendering(target, clear_color=[0, 0, 0, 1]) as c:
                c.bind_pipeline(p).draw(3)

        end_rendering is recorded on exit, exceptions included. clear_color takes
        the same single-or-per-attachment forms as begin_rendering.
        """
        ...

    def set_viewport(self, x: float, y: float, width: float, height: float) -> CommandBuffer:
        """Override the automatic full-target viewport (split-screen and similar)."""
        ...

    def set_scissor(self, x: int, y: int, width: int, height: int) -> CommandBuffer: ...

    def bind_pipeline(self, pipeline: Pipeline) -> CommandBuffer: ...
    def bind_vertex_buffer(self, buffer: Buffer) -> CommandBuffer: ...
    def bind_index_buffer(self, buffer: Buffer) -> CommandBuffer: ...
    def draw(self, vertex_count: int) -> CommandBuffer: ...
    def draw_indexed(self, index_count: int, first_index: int = 0,
                     vertex_offset: int = 0) -> CommandBuffer: ...
    def draw_indexed_instanced(self, index_count: int, instance_count: int,
                               first_index: int = 0, vertex_offset: int = 0) -> CommandBuffer: ...
    def dispatch(self, group_count_x: int, group_count_y: int = 1,
                 group_count_z: int = 1) -> CommandBuffer: ...

    def barrier(self, buffer: Buffer, src: Access, dst: Access) -> CommandBuffer:
        """Record a buffer barrier by hand. Required between dependent uses
        when auto_barriers=False; legal (if redundant) in auto mode. Refused
        inside a rendering scope — record it before begin_rendering."""
        ...
    def barrier(self, image: Image, src: Access, dst: Access) -> CommandBuffer:
        """Transition an image between shader accesses by hand, across every mip
        and layer. The layout follows the access: SHADER_WRITE = GENERAL (a
        storage image), SHADER_READ = SHADER_READ_ONLY (a sampled image); other
        accesses are buffer-only.

        The one case the automatic tracker can't reach is cross-submit: a compute
        shader bakes an image in one submit (GENERAL) and later frames sample it
        (SHADER_READ_ONLY). Generate it once, then
        `cmd.barrier(image, Access.SHADER_WRITE, Access.SHADER_READ)` after the
        dispatch, and sample it every frame without regenerating. In auto mode
        this also updates the tracker, so mixing it with automatic uses of the
        same image in one recording is safe. Refused inside a rendering scope."""
        ...

    def generate_mipmaps(self, image: Image, *,
                         src: Access = Access.SHADER_READ) -> CommandBuffer:
        """Fill mip levels 1..N of a mipped image by blitting mip 0 down the chain
        (every array layer / cube face at once), leaving every level sampleable.

        The pair to create_image(..., mip_levels=N): write mip 0 (upload, a
        compute imageStore, or a render pass), then generate the rest here. `src`
        names mip 0's current layout in cmd.barrier's vocabulary — SHADER_READ
        (SHADER_READ_ONLY, an uploaded or already-baked image; the default) or
        SHADER_WRITE (GENERAL, mip 0 fresh from compute).

        Raises ResourceError if the image has a single level (create it with
        mip_levels>1 or mipmaps=True), if the format can't be blitted/linearly
        filtered, or if called inside a rendering scope."""
        ...

    def push_constants(self, pipeline: Pipeline, offset: int, data: bytes) -> CommandBuffer:
        """The Pipeline already knows which stages its range covers."""
        ...

    def bind_descriptor_set(self, descriptor_set: DescriptorSet, pipeline: Pipeline,
                            set: int) -> CommandBuffer: ...

    def timer(self) -> Timer:
        """Start a GPU timer and return its handle. Records a timestamp here;
        stop it with a `with` block or Timer.stop(), read it back with Timer.ms:

            with cmd.timer() as t:
                cmd.bind_pipeline(blur).dispatch(gx, gy)
            ...                     # or, without `with`:
            t = cmd.timer()         #   t = cmd.timer()
            ...                     #   cmd.dispatch(...)
            ctx.submit(cmd)         #   t.stop()
            print(t.ms)

        The handle is the identity — no names, no keys — so several, nested and
        overlapping timers all work. Unlike frame.gpu_time_ms this needs no
        window: the blocking headless submit means t.ms is ready as soon as
        submit() returns. Self-gating: the query pool exists only once a timer is
        used, so apps that don't time pay nothing."""
        ...

class RenderingScope:
    """Returned by CommandBuffer.rendering(); use it in a `with` statement."""

    def __enter__(self) -> CommandBuffer: ...
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool: ...

class Timer:
    """A GPU timer handle from CommandBuffer.timer(). Usable as a context
    manager (`with cmd.timer() as t:`) or stopped by hand (t.stop())."""

    def stop(self) -> None:
        """Record the closing timestamp. Idempotent; called for you on `with`
        exit."""
        ...
    @property
    def ms(self) -> Optional[float]:
        """Measured GPU time in milliseconds, or None if timestamps are
        unsupported, the command buffer was re-recorded since (the handle is
        stale), or the submit has not completed (a blocking headless submit
        always has)."""
        ...
    def __enter__(self) -> Timer: ...
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool: ...

class Window:
    def __init__(self, width: int, height: int, title: str,
                 logger: Optional[Logger] = None) -> None: ...
    def is_open(self) -> bool: ...
    def should_close(self) -> bool: ...
    def poll_events(self) -> None: ...
    def is_key_pressed(self, key: int) -> bool: ...
    def is_mouse_button_pressed(self, button: int) -> bool: ...
    def set_cursor_mode(self, mode: int) -> None: ...
    def get_mouse_state(self) -> MouseState: ...
    def set_title(self, title: str) -> None: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...

class Context:
    """The GPU device, and the factory for everything that lives on it.

    Only one Context may be alive per process: volk binds its global function
    pointers to a single device, so a second one silently corrupts the first.
    Creating them one after another is fine.
    """

    def __init__(self, logger: Optional[Logger] = None, validation: str = "auto",
                 features: Sequence[Feature] = (), optional: Sequence[Feature] = (),
                 frames_in_flight: int = 2,
                 raw_extensions: Sequence[str] = (),
                 auto_barriers: bool = True,
                 hot_reload: bool = False,
                 gpu_timing: bool = False) -> None:
        """
        Args:
            logger: defaults to one printing warnings to stderr.
            validation: "auto" (on when the layers are installed), "on", "off",
                or "sync" ("on" plus synchronization validation — the only mode
                that reports missing barriers; costly, for debugging).
            features: required. Gates GPU selection; InitializationError if absent.
            optional: enabled when present; query with `supports()`.
            frames_in_flight: how many frames may be recorded ahead of the GPU
                (1-4). 2 is the classic latency/throughput trade-off; 1 is
                useful for debugging.
            raw_extensions: escape hatch. You shouldn't need this.
            auto_barriers: barriers between resources (SSBO -> vertex read,
                dispatch -> dispatch) are computed automatically at record time.
                False makes every one of them your job via cmd.barrier().
                Attachment layout transitions stay automatic either way.
            hot_reload: watch the files you loaded — shaders (and their
                #includes) and images — and apply edits live. A changed shader
                recompiles and rebuilds its pipelines in place; a changed image
                re-uploads into the same handle (same size and format only). A
                bad edit (typo, wrong size, corrupt file) is logged and the last
                good version keeps rendering — a mistake never kills the app.
                Changes apply at begin_frame() and at ctx.submit().
            gpu_timing: record frame.gpu_time_ms (a timestamp pair around each
                windowed submit). Off by default because it is a profiling
                diagnostic — the pool reset and two writes ride in every frame's
                command buffer, and per-frame queries are not guaranteed free on
                every GPU. Left off, frame.gpu_time_ms is always None, no cost.
        """
        ...

    @property
    def logger(self) -> Logger: ...
    @property
    def frames_in_flight(self) -> int: ...
    @property
    def auto_barriers(self) -> bool: ...
    @property
    def device_name(self) -> str: ...
    @property
    def api_version(self) -> str: ...
    @property
    def headless(self) -> bool:
        """True when no windowing extensions were available, so no
        SwapchainRenderer can be created against this Context."""
        ...

    def supports(self, feature: Feature) -> bool: ...
    def max_samples(self) -> int:
        """The highest MSAA sample count (1/2/4/8/…) this GPU supports for both a
        colour and a depth attachment — the valid ceiling for RenderTarget(...,
        samples=) and SwapchainRenderer(..., samples=)."""
        ...

    def create_buffer(self, list: list, type: BufferType, usage: MemoryUsage,
                      data_type: Optional[DataType] = None, *, name: str = "") -> Buffer: ...
    def create_buffer(self, array: Any, type: BufferType, usage: MemoryUsage,
                      *, name: str = "") -> Buffer: ...
    def create_buffer(self, size_in_bytes: int, type: BufferType,
                      usage: MemoryUsage, *, name: str = "") -> Buffer: ...

    def graphics_pipeline(self) -> GraphicsPipelineBuilder: ...
    def compute_pipeline(self) -> ComputePipelineBuilder: ...
    def compile_shader(self, path: str, stage: ShaderStage, *,
                       source: Optional[str] = None) -> ShaderModule:
        """Compile or load a shader. One function for every form: the extension
        of `path` decides how it is handled.

        - `.hlsl` — HLSL (entry point `main`, one file per stage). Use
          `[[vk::binding(n, set)]]` on resources; bare `register()` piles
          everything into one Vulkan binding space.
        - `.spv` — a prebuilt SPIR-V binary: loaded, not compiled. `stage` is
          verified against the binary's entry points (ShaderError on mismatch).
        - anything else — GLSL.

        `source=` compiles the given string instead of reading a file; `path`
        becomes a virtual name that still picks the language, tags diagnostics
        (ShaderError.path) and anchors relative #include resolution (a name
        with no directory resolves includes against the working directory).

        GLSL `#include "x"` / `<x>` resolve relative to the directory of the
        including file, recursively; the files used are recorded in
        ShaderModule.includes. A missing top-level file is a ResourceError; a
        missing include is a ShaderError (the compiler discovered it, and the
        error is recoverable — fix the include and recompile).
        """
        ...

    def load_image(self, path: str, *, mipmaps: bool = True, name: str = "") -> Image:
        """Decode an image file into an sRGB GPU image, with a full mip chain by
        default (`mipmaps=False` for a single level — e.g. a UI sprite sampled
        1:1).

        Returns IMMEDIATELY: the file header is validated here (a missing or
        corrupt file raises ResourceError at this call), but the decode and
        GPU copy run on a background worker. The image is usable for
        recording right away — a submit that samples it waits for the upload
        automatically. `img.ready`, `img.wait()` and `ctx.wait_for_uploads()`
        are the explicit-control verbs.

        With hot_reload=True the file is watched: re-saving it re-uploads into
        this same image (same size and format only; a resize or corrupt file
        logs a warning and keeps the old contents).

        `name` attaches a debug name to the VkImage (no-op without validation).
        """
        ...

    def load_image(self, paths: Sequence[str], *, cube: bool = False,
                   mipmaps: bool = True, name: str = "") -> Image:
        """From a list of image files → a layered image (async, sRGB, mipped by
        default): a texture array, or a cubemap when `cube=True` (6 square faces,
        order +X,-X,+Y,-Y,+Z,-Z). Every face must share a size. `mipmaps=False`
        keeps a single level. Returns immediately like the single-file load; hot
        reload is not wired for layered images in v1 (a re-saved face keeps the
        loaded contents)."""
        ...

    @property
    def uploads_done(self) -> bool:
        """Non-blocking: have all load_image uploads finished?"""
        ...
    @property
    def upload_progress(self) -> float:
        """0.0 .. 1.0 for the current batch of load_image calls (1.0 when
        idle) — a loading bar without user-side threads:

            while not ctx.uploads_done:
                draw_progress(ctx.upload_progress)

        "Batch" means everything queued since the last time uploads fully
        drained: once all in-flight uploads finish, progress resets to 1.0 and
        the next load_image starts a fresh batch from 0. This is the final
        semantics (settled in 0.9) — a second loading screen counts only its own
        images, not the ones a previous screen already finished."""
        ...
    def wait_for_uploads(self) -> None:
        """Block until every pending load_image upload has finished."""
        ...
    def create_image(self, width: int, height: int,
                     format: Format = Format.RGBA8, *, layers: int = 1,
                     cube: bool = False, mip_levels: int = 1, name: str = "") -> Image:
        """Empty image on the GPU. `layers > 1` makes a texture array (view
        2D_ARRAY); `cube=True` makes a cubemap (6 square faces, view CUBE). An
        empty layered image is filled by rendering into it or by a compute
        storage image (procedural skyboxes/arrays); the data forms below upload
        pixels instead.

        `mip_levels > 1` allocates a mip chain (1..full chain for the size); the
        extra levels start empty — write mip 0 (compute / a render pass) then
        `cmd.generate_mipmaps(img)` to fill the rest."""
        ...
    def create_image(self, array: Any, *, mipmaps: bool = False,
                     cube: bool = False, name: str = "") -> Image:
        """From one numpy array → a 2D image; shape + dtype pick the format
        (UNORM — arrays are data, files are pictures). One level by default;
        `mipmaps=True` generates the full chain (arrays stay 1-level unless asked,
        so a data texture gets no surprise filtering). (h, w, 3) has no portable
        GPU format and raises ResourceError with a padding hint. `cube=True` here
        is a mistake — a cubemap needs 6 faces, so pass a list (below)."""
        ...
    def create_image(self, images: Sequence[Any], *, mipmaps: bool = False,
                     cube: bool = False, name: str = "") -> Image:
        """From a list of numpy arrays → a layered image: a texture array, or a
        cubemap when `cube=True` (exactly 6 square faces, order
        +X,-X,+Y,-Y,+Z,-Z). Every layer must share shape and dtype. `mipmaps=True`
        generates the full chain across every layer."""
        ...
    def create_sampler(self, filter: Filter = Filter.LINEAR,
                       address_mode: AddressMode = AddressMode.REPEAT,
                       anisotropy: bool = True,
                       compare: Optional[CompareOp] = None) -> Sampler:
        """Cached: identical descriptions return the identical object.

        `compare=` makes a compare sampler (GLSL `sampler2DShadow`): reads
        return the comparison result instead of the texel, and LINEAR filtering
        becomes hardware PCF. (Linear filtering of depth formats is a format
        feature — universal on desktop GPUs, not spec-guaranteed.)"""
        ...
    def create_descriptor_pool(self, max_sets: int, samplers: int = 0,
                               uniform_buffers: int = 0,
                               storage_buffers: int = 0,
                               storage_images: int = 0) -> DescriptorPool: ...

    def create_command_buffer(self, auto_barriers: Optional[bool] = None) -> CommandBuffer:
        """Command buffers are a device resource, so they come from the Context —
        a headless Context has no renderer to ask.

        auto_barriers overrides the Context-wide mode for this one command
        buffer; None inherits it."""
        ...

    def submit(self, cmd: CommandBuffer) -> None:
        """Execute a command buffer with no swapchain and no present.

        Blocking. This is the headless path.
        """
        ...

class SwapchainRenderer(RenderTargetBase):
    """Presents to a window. One implementation of a render target."""

    def __init__(self, window: Window, context: Context,
                 present_mode: PresentMode = PresentMode.MAILBOX, samples: int = 1) -> None:
        """samples>1 turns on windowed MSAA: rendering goes into a multisampled
        colour+depth image that resolves into the swapchain image on present.
        Must be a power of two <= ctx.max_samples()."""
        ...
    def __init__(self, win32_hwnd: int, context: Context,
                 present_mode: PresentMode = PresentMode.MAILBOX, samples: int = 1) -> None:
        """Attach to an existing native window (Windows only)."""
        ...

    @property
    def present_mode(self) -> PresentMode:
        """The mode actually in use (post-fallback), not the requested one."""
        ...

    def begin_frame(self) -> Optional[Frame]:
        """Acquire the next swapchain image.

        None when the frame should be skipped (minimized, mid-resize):

            frame = renderer.begin_frame()
            if frame:
                frame.submit(cmd)
        """
        ...

    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...

class Frame:
    """One acquired swapchain frame. Submit it and drop it within one tick.

    Submitting twice, or holding a Frame across begin_frame() calls, raises
    ResourceError.
    """

    def submit(self, cmd: CommandBuffer) -> None:
        """Record the command buffer for this frame, submit it and present."""
        ...

    @property
    def frame_index(self) -> int: ...
    @property
    def gpu_time_ms(self) -> Optional[float]:
        """GPU time in milliseconds of the frame submitted frames_in_flight ago
        (a timestamp pair around each submit, read back once its fence signals).
        None unless the Context was created with gpu_timing=True; also None
        until the ring has cycled once, and on devices without timestamp
        support."""
        ...

# ── Keyboard Constants ─────────────────────────────────────────────────

KEY_SPACE: int
KEY_APOSTROPHE: int
KEY_COMMA: int
KEY_MINUS: int
KEY_PERIOD: int
KEY_SLASH: int
KEY_0: int
KEY_1: int
KEY_2: int
KEY_3: int
KEY_4: int
KEY_5: int
KEY_6: int
KEY_7: int
KEY_8: int
KEY_9: int
KEY_SEMICOLON: int
KEY_EQUAL: int
KEY_A: int
KEY_B: int
KEY_C: int
KEY_D: int
KEY_E: int
KEY_F: int
KEY_G: int
KEY_H: int
KEY_I: int
KEY_J: int
KEY_K: int
KEY_L: int
KEY_M: int
KEY_N: int
KEY_O: int
KEY_P: int
KEY_Q: int
KEY_R: int
KEY_S: int
KEY_T: int
KEY_U: int
KEY_V: int
KEY_W: int
KEY_X: int
KEY_Y: int
KEY_Z: int
KEY_LEFT_BRACKET: int
KEY_BACKSLASH: int
KEY_RIGHT_BRACKET: int
KEY_GRAVE_ACCENT: int
KEY_WORLD_1: int
KEY_WORLD_2: int
KEY_ESCAPE: int
KEY_ENTER: int
KEY_TAB: int
KEY_BACKSPACE: int
KEY_INSERT: int
KEY_DELETE: int
KEY_RIGHT: int
KEY_LEFT: int
KEY_DOWN: int
KEY_UP: int
KEY_PAGE_UP: int
KEY_PAGE_DOWN: int
KEY_HOME: int
KEY_END: int
KEY_CAPS_LOCK: int
KEY_SCROLL_LOCK: int
KEY_NUM_LOCK: int
KEY_PRINT_SCREEN: int
KEY_PAUSE: int
KEY_F1: int
KEY_F2: int
KEY_F3: int
KEY_F4: int
KEY_F5: int
KEY_F6: int
KEY_F7: int
KEY_F8: int
KEY_F9: int
KEY_F10: int
KEY_F11: int
KEY_F12: int
KEY_F13: int
KEY_F14: int
KEY_F15: int
KEY_F16: int
KEY_F17: int
KEY_F18: int
KEY_F19: int
KEY_F20: int
KEY_F21: int
KEY_F22: int
KEY_F23: int
KEY_F24: int
KEY_F25: int
KEY_KP_0: int
KEY_KP_1: int
KEY_KP_2: int
KEY_KP_3: int
KEY_KP_4: int
KEY_KP_5: int
KEY_KP_6: int
KEY_KP_7: int
KEY_KP_8: int
KEY_KP_9: int
KEY_KP_DECIMAL: int
KEY_KP_DIVIDE: int
KEY_KP_MULTIPLY: int
KEY_KP_SUBTRACT: int
KEY_KP_ADD: int
KEY_KP_ENTER: int
KEY_KP_EQUAL: int
KEY_LEFT_SHIFT: int
KEY_LEFT_CONTROL: int
KEY_LEFT_ALT: int
KEY_LEFT_SUPER: int
KEY_RIGHT_SHIFT: int
KEY_RIGHT_CONTROL: int
KEY_RIGHT_ALT: int
KEY_RIGHT_SUPER: int
KEY_MENU: int
KEY_LAST: int

# ── Mouse Constants ────────────────────────────────────────────────────

MOUSE_BUTTON_LEFT: int
MOUSE_BUTTON_RIGHT: int
MOUSE_BUTTON_MIDDLE: int

# ── Cursor Mode Constants ──────────────────────────────────────────────

CURSOR_NORMAL: int
CURSOR_DISABLED: int
CURSOR_HIDDEN: int
