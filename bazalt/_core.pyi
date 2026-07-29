from __future__ import annotations

from enum import IntEnum
from typing import Any, Callable, Optional, Sequence, overload

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
    #: A different blend state or colour mask per MRT attachment.
    INDEPENDENT_BLEND = 7
    #: The TESS_CONTROL and TESS_EVALUATION stages, and Topology.PATCH_LIST.
    TESSELLATION = 8
    #: The GEOMETRY stage. Absent on MoltenVK, and slow on modern hardware.
    GEOMETRY_SHADER = 9
    #: A FRAGMENT shader writing a storage buffer or storage image
    #: (fragmentStoresAndAtomics). Needed by the graphics storage_image()
    #: declarator, which shipped in 0.17 without it.
    FRAGMENT_STORES = 10
    #: The same for the pre-rasterization stages — vertex, tessellation, geometry
    #: (vertexPipelineStoresAndAtomics).
    VERTEX_STAGE_STORES = 11

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
    """Which stage a shader is compiled for.

    The tessellation and geometry values are appended rather than placed in
    pipeline order, because the numbers are API. Each needs its device feature:
    compile_shader refuses the stage without it, since SPIR-V for these stages
    declares a capability the driver rejects when the feature is off.
    """
    VERTEX = 0
    FRAGMENT = 1
    COMPUTE = 2
    #: Needs Feature.TESSELLATION. Sets the subdivision levels per patch.
    TESS_CONTROL = 3
    #: Needs Feature.TESSELLATION. Places each vertex the tessellator generated.
    TESS_EVALUATION = 4
    #: Needs Feature.GEOMETRY_SHADER. Per primitive, and may change its type.
    GEOMETRY = 5

class VertexFormat(IntEnum):
    """Vertex attribute layout. Renamed from `Format`, which is reserved for
    pixel formats."""
    FLOAT2 = 0
    FLOAT3 = 1
    FLOAT4 = 2
    FLOAT = 3
    #: Four bytes read as 0..1 floats — vertex colours, skin weights.
    UBYTE4_NORM = 4
    #: An unsigned integer attribute (`in uint`), e.g. a material index.
    UINT = 5

class Topology(IntEnum):
    """Primitive topology for graphics pipelines. TRIANGLE_LIST is the default.

    The strips extend the previous primitive with each new vertex. There is no
    restart index: one strip per draw.
    """
    TRIANGLE_LIST = 0
    POINT_LIST = 1
    LINE_LIST = 2
    TRIANGLE_STRIP = 3
    LINE_STRIP = 4
    #: The input to a tessellation control shader: a run of patch_control_points
    #: vertices with no implied topology. Only valid with tessellation shaders,
    #: and they are only valid with this — the pipeline build checks both ways.
    PATCH_LIST = 5

class BlendMode(IntEnum):
    """How a fragment combines with what the attachment already holds.

    Read only when blend() is enabled. ALPHA is ordinary transparency,
    ADDITIVE is order-independent accumulation (particles, glow), and
    PREMULTIPLIED is for colours that already carry their alpha.
    """
    ALPHA = 0
    ADDITIVE = 1
    PREMULTIPLIED = 2

class PolygonMode(IntEnum):
    """Fill triangles, or draw only their edges (the wireframe view) or
    vertices. All three are core Vulkan and need no Feature."""
    FILL = 0
    LINE = 1
    POINT = 2

class StencilOp(IntEnum):
    """What happens to a stencil value when a fragment arrives, 1:1 with
    VkStencilOp. REPLACE writes the reference value, which is how a mask is
    painted."""
    KEEP = 0
    ZERO = 1
    REPLACE = 2
    INCREMENT_CLAMP = 3
    DECREMENT_CLAMP = 4
    INVERT = 5
    INCREMENT_WRAP = 6
    DECREMENT_WRAP = 7

class Access(IntEnum):
    """What a command does to a buffer — the vocabulary of cmd.barrier()
    in manual mode (auto_barriers=False)."""
    SHADER_READ = 0
    SHADER_WRITE = 1
    VERTEX_READ = 2
    INDEX_READ = 3
    UNIFORM_READ = 4
    #: The draw or dispatch arguments themselves, read by the command processor
    #: rather than by a shader — the hazard behind cmd.draw_indirect (0.19).
    INDIRECT_READ = 5

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
    #: An integer target: an id buffer for picking. No filtering, no conversion.
    R32_UINT = 10
    #: Packed HDR colour in 4 bytes. A cheap bloom / light-accumulation target.
    #: Not readable with Image.read() — three channels share 32 bits.
    R11G11B10F = 11
    #: Depth AND stencil. The exact VkFormat is chosen per device. Attachment
    #: only: it cannot be sampled or read back.
    DEPTH_STENCIL = 12

class Filter(IntEnum):
    LINEAR = 0
    NEAREST = 1

class AddressMode(IntEnum):
    REPEAT = 0
    CLAMP = 1
    MIRROR = 2
    #: Reads the border colour outside 0..1 — the shadow-map fix, where CLAMP
    #: smears the edge texel across everything past the map.
    CLAMP_TO_BORDER = 3

class BorderColor(IntEnum):
    """The texel a CLAMP_TO_BORDER sampler reads outside its range."""
    OPAQUE_BLACK = 0
    OPAQUE_WHITE = 1

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
    preference), IMMEDIATE and FIFO_RELAXED fall back to FIFO with an Info log
    when the surface cannot do them."""
    FIFO = 0
    MAILBOX = 1
    IMMEDIATE = 2
    FIFO_RELAXED = 3

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
    """A snapshot from window.get_mouse_state().

    x/y are the current cursor position. dx/dy and the scroll are the change
    during the last poll_events() cycle, so a camera uses them directly and
    keeps no previous value of its own. Reading twice in one frame gives the
    same answer.
    """
    @property
    def x(self) -> float: ...
    @property
    def y(self) -> float: ...
    @property
    def dx(self) -> float: ...
    @property
    def dy(self) -> float: ...
    @property
    def scroll_dx(self) -> float: ...
    @property
    def scroll_dy(self) -> float: ...

class WindowMode(IntEnum):
    """How the window presents itself.

    FULLSCREEN takes the monitor at its current video mode. It is not
    exclusive fullscreen: the swapchain stays composited, because exclusive
    fullscreen needs VK_EXT_full_screen_exclusive.
    """
    WINDOWED = 0
    FRAMELESS = 1
    FULLSCREEN = 2
    FULLSCREEN_WINDOWED = 3

# ── Resources ──────────────────────────────────────────────────────────

class Buffer:
    def update(self, data: bytes, *, offset: int = 0) -> None: ...
    def update(self, array: Any, *, offset: int = 0) -> None:
        """Upload from any C-contiguous buffer-protocol object.

        offset is a BYTE offset into the buffer. Without it, changing one matrix
        of an instance array rewrites the whole array, and update() is already
        the per-frame path, so the wasted copy is per frame too.

        Raises ResourceError for a strided view (`arr.T`, `arr[::2]`): copying
        silently would hide an allocation on every upload. Pass
        `numpy.ascontiguousarray(arr)` to be explicit.
        """
        ...
    def update(self, list: list, data_type: Optional[DataType] = None, *,
               offset: int = 0) -> None: ...

    def read(self, dtype: Any) -> Any:
        """Copy the buffer back to host memory as a 1-D numpy array.

        dtype is mandatory — buffers carry no format, so the caller says how
        to interpret the bytes (e.g. `ssbo.read(np.float32)`). STATIC buffers
        take a blocking GPU round trip; DYNAMIC ones return what update()
        last wrote into the current frame's copy.

        A pending upload is waited for first, so this is correct on the line
        after create_buffer.
        """
        ...

    @property
    def ready(self) -> bool:
        """Non-blocking: is the data on the GPU?

        False while a STATIC buffer's staging copy is still running. You never
        have to poll this — a submit that binds the buffer waits automatically,
        and read() waits too; it exists for loading screens and explicit
        control. Always True for a DYNAMIC buffer, which is written by mapping
        and has no copy to wait for.
        """
        ...

    def wait(self) -> None:
        """Block until this buffer's upload has finished."""
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
    @property
    def writes(self) -> list[tuple[int, int]]:
        """The (set, binding) pairs this shader WRITES, from SPIR-V reflection
        (0.19). Sorted, and empty when the shader provably writes nothing.

        This is what lets bazalt insert automatic barriers for a storage buffer or
        storage image written by a graphics shader, which used to need a manual
        cmd.barrier(). It counts stores, image writes and atomics — a buffer touched
        only by atomicAdd is written.

        Exposed mainly so the reflection has a referee: it is also the answer to
        "why is there no barrier here". Reads are not reported; bazalt takes the
        bindings a shader may read from the pipeline builder's declarators.
        """
        ...
    @property
    def writes_unknown(self) -> bool:
        """True when the write scan could not follow something, so every binding is
        assumed written and no barrier is narrowed (0.19).

        Set for ready SPIR-V (a .spv file or source=bytes), because bazalt only
        knows the write opcodes its own GLSL and HLSL compile down to. Also set by a
        descriptor passed into a function or appearing in a pointer phi. The rule is
        one-directional: the tracker may be pessimistic, never optimistic.
        """
        ...
    @property
    def prints(self) -> bool:
        """True when the shader calls debugPrintfEXT (0.19).

        A printf Context compiles only these shaders unoptimized — a print is a
        non-semantic instruction the optimizer may delete, so it cannot survive -O.
        Every other shader in that Context is optimized normally.
        """
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

        False while any upload is still in flight — a load_image decode/copy,
        a create_image(array) copy or an update(). You never have to poll this
        — a submit that uses the image waits automatically; it exists for
        loading screens and explicit control.
        """
        ...

    def wait(self) -> None:
        """Block until this image's upload has finished.

        A failed decode (corrupt file) surfaces here as ResourceError.
        """
        ...

    def read(self, *, layer: int = 0, mip: int = 0) -> Any:
        """Copy one subresource back to host memory as a numpy array.

        Shape is (height, width, channels) — or (height, width) for
        single-channel formats — and the dtype follows the format (uint8,
        float16 or float32). Blocking; a debugging and test path.

        layer picks a cube face or array slice, mip picks a level, and the shape
        follows the MIP: level 2 of a 64x64 texture reads back 16x16. Before
        0.18 there was no choice — read() meant mip 0 of layer 0, so a cube face
        could not be inspected and "did generate_mipmaps compute anything" was a
        question with no way to ask it.

        Raises ResourceError if the image has no contents yet, or if the layer
        or mip does not exist.
        """
        ...

    def update(self, array: Any, *, layer: int = 0, mip: int = 0,
               region: Optional[Sequence[int]] = None) -> None:
        """Change the pixels of an image that already exists.

        A video frame, a camera feed, a matplotlib figure, a procedural texture
        computed in numpy, painting — before 0.18 every one of them meant
        creating a new Image per frame, because there was no way to write into
        one.

        The array's dtype and shape must match the image's format and the region
        being written, and it must be C-contiguous: memcpy ignores strides, so a
        transposed or sliced array would upload garbage rather than raise. Use
        numpy.ascontiguousarray() if needed.

        region=(x, y, width, height) writes a rectangle and leaves the rest
        alone — what painting and a sprite atlas need. Omitted, the whole level
        is replaced.

        ASYNCHRONOUS, like load_image: it returns at once and the copy runs on
        the upload worker, because the case this exists for is a video frame at
        60 fps. Use img.wait(), img.ready or ctx.wait() to know when
        it has landed; a submit that samples the image waits for it GPU-side
        either way. Two updates of one image reach the GPU in the order they
        were called — one FIFO worker, and that is a guarantee.
        """
        ...

class Sampler:
    """How to read texels. Cached on the Context: identical descriptions are
    the identical object."""

    @property
    def name(self) -> str:
        """The debug name, which is every name this shared sampler was given,
        joined with " + ". Empty when nobody named it."""
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
    """An offscreen target backed by Images. No window required.

    The attachments are ordinary Images: `target.color[0]` and `target.depth`
    go straight into DescriptorSet.set_image — that is the whole
    render-to-texture and shadow-map API.

    Two ways to build one, and they do different jobs. Pass a width and height and
    the target allocates its attachments from pixel formats. Pass images from
    create_image and it renders into those instead — that signature has no size,
    samples, layers, cube or mip_levels, because the images already answer all of
    them (0.19).
    """

    @overload
    def __init__(self, context: Context, *,
                 color: Optional[Image | Sequence[Image]] = None,
                 depth: Optional[Image] = None, name: str = "") -> None:
        """Render into images you already own, rather than attachments the target
        allocates.

        What this makes reachable: a graphics ping-pong between two textures,
        drawing over a texture a compute pass baked, and drawing into an image
        carried from another Context. All three were impossible while a target
        insisted on owning its attachments.

        Every attachment must be the same size with the same layer and mip count;
        a mismatch is refused rather than intersected. Single-sample only, because
        create_image has no samples=. The target holds the images, so dropping your
        reference does not take the attachment with it — and it does write to their
        layout tracking, which is what leaves the result sampleable.
        """
        ...

    @overload
    def __init__(self, context: Context, width: int, height: int,
                 color: Optional[Format | Sequence[Format]] = Format.RGBA8,
                 depth: Optional[Format] = None,
                 samples: int = 1, *, layers: int = 1, cube: bool = False,
                 mip_levels: int = 1, name: str = "") -> None:
        """color=None with depth=D32F makes a depth-only (shadow) target;
        a list of formats makes an MRT target. At least one attachment is
        required.

        samples>1 turns on MSAA: the target renders into a multisampled image and
        resolves into target.color/target.depth (which stay single-sample and
        sampleable — depth resolves too, via SAMPLE_ZERO). Must be a power of two
        <= ctx.max_samples(). name labels the attachments in validation messages.

        layers>1 / cube=True / mip_levels>1 make the attachments layered / cube /
        mipped so a scene can be rasterized into one subresource with
        .layer(i, mip=m) (render-to-layer / render-to-mip: dynamic env capture,
        cascade shadows). cube fixes 6 square layers and gives the colour attachment a
        CUBE view so target.color[0] samples as a cubemap. Single-sample only:
        samples>1 cannot combine with layers/cube/mip_levels this release.
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

    def layer(self, index: int, mip: int = 0) -> RenderTargetBase:
        """A view of one array layer / cube face (and optionally one mip) of this
        target, to render into with cmd.rendering(target.layer(i)). `mip=` selects
        a level for a layered AND mipped target (e.g. a mipped cube for prefiltered
        reflections). Cube face i == layer i, Vulkan order +X, -X, +Y, -Y, +Z, -Z.
        Render every layer you intend to sample before sampling target.color /
        target.depth."""
        ...

    def all_layers(self) -> RenderTargetBase:
        """A multiview view of the whole target: cmd.rendering(target.all_layers())
        renders into EVERY layer in ONE pass instead of a pass per layer. The
        shader selects per-layer work with gl_ViewIndex (e.g. a per-face matrix
        for cube capture). Needs a layered target and ctx.supports_multiview();
        composes with MSAA (each view resolves into its own layer). Renders every
        layer, so the result is fully sampleable with no partial-render caveat."""
        ...

class GraphicsPipelineBuilder:
    def vertex_shader(self, shader: ShaderModule) -> GraphicsPipelineBuilder: ...
    def fragment_shader(self, shader: ShaderModule) -> GraphicsPipelineBuilder: ...
    def tess_control_shader(self, shader: ShaderModule) -> GraphicsPipelineBuilder:
        """The tessellation control stage: how finely to subdivide each patch.

        Set together with tess_evaluation_shader — one without the other is not a
        partial pipeline but an invalid one, and build() says so. A tessellation
        pipeline also needs topology(PATCH_LIST) and patch_control_points(n).
        """
    def tess_evaluation_shader(self, shader: ShaderModule) -> GraphicsPipelineBuilder:
        """The tessellation evaluation stage: where each generated vertex goes.

        This is where displacement happens. See tess_control_shader.
        """
    def geometry_shader(self, shader: ShaderModule) -> GraphicsPipelineBuilder:
        """The geometry stage: one invocation per primitive, and it may emit a
        DIFFERENT primitive type — triangles to lines, a point to a quad.

        Needs Feature.GEOMETRY_SHADER. It is slow on modern hardware and absent on
        MoltenVK, so prefer it for debug views over hot paths. Routing one draw
        into every layer of an array attachment does NOT need it: that is
        target.all_layers().
        """
    def patch_control_points(self, count: int) -> GraphicsPipelineBuilder:
        """How many vertices of the vertex buffer make one patch — 3 for a
        triangle patch, 4 for a quad.

        This is the INPUT patch size, which is why it lives on the pipeline: the
        control shader's own `layout(vertices = N) out` is its OUTPUT count, a
        different number that neither one can derive from the other. Required
        with tessellation, and checked against the device's limit.
        """
    def vertex_format(self, formats: list[VertexFormat]) -> GraphicsPipelineBuilder: ...
    def instance_format(self, formats: list[VertexFormat]) -> GraphicsPipelineBuilder:
        """The attributes of a second vertex buffer, advanced once per instance.

        Feed it with cmd.bind_vertex_buffer(buf, binding=1) and draw with
        draw(n, instances=k). Locations continue after the vertex attributes: a
        vertex_format of 3 puts the first instance attribute at location 3, and
        a mat4 is four FLOAT4s taking four locations.
        """
        ...
    def depth_test(self, enable: bool, write: bool = True,
                   compare: CompareOp = CompareOp.LESS_OR_EQUAL) -> GraphicsPipelineBuilder:
        """Depth test, depth write and the compare op.

        write=False keeps the test and stops the depth update, which is what a
        transparency pass needs. Nothing is written when enable is False,
        whatever write says.
        """
        ...
    def cull_mode(self, mode: CullMode, front_face: FrontFace) -> GraphicsPipelineBuilder: ...
    def polygon_mode(self, mode: PolygonMode) -> GraphicsPipelineBuilder: ...
    def line_width(self, width: float) -> GraphicsPipelineBuilder:
        """Line width in pixels for PolygonMode.LINE or Topology.LINE_LIST.

        Anything other than 1.0 needs Feature.WIDE_LINES — build() raises
        ShaderError without it, because a driver may support exactly one width.
        """
        ...
    def depth_bias(self, constant: float, slope: float = 0.0) -> GraphicsPipelineBuilder:
        """Offset the depth this pipeline writes. The fix for shadow acne.

        `slope` scales with the polygon's depth gradient, which is what makes
        one setting hold at every angle. depth_bias(0) is the same pipeline as
        no call at all.
        """
        ...
    def blend(self, enable: bool, mode: BlendMode = BlendMode.ALPHA,
              attachment: Optional[int] = None) -> GraphicsPipelineBuilder:
        """How a fragment combines with the attachment it lands on.

        attachment= narrows the setting to one colour attachment of an MRT
        target; None (the default) sets it for all of them. Per-attachment
        blend() and color_mask() calls merge, in either order.
        """
        ...
    def color_mask(self, red: bool = True, green: bool = True, blue: bool = True,
                   alpha: bool = True, attachment: Optional[int] = None) -> GraphicsPipelineBuilder:
        """Which channels the pipeline writes. All four False writes no colour."""
        ...
    def stencil_test(self, enable: bool, compare: CompareOp = CompareOp.ALWAYS, ref: int = 0,
                     pass_op: StencilOp = StencilOp.KEEP, fail_op: StencilOp = StencilOp.KEEP,
                     depth_fail_op: StencilOp = StencilOp.KEEP, read_mask: int = 0xFF,
                     write_mask: int = 0xFF) -> GraphicsPipelineBuilder:
        """The stencil test. Needs a target with depth=Format.DEPTH_STENCIL.

        An outline is two passes: the first writes the silhouette with
        compare=ALWAYS, ref=1, pass_op=REPLACE; the second draws a scaled copy
        with compare=NOT_EQUAL, ref=1 and depth_test(False), so only the pixels
        around the object survive.

        Front and back faces share the state.
        """
        ...
    def depth_clamp(self, enable: bool = True) -> GraphicsPipelineBuilder:
        """Clamp depth to the view volume instead of clipping the primitive —
        a shadow caster between the light and the near plane still casts.
        Needs Feature.DEPTH_CLAMP."""
        ...
    def alpha_to_coverage(self, enable: bool = True) -> GraphicsPipelineBuilder:
        """Turn a fragment's alpha into an MSAA coverage mask: antialiased
        cutout foliage and hair with no sorting. No effect on a target with
        samples=1."""
        ...
    def constant(self, id: int, value: bool | int | float,
                 stage: ShaderStage) -> GraphicsPipelineBuilder:
        """A specialization constant, baked into the SPIR-V at build time.

        The driver folds it, so a constant loop count unrolls and a constant
        False deletes its branch. One compiled shader can therefore serve
        several pipelines that differ by a number — quality levels, a kernel
        radius. `id` is the constant_id in the shader. The stage is needed
        because the two stages are separate modules.
        """
        ...
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
    def storage_image(self, binding: int, stage: ShaderStage, set: int) -> GraphicsPipelineBuilder:
        """A read/write image addressed by coordinate (imageLoad/imageStore) in a
        graphics shader.

        The automatic tracker does NOT see writes through it — that needs shader
        reflection — so a storage image written by a fragment shader and read
        later needs cmd.barrier(image, Access.SHADER_WRITE, Access.SHADER_READ),
        exactly like an SSBO written from graphics.
        """
        ...
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
    def constant(self, id: int, value: bool | int | float) -> ComputePipelineBuilder:
        """A specialization constant. No stage argument — compute has one stage.

        The classic use is the workgroup size: declare it with
        `layout(local_size_x_id = 0)` and pick the number when the pipeline is
        built.
        """
        ...
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
                        clear_color: Sequence[float] | Sequence[Sequence[float]] | None = (0.0, 0.0, 0.0, 1.0),
                        clear_depth: float = 1.0, clear_stencil: int = 0) -> CommandBuffer:
        """Start rendering into `target`.

        clear_color is either a single [r, g, b, a] applied to every attachment
        (the common case) or, for MRT, a list of them ([[r,g,b,a], …]) clearing
        each attachment independently.

        None preserves the colour and the depth instead of clearing them, which
        is how a second pass draws over the first one on the same target. The
        first pass of a frame must still clear, because an acquired swapchain
        image starts with undefined contents. Raises ResourceError on a
        multisampled target, which has nothing to preserve.

        clear_depth is the depth value, not a second preserve switch: 1.0 is the
        far plane, and 0.0 is where a reversed-depth buffer starts (the only way
        depth_test(compare=GREATER) can pass). Ignored when the pass preserves.

        clear_stencil is the same for the stencil half of a Format.DEPTH_STENCIL
        attachment, and does nothing without one. Depth and stencil load
        together: a pass cannot preserve one and clear the other.

        Also emits a viewport and scissor covering the whole target, so the
        common case needs no further calls.
        """
        ...

    def end_rendering(self, target: RenderTargetBase) -> CommandBuffer: ...

    def rendering(self, target: RenderTargetBase,
                  clear_color: Sequence[float] | Sequence[Sequence[float]] | None = (0.0, 0.0, 0.0, 1.0),
                  clear_depth: float = 1.0, clear_stencil: int = 0) -> RenderingScope:
        """The begin/end pair as a context manager:

            with cmd.rendering(target, clear_color=[0, 0, 0, 1]) as c:
                c.bind_pipeline(p).draw(3)

        end_rendering is recorded on exit, exceptions included. clear_color takes
        the same single-or-per-attachment forms as begin_rendering, and None
        preserves the attachment for a second pass.
        """
        ...

    def set_viewport(self, x: float, y: float, width: float, height: float) -> CommandBuffer:
        """Override the automatic full-target viewport (split-screen and similar)."""
        ...

    def set_scissor(self, x: int, y: int, width: int, height: int) -> CommandBuffer: ...

    def bind_pipeline(self, pipeline: Pipeline) -> CommandBuffer: ...
    def bind_vertex_buffer(self, buffer: Buffer, binding: int = 0) -> CommandBuffer:
        """Bind a vertex buffer. binding=0 feeds vertex_format (per vertex),
        binding=1 feeds instance_format (per instance)."""
        ...
    def bind_index_buffer(self, buffer: Buffer) -> CommandBuffer: ...
    def draw(self, vertex_count: int, instances: int = 1) -> CommandBuffer: ...
    def draw_indexed(self, index_count: int, first_index: int = 0,
                     vertex_offset: int = 0, instances: int = 1) -> CommandBuffer:
        """instances= replaces draw_indexed_instanced, which was a second name
        for one extra argument."""
        ...
    def dispatch(self, group_count_x: int, group_count_y: int = 1,
                 group_count_z: int = 1) -> CommandBuffer: ...

    def draw_indirect(self, buffer: Buffer, offset: int = 0,
                      count: int = 1) -> CommandBuffer:
        """Draw with arguments read out of a buffer, so a compute pass decides what
        gets drawn and the CPU never learns the answer (0.19).

        `buffer` must be BufferType.STORAGE — the only type carrying the indirect
        usage flag, and what a compute shader needs anyway. bazalt declares no
        struct type: the layout is VkDrawIndirectCommand, four uint32s, and numpy
        writes it directly.

            args = ctx.create_buffer(
                np.array([vertex_count, instances, 0, 0], dtype=np.uint32),
                bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
            cmd.draw_indirect(args)

        A std430 GLSL struct of four uints is byte-identical, so a compute shader
        can zero it with cmd.fill_buffer and accumulate instanceCount atomically.
        count>1 reads that many consecutive structs and needs
        Feature.MULTI_DRAW_INDIRECT. To draw nothing, write 0 into instanceCount —
        count=0 is refused, because only one of the two can be decided on the GPU.
        """
        ...
    def draw_indexed_indirect(self, buffer: Buffer, offset: int = 0,
                              count: int = 1) -> CommandBuffer:
        """draw_indirect through the bound index buffer (0.19).

        The struct is VkDrawIndexedIndirectCommand, five words: indexCount,
        instanceCount, firstIndex, vertexOffset (SIGNED int32), firstInstance.
        """
        ...
    def dispatch_indirect(self, buffer: Buffer, offset: int = 0) -> CommandBuffer:
        """Dispatch with the group counts read out of a buffer (0.19).

        The struct is VkDispatchIndirectCommand: three uint32s, x/y/z. There is no
        count — the command takes exactly one. Unlike the draw verbs this needs no
        feature bit.
        """
        ...

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

    def copy_image(self, src: Image, dst: Image, *,
                   src_access: Access = Access.SHADER_READ) -> CommandBuffer:
        """Copy one image into another of the same size, format and layer count.

        The history buffer a temporal effect needs: keep the last frame's result
        to blend against this one (motion blur, a feedback trail), or ping-pong
        two storage images.

        Every mip level the two images share is copied. Until 0.18 this was mip
        0 only, which left levels 1..N holding the destination's old pixels —
        a copy of the image's top level rather than of the image, and the
        difference showed up the moment anything sampled with a mip bias.

        src_access names where the SOURCE currently is: SHADER_READ for an image
        that is sampled (the default), SHADER_WRITE for one a compute dispatch
        just wrote. Both images end sampleable. Refused inside a rendering scope.
        """
        ...

    def blit_image(self, src: Image, dst: Image, *,
                   src_access: Access = Access.SHADER_READ,
                   filter: Filter = Filter.LINEAR) -> CommandBuffer:
        """Copy one image into another of a DIFFERENT size, scaling on the way.

        copy_image needs the two to match; generate_mipmaps scales, but only
        inside one image. Downsampling for bloom, upscaling a compute result and
        making a thumbnail all sat in that gap, and each one used to be a full
        graphics pass with a fullscreen shader.

        Mip 0 of every shared layer. Call generate_mipmaps on the destination if
        it needs a chain.

        Raises ResourceError when this GPU cannot blit between the two formats —
        BLIT_SRC/BLIT_DST are format features, not a given, and a linear filter
        needs the source to be filterable on top.

        src_access, and both images ending sampleable, work exactly as for
        copy_image. Refused inside a rendering scope.
        """
        ...

    def copy_buffer(self, src: Buffer, dst: Buffer, *,
                    src_offset: int = 0, dst_offset: int = 0,
                    size: int = 0) -> CommandBuffer:
        """Copy bytes from one buffer into another, GPU-side.

        size=0 means the rest of the source. A compute ping-pong and "keep last
        frame's values" are both this one command; before it, moving buffer
        contents meant a round trip through the host or a compute shader written
        to do nothing but assign.

        Refused inside a rendering scope.
        """
        ...

    def fill_buffer(self, buffer: Buffer, value: int = 0, *,
                    offset: int = 0, size: int = 0) -> CommandBuffer:
        """Fill a buffer with a repeated 32-bit value, GPU-side.

        Zeroing is the reason it exists: a counter an atomic increments has to
        start each frame at a known value, and saying so used to take a dispatch
        whose whole body was an assignment.

        offset and size must be multiples of 4, and size=0 means the rest of the
        buffer. 32-bit because the underlying command is: the value is one word
        repeated. Refused inside a rendering scope.
        """
        ...

    def clear_image(self, image: Image,
                    color: Sequence[float] = (0.0, 0.0, 0.0, 1.0)) -> CommandBuffer:
        """Fill a colour image with one value, with no pipeline and no pass.

        Resets an accumulation or history buffer. A depth image is refused: its
        clear belongs to the pass that renders into it (clear_depth=).
        """
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
        overlapping timers all work. Unlike renderer.gpu_time_ms this needs no
        window: t.ms is ready as soon as the submit you timed has finished. Self-gating: the query pool exists only once a timer is
        used, so apps that don't time pay nothing."""
        ...

    def label(self, name: str) -> LabelScope:
        """Name a scope so a capture reads as a frame instead of a list of draws:

            with cmd.label("shadow pass"):
                with cmd.rendering(shadow_target):
                    ...

        Labels nest. A no-op without VK_EXT_debug_utils — which bazalt requests
        only when validation is on — so a release run pays nothing and simply
        shows no labels. Object names (name= on create_buffer / create_image /
        the pipeline builders) answer "which object"; this answers "which pass"."""
        ...

    def begin_label(self, name: str) -> CommandBuffer:
        """The explicit half of label(), for a recording split across functions
        — a helper that opens the scope and a matching one that closes it, where
        no `with` block spans both. Same escape hatch as begin_rendering /
        end_rendering. Prefer `with cmd.label(...)` whenever one block covers
        the scope: it cannot leave a label open."""
        ...

    def end_label(self) -> CommandBuffer:
        """Close the innermost open label. An unbalanced call is ignored rather
        than recorded: ending a label that was never begun is undefined
        behaviour in Vulkan."""
        ...

    def occlusion_query(self) -> OcclusionQuery:
        """Count the fragments of the draws inside the scope that passed the
        depth and stencil tests:

            with cmd.rendering(target):
                with cmd.occlusion_query() as q:
                    cmd.bind_pipeline(bounding_box).draw(36)
            ctx.submit(cmd)
            visible = q.samples > 0

        The handle is the identity, exactly as for timer(). Must sit inside a
        rendering scope — Vulkan requires the query to begin and end within one
        render pass — and raises ResourceError otherwise.

        The count is not requested as precise, because precision needs the
        occlusionQueryPrecise feature and without it the spec allows any non-zero
        value. Treat `samples` as "how much, roughly", and `samples > 0` as the
        reliable part."""
        ...

class LabelScope:
    """Returned by CommandBuffer.label(); use it in a `with` statement."""

    def __enter__(self) -> CommandBuffer: ...
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool: ...

class OcclusionQuery:
    """An occlusion query handle from CommandBuffer.occlusion_query(). Usable as
    a context manager or stopped by hand (q.stop())."""

    def stop(self) -> None:
        """End the query. Idempotent; called for you on `with` exit."""
        ...
    @property
    def samples(self) -> Optional[int]:
        """Fragments that passed the depth and stencil tests, or None if the
        command buffer was re-recorded since (the handle is stale) or the submit
        has not completed. Read it after the submit you measured has finished —
        the default submit(wait=True) is enough; submit(wait=False) needs a
        ctx.wait() first."""
        ...
    def __enter__(self) -> OcclusionQuery: ...
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool: ...

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
        stale), or the submit has not completed. Read it after the submit you
        timed has finished — the default submit(wait=True) is enough;
        submit(wait=False) needs a ctx.wait() first."""
        ...
    def __enter__(self) -> Timer: ...
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool: ...

class Window:
    def __init__(self, width: int, height: int, title: str,
                 logger: Optional[Logger] = None,
                 mode: WindowMode = WindowMode.WINDOWED) -> None:
        """width/height/mode describe the window it opens with.

        Whatever mode it opens in, that width and height are what WINDOWED
        returns to later.
        """
        ...
    def is_open(self) -> bool: ...
    def is_key_pressed(self, key: int) -> bool: ...
    def is_mouse_button_pressed(self, button: int) -> bool: ...
    def was_key_pressed(self, key: int) -> bool:
        """True when the key went down during the last poll_events() cycle.

        The edge, where is_key_pressed is the level. Use it for a toggle, and
        read it as often as you like within one frame. Key auto-repeat does not
        count as an edge.
        """
        ...
    def was_mouse_button_pressed(self, button: int) -> bool: ...
    def dropped_files(self) -> list[str]:
        """Paths dropped onto the window during the last poll cycle (0.19).

        Empty on almost every frame. A drop is a change that expires with the
        cycle, like a key edge or a scroll notch, so this reads the same twice
        inside one frame and does not consume: two readers both see the drop.
        """
        ...
    def set_cursor_mode(self, mode: int) -> None: ...
    def set_cursor_position(self, x: float, y: float) -> None:
        """Move the cursor, without the move reading as the user moving it (0.19).

        get_mouse_state().dx/dy stay at rest across the jump, so a warp never
        injects a delta the size of itself. That is what makes the hidden-cursor
        recentring pattern work: warp to the centre, then read the delta from it.

        Do NOT combine it with CURSOR_DISABLED and a per-frame warp. That mode
        already hands out unbounded virtual motion and recentres itself, so warping
        every frame cancels every frame's delta and the camera stops turning. Pick
        one: disabled and no warp, or hidden and warp.
        """
        ...
    def set_icon(self, icon: Optional[np.ndarray]) -> None:
        """The task-bar and title-bar icon, as a (height, width, 4) uint8 RGBA
        array. None restores the system default (0.19).

        The array must be C-contiguous — a strided view would copy other bytes.
        The platform may ignore the request: macOS takes the icon from the app
        bundle and Wayland from the desktop file. Same contract as set_opacity.
        """
        ...
    def get_mouse_state(self) -> MouseState: ...
    def set_title(self, title: str) -> None: ...
    def set_mode(self, mode: WindowMode) -> None:
        """Switch between windowed, frameless and the two fullscreen modes.

        WINDOWED restores the position and size the window had before it left
        that mode. Fullscreen takes the monitor the window covers most of. The
        swapchain follows on its own, through the same path a resize takes.
        Raises WindowError when no monitor reports a video mode.
        """
        ...
    def set_size(self, width: int, height: int) -> None: ...
    def set_position(self, x: int, y: int) -> None: ...
    def set_resizable(self, enable: bool) -> None: ...
    def set_always_on_top(self, enable: bool) -> None: ...
    def set_opacity(self, opacity: float) -> None:
        """0.0 is invisible and 1.0 is opaque. Ignored where the platform has
        no compositor."""
        ...
    @property
    def mode(self) -> WindowMode: ...
    @property
    def position(self) -> tuple[int, int]: ...
    @property
    def resizable(self) -> bool: ...
    @property
    def always_on_top(self) -> bool: ...
    @property
    def opacity(self) -> float: ...
    @property
    def content_scale(self) -> tuple[float, float]:
        """Framebuffer pixels per screen coordinate — 2.0 on a HiDPI display.

        This is why width/height (screen coordinates) and the swapchain extent
        (pixels) can disagree.
        """
        ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...

class MemoryStats:
    """GPU memory as VMA sees it, from Context.memory_stats(). Bytes, not
    megabytes: a rounded number cannot be un-rounded."""

    @property
    def used(self) -> int:
        """Bytes VMA has allocated for this Context's resources."""
        ...
    @property
    def reserved(self) -> int:
        """Bytes VMA has reserved from the driver. Always >= used: VMA
        sub-allocates out of larger blocks, so a freed resource does not
        immediately give memory back to the driver."""
        ...
    @property
    def budget(self) -> int:
        """Bytes the driver says this process may use. Not the same as the card's
        capacity — it accounts for whatever else is running."""
        ...

class Device:
    """One GPU this machine can offer, as inert data — no live Vulkan handle.

    Returned by list_devices(); pass one to Context(device=...). Safe to keep
    around: it holds copies, not handles, so it stays valid with no Context at
    all and across Contexts created later.
    """

    @property
    def name(self) -> str: ...
    @property
    def type(self) -> str:
        """"discrete", "integrated", "virtual", "cpu" or "other"."""
        ...
    @property
    def api_version(self) -> str: ...
    @property
    def memory_mb(self) -> int:
        """Total device-local memory. On an integrated GPU this is shared
        system memory."""
        ...

    def supports(self, feature: Feature) -> bool:
        """The same question as ctx.supports(), asked before there is a
        Context — so you can pick the card that can do the job."""
        ...
    def supports_multiview(self) -> bool: ...

def poll_events() -> None:
    """Drain the OS event queue and dispatch each event to the window the OS
    addressed it to.

    A free function, not a Window method, because there is no such thing as
    polling one window: the OS message queue is per-thread and glfwPollEvents
    takes no window. `window_a.poll_events()` would read as A's events while
    pumping everyone's.

    The per-window distinction is real, it just lives in the queries — each
    reads one window's own state, which this dispatch is what updates:

        bz.poll_events()
        if window_a.is_key_pressed(bz.KEY_W):   # only while A has focus
            ...
        if renderer_b.acquire():                # False while B is minimized
            renderer_b.present(cmd_b)

    Raises WindowError when no window exists — with none open there is no queue
    to drain, and a loop still pumping is a bug rather than a no-op.
    """
    ...

def get_clipboard() -> str:
    """The system clipboard as text (0.19).

    Empty when the clipboard holds nothing, or holds something that is not text
    (an image, a file list) — "nothing to paste" is an answer, not an error.

    A free function for the same reason poll_events is one: the GLFW calls take
    no window and the clipboard belongs to the process, so a method would invent
    a per-window distinction that does not exist. Raises WindowError with no live
    Window, because GLFW is initialized with the first one.
    """
    ...

def set_clipboard(text: str) -> None:
    """Put text on the system clipboard. See get_clipboard (0.19)."""
    ...

def list_devices() -> list[Device]:
    """Every GPU on this machine, without creating a Context.

        for d in bz.list_devices():
            print(d.name, d.type, d.memory_mb)

        ctx = bz.Context(device=bz.list_devices()[1])

    Creates and destroys a throwaway Vulkan instance (Vulkan has no way to
    enumerate GPUs without one), so it costs a few milliseconds; the Devices it
    returns outlive it.
    """
    ...

class Context:
    """The GPU device, and the factory for everything that lives on it.

    Any number of Contexts may be alive at once, on the same GPU or on
    different ones — pick with `device=` (see list_devices()). Each owns its
    own device, frame ring and upload worker; nothing is shared between them.

    Resources belong to the Context that made them. Handing one to another
    Context's command buffer or descriptor set raises ResourceError rather than
    reaching the driver. To move pixels across, pass the image to the other
    Context's create_image().
    """

    def __init__(self, logger: Optional[Logger] = None, validation: str = "auto",
                 features: Sequence[Feature] = (), optional: Sequence[Feature] = (),
                 frames_in_flight: int = 2,
                 device: Optional[Device] = None,
                 raw_extensions: Sequence[str] = (),
                 auto_barriers: bool = True,
                 hot_reload: bool = False,
                 gpu_timing: bool = False,
                 shader_printf: bool = False) -> None:
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
            device: which GPU to run on, from list_devices(). None picks
                automatically (prefer discrete, must satisfy `features`) — the
                choice is still filtered by `features` either way, so a device
                that cannot do the job raises rather than misrendering.
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
            gpu_timing: record renderer.gpu_time_ms (a timestamp pair around each
                windowed submit). Off by default because it is a profiling
                diagnostic — the pool reset and two writes ride in every frame's
                command buffer, and per-frame queries are not guaranteed free on
                every GPU. Left off, renderer.gpu_time_ms is always None, no cost.
            shader_printf: deliver debugPrintfEXT() output from your shaders to
                the logger, as Severity.INFO from Source.SHADER. Write
                `#extension GL_EXT_debug_printf : enable` in the shader and call
                `debugPrintfEXT("value = %f", x)`. The prints arrive whatever the
                logger's min_severity is.

                The validation layers implement this, so it needs them: with
                validation="off" the constructor raises InitializationError, and
                with validation="auto" on a machine that has no layers installed
                you get a warning and no prints.

                Two costs, which is why it is off by default. The layer
                instruments every shader, and bazalt compiles every shader in
                this Context without optimization, because a print is a
                non-semantic instruction that the optimizer is entitled to
                delete.
        """
        ...

    @property
    def logger(self) -> Logger: ...
    @property
    def frames_in_flight(self) -> int: ...
    @property
    def frame_index(self) -> int:
        """Which ring slot the current frame writes into (0..frames_in_flight-1)."""
        ...
    @property
    def auto_barriers(self) -> bool: ...
    @property
    def shader_printf(self) -> bool: ...
    @property
    def subgroup_size(self) -> int:
        """The width this GPU runs shader subgroups at, or 0 where the driver
        reports none. A compute reduction using subgroupAdd sizes its workgroup
        against this."""
        ...

    def memory_stats(self) -> MemoryStats:
        """How much GPU memory this Context holds, and how much room is left.

        VMA already keeps the numbers, so this is a read, not new bookkeeping.
        Summed across heaps — which heap an allocation lands in is a driver
        decision, and "am I growing" is the question this answers."""
        ...

    def begin_frame(self) -> None:
        """Open one logical frame — the frame verb of a windowed loop.

        Advances the ring slot that CommandBuffer, DynamicBuffer and the
        per-frame descriptor sets index, applies pending hot reloads, and
        reclaims deferred handles. All of that is Context-owned, so it happens
        once per frame no matter how many windows draw into it:

            while window.is_open():
                bz.poll_events()
                ctx.begin_frame()
                if renderer.acquire():
                    renderer.present(cmd)

        The headless path needs no counterpart: ctx.submit() advances the ring
        itself.
        """
        ...
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
    def supports_multiview(self) -> bool:
        """Whether this GPU supports multiview — one-pass render into every layer
        of an array/cube target via RenderTarget.all_layers()."""
        ...
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
                      usage: MemoryUsage, *, name: str = "") -> Buffer:
        """A GPU buffer from a list, any C-contiguous array, or a size in bytes.

        A STATIC buffer is device-local and filled by a staging copy. That copy
        is ASYNCHRONOUS since 0.18.0: it is submitted here — so a failure still
        raises here — but not waited for, because 30 meshes used to mean 30 full
        queue drains at startup. The buffer is usable at once; a submit that
        binds it waits GPU-side, and read() waits CPU-side. `buf.ready`,
        `buf.wait()` and `ctx.wait()` are the explicit-control verbs.

        A DYNAMIC buffer is host-visible and written by update(), so it has no
        copy and is never pending.
        """
        ...

    def graphics_pipeline(self) -> GraphicsPipelineBuilder: ...
    def compute_pipeline(self) -> ComputePipelineBuilder: ...
    def compile_shader(self, path: str, stage: ShaderStage, *,
                       source: Optional[str | bytes] = None,
                       include_dirs: Sequence[str] = (),
                       entry_point: str = "") -> ShaderModule:
        """Compile or load a shader. One function for every form: the extension
        of `path` decides how it is handled.

        - `.hlsl` — HLSL. Use `[[vk::binding(n, set)]]` on resources; bare
          `register()` piles everything into one Vulkan binding space.
        - `.spv` — a prebuilt SPIR-V binary: loaded, not compiled. `stage` is
          verified against the binary's entry points (ShaderError on mismatch).
        - anything else — GLSL.

        `source=` supplies the content instead of reading a file, and its type
        says what it is. A `str` is compiled as text. `bytes` is taken as ready
        SPIR-V words: nothing is compiled, the extension of `path` stops
        mattering, and the binary gets the same magic-number and stage checks a
        `.spv` file gets. With `source=`, `path` becomes a virtual name that
        still picks the language, tags diagnostics (ShaderError.path) and anchors
        relative #include resolution (a name with no directory resolves includes
        against the working directory).

        `entry_point=` names an HLSL entry point, for a file that holds several
        (VSMain, PSMain). It is an error for GLSL, whose entry point must be
        main. The default is main.

        GLSL `#include "x"` / `<x>` resolve relative to the directory of the
        including file, recursively; the files used are recorded in
        ShaderModule.includes. `include_dirs=` adds fallback directories, tried
        in order and only when the name is not beside the including file, so
        adding one cannot change what an existing shader includes. A missing
        top-level file is a ResourceError; a missing include is a ShaderError
        (the compiler discovered it, and the error is recoverable — fix the
        include and recompile).

        A hot reload recompiles with the include_dirs and entry_point of the
        first compile.
        """
        ...

    def load_image(self, data: bytes, *, mipmaps: bool = True, name: str = "") -> Image:
        """Decode encoded image BYTES rather than a file: a PNG off the network,
        out of a zip, or straight from PIL, none of which has a path on disk.

        Everything after the decode is the file path's path — async, sRGB,
        mipped by default. Never hot-reloaded, by construction: there is no file
        for bazalt to watch.

        Raises ResourceError when the bytes are not a decodable image.
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
        automatically. `img.ready`, `img.wait()` and `ctx.wait()`
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
    def upload_progress(self) -> float:
        """0.0 .. 1.0 for the current batch of uploads (1.0 when idle) — a
        loading bar without user-side threads:

            while ctx.upload_progress < 1.0:
                draw_progress(ctx.upload_progress)

        "Batch" means everything queued since the last time uploads fully
        drained: once all in-flight uploads finish, progress resets to 1.0 and
        the next load_image starts a fresh batch from 0. This is the final
        semantics (settled in 0.9) — a second loading screen counts only its own
        images, not the ones a previous screen already finished.

        Covers both kinds: the load_image decodes on the worker, and the
        one-shot copies of create_buffer and create_image(array), which have
        nothing to decode and join the batch already submitted."""
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
        is a mistake — a cubemap needs 6 faces, so pass a list (below).

        ASYNCHRONOUS since 0.18.0, like load_image: the copy is submitted here
        (so every error still raises here) but not waited for. `img.ready` is
        therefore False right after the call, and that is not a problem to solve
        — a submit that samples the image waits for it GPU-side, and read()
        waits CPU-side."""
        ...
    def create_image(self, images: Sequence[Any], *, mipmaps: bool = False,
                     cube: bool = False, name: str = "") -> Image:
        """From a list of numpy arrays → a layered image: a texture array, or a
        cubemap when `cube=True` (exactly 6 square faces, order
        +X,-X,+Y,-Y,+Z,-Z). Every layer must share shape and dtype. `mipmaps=True`
        generates the full chain across every layer."""
        ...
    def create_image(self, source: Image, *, name: str = "") -> Image:
        """From an Image on another Context (or this one — that is a clone).

            texture = viewer_ctx.create_image(baked_on_the_other_gpu)

        Carries size, format, array layers, cube-ness and whether the source was
        mipped. `other.create_image(img.read())` cannot: read() returns mip 0 of
        layer 0 as a bare array, so a cubemap would arrive flattened.

        Blocking on the SOURCE Context, and it routes through host memory —
        without external memory there is no portable device-to-device path. A
        setup step, not a per-frame one. The result is an ordinary async Image
        of this Context (`.ready()` / `.wait()`).

        Mip levels above 0 are regenerated here rather than copied, so a
        hand-authored chain (rendered per level) becomes a generated one.
        ResourceError if the source is multisampled or still has no contents."""
        ...
    def create_sampler(self, filter: Filter = Filter.LINEAR,
                       address_mode: AddressMode = AddressMode.REPEAT,
                       anisotropy: bool = True,
                       compare: Optional[CompareOp] = None,
                       border_color: BorderColor = BorderColor.OPAQUE_BLACK,
                       mip_lod_bias: float = 0.0,
                       name: str = "") -> Sampler:
        """Cached: identical descriptions return the identical object.

        `compare=` makes a compare sampler (GLSL `sampler2DShadow`): reads
        return the comparison result instead of the texel, and LINEAR filtering
        becomes hardware PCF. (Linear filtering of depth formats is a format
        feature — universal on desktop GPUs, not spec-guaranteed.)

        `border_color=` is read outside 0..1 with
        `address_mode=AddressMode.CLAMP_TO_BORDER`. OPAQUE_WHITE is the
        shadow-map answer: "nothing occludes here", instead of CLAMP smearing
        the edge texel of the map across the rest of the scene.

        `mip_lod_bias=` shifts the mip level a sample reads: negative sharpens,
        positive blurs.

        `name=` labels the sampler for validation messages. Because the cache
        shares one sampler between identical descriptions, names accumulate: two
        calls that differ only by name give one object named "a + b", which
        `sampler.name` reports."""
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

    def submit(self, cmd: CommandBuffer, *, wait: bool = True) -> None:
        """Execute a command buffer with no swapchain and no present.

        Blocking by default, which is right when the next line reads the result.

        wait=False returns as soon as the work is queued. A compute prototype
        that submits in a loop wants this: with the wait, the GPU idles between
        iterations and the loop runs at the speed of the round trip rather than
        of the work. Call ctx.wait() before reading anything back.

        Reusing one CommandBuffer asynchronously is safe: the ring paces it, so
        a submit into a slot whose previous submit is still running waits for
        that one first. frames_in_flight is therefore how many submits can be in
        flight at once.
        """
        ...

    def wait(self) -> None:
        """Block until everything this Context started has finished — every
        upload and every submit.

        The one wait verb, and the other half of submit(wait=False). This is
        also where deferred destruction is reclaimed for that work. Waits on the
        submission timeline rather than on the device, so the other Contexts
        sharing the GPU are unaffected. Calling it with nothing outstanding does
        nothing.

        To wait for less, wait on the resource: buf.wait() / img.wait().
        """
        ...

class SwapchainRenderer(RenderTargetBase):
    """Presents to a window. One implementation of a render target."""

    def __init__(self, window: Window, context: Context,
                 present_mode: PresentMode = PresentMode.MAILBOX, samples: int = 1,
                 stencil: bool = False) -> None:
        """samples>1 turns on windowed MSAA: rendering goes into a multisampled
        colour+depth image that resolves into the swapchain image on present.
        Must be a power of two <= ctx.max_samples().

        stencil=True gives the window's depth buffer a stencil aspect, which is
        what a masked pass (an outline, a portal) needs on screen. Without it a
        stencil_test() pipeline can only be built against an offscreen
        RenderTarget with depth=Format.DEPTH_STENCIL."""
        ...
    def __init__(self, win32_hwnd: int, context: Context,
                 present_mode: PresentMode = PresentMode.MAILBOX, samples: int = 1,
                 stencil: bool = False) -> None:
        """Attach to an existing native window (Windows only)."""
        ...

    @property
    def present_mode(self) -> PresentMode:
        """The mode actually in use (post-fallback), not the requested one."""
        ...

    def set_present_mode(self, mode: PresentMode) -> None:
        """Switch vsync at runtime. Recreates the swapchain.

        The mode is a preference, so read present_mode back to see what the
        driver gave you. Call it outside the acquire/present pair: raises
        ResourceError while an image is acquired, because recreation would
        destroy the swapchain that image belongs to.
        """
        ...

    def acquire(self) -> bool:
        """Take this window's next swapchain image, inside the frame
        ctx.begin_frame() opened.

        False when this window should sit the frame out (minimized,
        mid-resize) — the other windows carry on:

            ctx.begin_frame()
            if renderer.acquire():
                renderer.present(cmd)

        Acquiring twice on one frame raises ResourceError; in practice that
        means ctx.begin_frame() was not called.
        """
        ...

    def present(self, cmd: CommandBuffer, *, capture: bool = False) -> None:
        """Record the command buffer for this window, submit it and present.

        ResourceError when there is no acquired image — acquire() was not
        called, returned False, or its image was already presented. Each window
        needs its own CommandBuffer: one holds a single command buffer per frame
        slot, so replaying it in two windows would overwrite work in flight.

        capture=True copies this frame's image into a staging buffer as part of
        the same submit, for read_pixels() to collect. It costs a full-frame copy
        on the frames that ask and nothing on the ones that do not.
        """
        ...

    def read_pixels(self) -> Any:
        """The frame captured by present(capture=True), as an (h, w, 4) uint8
        array. Blocks until that frame's submit has finished.

        It takes two calls, and that is not an oversight. A presentable image may
        only be touched between acquire and present, so reading the LAST frame is
        illegal by the spec and the validation layer reports it. The copy has to
        ride the frame's own submit.

        Raises ResourceError when nothing has been captured, or when the
        compositor refused to let the swapchain images be copied from — render
        into a bz.RenderTarget and read that instead.
        """
        ...

    @property
    def gpu_time_ms(self) -> Optional[float]:
        """GPU time in milliseconds of the frame submitted frames_in_flight ago
        (a timestamp pair around each submit, read back once its fence signals).
        None unless the Context was created with gpu_timing=True; also None
        until the ring has cycled once, and on devices without timestamp
        support."""
        ...

    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...

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
