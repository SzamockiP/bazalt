"""Type stubs for lumapy._core (native Vulkan renderer)."""

from __future__ import annotations

from enum import IntEnum
from typing import Optional, Sequence, overload

# ── Enums ──────────────────────────────────────────────────────────────

class BufferType(IntEnum):
    """Type of GPU buffer."""
    VERTEX: int
    INDEX: int
    UNIFORM: int
    STORAGE: int

class DataType(IntEnum):
    """Element data type for buffer creation."""
    FLOAT: int
    UINT32: int
    UINT16: int
    INT32: int

class ShaderStage(IntEnum):
    """Shader pipeline stage."""
    VERTEX: int
    FRAGMENT: int

class Format(IntEnum):
    """Vertex attribute format."""
    FLOAT2: int
    FLOAT3: int
    FLOAT4: int

class CullMode(IntEnum):
    """Triangle culling mode."""
    NONE: int
    BACK: int
    FRONT: int
    FRONT_AND_BACK: int

class FrontFace(IntEnum):
    """Winding order for front-facing triangles."""
    CLOCKWISE: int
    COUNTER_CLOCKWISE: int

# ── Data Classes ───────────────────────────────────────────────────────

class MouseState:
    """Accumulated mouse movement state."""
    dx: float
    """Cumulative X movement since window creation."""
    dy: float
    """Cumulative Y movement since window creation."""

# ── GPU Resources ──────────────────────────────────────────────────────

class Buffer:
    """GPU buffer (vertex, index, uniform, or storage)."""

    @overload
    def update(self, data: bytes) -> None: ...
    @overload
    def update(self, array: buffer) -> None: ...
    @overload
    def update(self, list: list, dataType: Optional[DataType] = None) -> None: ...
    def update(self, *args, **kwargs) -> None:
        """Update buffer contents.

        Accepts raw bytes, a numpy array (buffer protocol), or a Python list.
        For lists, the data type is inferred from elements unless ``dataType``
        is specified explicitly.
        """
        ...

class ShaderModule:
    """Compiled SPIR-V shader module."""
    ...

class Texture:
    """GPU texture loaded from an image file."""

    @property
    def width(self) -> int:
        """Texture width in pixels."""
        ...

    @property
    def height(self) -> int:
        """Texture height in pixels."""
        ...

class Pipeline:
    """Compiled graphics pipeline (immutable after creation)."""
    ...

class PipelineBuilder:
    """Fluent builder for constructing graphics pipelines.

    All configuration methods return ``self`` for chaining.
    """

    def vertexShader(self, shader: ShaderModule) -> PipelineBuilder:
        """Set the vertex shader stage."""
        ...

    def fragmentShader(self, shader: ShaderModule) -> PipelineBuilder:
        """Set the fragment shader stage."""
        ...

    def vertexFormat(self, formats: list[Format]) -> PipelineBuilder:
        """Define vertex input layout as a list of attribute formats."""
        ...

    def depthTest(self, enable: bool) -> PipelineBuilder:
        """Enable or disable depth testing and writing."""
        ...

    def cullMode(self, mode: CullMode, frontFace: FrontFace) -> PipelineBuilder:
        """Set triangle culling mode and front-face winding order."""
        ...

    def blend(self, enable: bool) -> PipelineBuilder:
        """Enable or disable alpha blending."""
        ...

    def pushConstant(self, size: int, stage: ShaderStage) -> PipelineBuilder:
        """Declare a push constant range.

        Args:
            size: Size in bytes.
            stage: Shader stage that uses this constant.
        """
        ...

    def uniformBuffer(self, binding: int, stage: ShaderStage) -> PipelineBuilder:
        """Declare a uniform buffer descriptor binding."""
        ...

    def storageBuffer(self, binding: int, stage: ShaderStage) -> PipelineBuilder:
        """Declare a storage buffer descriptor binding."""
        ...

    def texture(self, binding: int, stage: ShaderStage) -> PipelineBuilder:
        """Declare a combined image sampler descriptor binding."""
        ...

    def build(self) -> Pipeline:
        """Compile the pipeline. Raises ``RuntimeError`` on failure."""
        ...

# ── Command Buffer ─────────────────────────────────────────────────────

class CommandBuffer:
    """Deferred GPU command recorder.

    Commands are recorded once via the ``begin…``/``end…`` methods,
    then replayed every frame when passed to ``Engine.submit()``.
    """

    def begin(self) -> None:
        """Clear previously recorded commands and start a new recording."""
        ...

    def beginRendering(self, *, clear_color: list[float]) -> None:
        """Begin a dynamic rendering pass.

        Args:
            clear_color: RGBA clear color as ``[r, g, b, a]`` (0.0–1.0).
        """
        ...

    def endRendering(self) -> None:
        """End the current rendering pass and transition to present layout."""
        ...

    def setViewport(self) -> None:
        """Set the viewport to cover the full swapchain extent."""
        ...

    def setScissor(self) -> None:
        """Set the scissor rectangle to cover the full swapchain extent."""
        ...

    def bindPipeline(self, pipeline: Pipeline) -> None:
        """Bind a graphics pipeline for subsequent draw calls."""
        ...

    def bindVertexBuffer(self, buffer: Buffer) -> None:
        """Bind a vertex buffer at binding 0."""
        ...

    def bindIndexBuffer(self, buffer: Buffer) -> None:
        """Bind an index buffer (uint32 indices)."""
        ...

    def draw(self, vertexCount: int) -> None:
        """Record a non-indexed draw call."""
        ...

    def drawIndexed(self, indexCount: int) -> None:
        """Record an indexed draw call (1 instance)."""
        ...

    def drawIndexedInstanced(self, indexCount: int, instanceCount: int) -> None:
        """Record an indexed, instanced draw call."""
        ...

    def pushConstants(
        self,
        pipeline: Pipeline,
        stage: ShaderStage,
        offset: int,
        data: bytes,
    ) -> None:
        """Upload push constant data.

        Args:
            pipeline: Pipeline whose layout defines the push constant range.
            stage: Target shader stage.
            offset: Byte offset into the push constant range.
            data: Raw bytes to upload.
        """
        ...

    def bindUniformBuffer(
        self, binding: int, buffer: Buffer, pipeline: Pipeline
    ) -> None:
        """Bind a uniform buffer to a descriptor set binding."""
        ...

    def bindStorageBuffer(
        self, binding: int, buffer: Buffer, pipeline: Pipeline
    ) -> None:
        """Bind a storage buffer to a descriptor set binding."""
        ...

    def bindTexture(
        self, binding: int, texture: Texture, pipeline: Pipeline
    ) -> None:
        """Bind a texture (combined image sampler) to a descriptor set binding."""
        ...

# ── Engine ─────────────────────────────────────────────────────────────

class Engine:
    """Main entry point — manages window, Vulkan context, and render loop."""

    def __init__(self) -> None: ...

    def init(self, width: int, height: int, title: str) -> None:
        """Create a window and initialize the Vulkan renderer.

        Args:
            width: Initial window width in pixels.
            height: Initial window height in pixels.
            title: Window title string.
        """
        ...

    def run(self, app_instance: object = None) -> None:
        """Enter the main render loop (blocks until the window is closed).

        If ``app_instance`` is provided, the frame callback receives it
        as its first argument.
        """
        ...

    def running(self) -> bool:
        """Return ``True`` while the render loop is active."""
        ...

    def stop(self) -> None:
        """Signal the render loop to exit after the current frame."""
        ...

    def onError(self, callback: ...) -> ...:
        """Register an error/log callback. Can be used as a decorator::

            @engine.onError
            def on_error(msg: str):
                print(msg)
        """
        ...

    def onFrame(self, callback: ...) -> ...:
        """Register the per-frame update function. Can be used as a decorator::

            @engine.onFrame
            def on_update():
                ...
        """
        ...

    def log(self, msg: str) -> None:
        """Send a message to the logger (delivered asynchronously to error callbacks)."""
        ...

    def setTitle(self, title: str) -> None:
        """Update the window title."""
        ...

    def getMouseState(self) -> MouseState:
        """Return the current accumulated mouse state."""
        ...

    def isKeyPressed(self, key: int) -> bool:
        """Check if a keyboard key is currently held down.

        Use module-level ``KEY_*`` constants for key codes.
        """
        ...

    def isMouseButtonPressed(self, button: int) -> bool:
        """Check if a mouse button is currently held down.

        Use ``MOUSE_BUTTON_LEFT``, ``MOUSE_BUTTON_RIGHT``,
        ``MOUSE_BUTTON_MIDDLE``.
        """
        ...

    def setCursorMode(self, mode: int) -> None:
        """Set cursor visibility/capture mode.

        Use ``CURSOR_NORMAL``, ``CURSOR_DISABLED``, or ``CURSOR_HIDDEN``.
        """
        ...

    def getDeltaTime(self) -> float:
        """Time in seconds since the last frame."""
        ...

    def getTime(self) -> float:
        """Elapsed time in seconds since ``run()`` was called."""
        ...

    def getFrameCount(self) -> int:
        """Total number of frames rendered since ``run()`` was called."""
        ...

    def getWidth(self) -> int:
        """Current window width in pixels (updates on resize)."""
        ...

    def getHeight(self) -> int:
        """Current window height in pixels (updates on resize)."""
        ...

    @overload
    def createBuffer(
        self, list: list, type: BufferType, dataType: Optional[DataType] = None
    ) -> Buffer: ...
    @overload
    def createBuffer(self, array: buffer, type: BufferType) -> Buffer: ...
    @overload
    def createBuffer(self, size_in_bytes: int, type: BufferType) -> Buffer: ...
    def createBuffer(self, *args, **kwargs) -> Buffer:
        """Create a GPU buffer.

        Three overloads:

        1. ``createBuffer(list, type, dataType=None)`` — from a Python list
        2. ``createBuffer(array, type)`` — from a numpy array (buffer protocol)
        3. ``createBuffer(size_in_bytes, type)`` — empty buffer of given size
        """
        ...

    def createCommandBuffer(self) -> CommandBuffer:
        """Allocate a new command buffer."""
        ...

    def createPipeline(self) -> PipelineBuilder:
        """Create a new pipeline builder."""
        ...

    def compileShader(self, path: str, stage: ShaderStage) -> ShaderModule:
        """Compile a GLSL shader file to SPIR-V at runtime.

        Args:
            path: Path to the ``.vert`` or ``.frag`` source file.
            stage: Shader stage (``VERTEX`` or ``FRAGMENT``).
        """
        ...

    def loadTexture(self, path: str) -> Texture:
        """Load an image file as a GPU texture (supports PNG, JPG, BMP, etc.)."""
        ...

    def submit(self, cmd: CommandBuffer) -> None:
        """Submit a recorded command buffer for the current frame."""
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
