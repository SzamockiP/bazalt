"""Bazalt — Python library for rapid GPU shader prototyping using Vulkan."""

from bazalt import _core
from bazalt._core import *  # noqa: F401, F403

__version__ = "0.20.0"

__all__ = [
    # Core
    "Context",
    "RenderTarget",
    "RenderTargetBase",
    "SwapchainRenderer",
    "Window",
    "Device",
    "list_devices",
    "poll_events",
    # Process-wide, like poll_events: the clipboard is not per window.
    "get_clipboard",
    "set_clipboard",
    "Logger",
    "LogMessage",
    # Resources
    "Buffer",
    "ShaderModule",
    "Image",
    "Sampler",
    "Pipeline",
    "GraphicsPipelineBuilder",
    "ComputePipelineBuilder",
    "DescriptorPool",
    "DescriptorSet",
    "CommandBuffer",
    "RenderingScope",
    "Timer",
    "LabelScope",
    "OcclusionQuery",
    # Data types
    "MemoryStats",
    "MouseState",
    "WindowMode",
    "BufferType",
    "DataType",
    "ShaderStage",
    "VertexFormat",
    "Topology",
    "Access",
    "Format",
    "Filter",
    "AddressMode",
    "CompareOp",
    "PresentMode",
    "CullMode",
    "FrontFace",
    "PolygonMode",
    "BlendMode",
    "StencilOp",
    "BorderColor",
    "MemoryUsage",
    "Severity",
    "Source",
    "Feature",
    # Errors — the type is the recoverability contract
    "BazaltError",
    "InitializationError",
    "DeviceLostError",
    "OutOfMemoryError",
    "ShaderError",
    "WindowError",
    "ResourceError",
    # Version
    "__version__",
]

# The key/button/cursor constants are part of the public surface, so
# `from bazalt import *` has to carry them. Listed by prefix rather than by
# hand: there are 127 of them and a hand-written copy goes stale.
__all__ += [_n for _n in dir(_core) if _n.startswith(("KEY_", "MOUSE_BUTTON_", "CURSOR_"))]
