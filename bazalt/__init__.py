"""Bazalt — Python library for rapid GPU shader prototyping using Vulkan."""

from bazalt._core import *

__version__ = "0.30.0"

__all__ = [
    # Core
    "Context",
    "RenderTarget",
    "RenderTargetBase",
    "SubresourceTarget",
    "MultiviewTarget",
    "SwapchainRenderer",
    "Window",
    "Device",
    "list_devices",
    # Also process-wide, and the one such query that needs no live Window:
    # choosing where to open one happens first (0.25).
    "Monitor",
    "VideoMode",
    "list_monitors",
    "poll_events",
    # The sleeping half of the same pump, for a program that only redraws on
    # input (0.24).
    "wait_events",
    # Process-wide, like poll_events: the clipboard is not per window.
    "get_clipboard",
    "set_clipboard",
    # Same reason again: a gamepad belongs to the process, not to a window.
    "get_gamepad",
    "Gamepad",
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
    "Graph",
    "Pass",
    "Queue",
    "Serial",
    "Timer",
    "LabelScope",
    "OcclusionQuery",
    # Data types
    "MemoryStats",
    "Limits",
    "MouseState",
    "WindowMode",
    "BufferUsage",
    "ShaderStage",
    "ShaderLanguage",
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
    "BlendFactor",
    "BlendOp",
    "StencilOp",
    "BorderColor",
    "MemoryUsage",
    "Severity",
    "Source",
    "Feature",
    "GamepadButton",
    "GamepadAxis",
    "Key",
    "MouseButton",
    "CursorMode",
    "Cursor",
    "Face",
    # Errors — the type is the recoverability contract
    "BazaltError",
    "InitializationError",
    "DeviceLostError",
    "OutOfMemoryError",
    "ShaderError",
    "WindowError",
    "ResourceError",
    "StateError",
    "UnsupportedError",
    # Version
    "__version__",
]
