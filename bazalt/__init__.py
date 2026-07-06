"""Bazalt — Python library for rapid GPU shader prototyping using Vulkan."""

from bazalt._core import *  # noqa: F401, F403
from bazalt._core import (  # noqa: F401 — explicit re-exports for IDE visibility
    Engine,
    Buffer,
    ShaderModule,
    Texture,
    Pipeline,
    PipelineBuilder,
    CommandBuffer,
    MouseState,
    BufferType,
    DataType,
    ShaderStage,
    Format,
    CullMode,
    FrontFace,
)

__version__ = "0.0.1"

__all__ = [
    # Core
    "Engine",
    # Resources
    "Buffer",
    "ShaderModule",
    "Texture",
    "Pipeline",
    "PipelineBuilder",
    "CommandBuffer",
    # Data types
    "MouseState",
    "BufferType",
    "DataType",
    "ShaderStage",
    "Format",
    "CullMode",
    "FrontFace",
    # Version
    "__version__",
]
