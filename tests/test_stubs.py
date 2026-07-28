"""The .pyi is hand-written, so it drifts unless something checks it.

These tests are cheap and catch the common failure: an API renamed in C++ and
forgotten in the stub, which silently misleads every user's type checker.
"""

import pathlib
import re

import bazalt as bz

PYI = pathlib.Path(bz.__file__).parent / "_core.pyi"


def stub_text():
    return PYI.read_text(encoding="utf-8")


def test_stub_exists_and_package_is_typed():
    assert PYI.is_file()
    assert (PYI.parent / "py.typed").is_file()


def test_every_public_name_is_in_all():
    """`from bazalt import *` should give the same names as `bz.<name>`."""
    for name in bz.__all__:
        assert hasattr(bz, name), f"{name} is in __all__ but missing from the module"


def test_removed_names_are_gone_from_the_stub():
    """Names dropped in 0.4/0.5 must not linger in the stub.

    (0.4 removed `Format` as a vertex-attribute enum; 0.5 reintroduced the
    name for pixel formats, which is why it is asserted PRESENT below.)
    """
    text = stub_text()
    assert "def on_error" not in text, "on_error was replaced by on_message"
    assert "class Texture" not in text, "Texture was split into Image + Sampler"
    assert "load_texture" not in text, "load_texture became load_image"
    assert "set_texture" not in text, "set_texture became set_image"
    # 0.6: the builder split. Trailing colon/parenthesis on purpose —
    # "PipelineBuilder" is a substring of "GraphicsPipelineBuilder".
    assert "class PipelineBuilder:" not in text, "PipelineBuilder became GraphicsPipelineBuilder"
    assert "def pipeline_builder" not in text, "pipeline_builder became graphics_pipeline"
    # 0.14: the frame moved to the Context, so the window's verbs are acquire()
    # and present() and there is nothing left for a Frame object to be.
    assert "class Frame" not in text, "Frame was split into ctx.begin_frame + renderer.acquire/present"
    # 0.14: glfwPollEvents is process-wide, so it stopped pretending to be a
    # method on one window.
    assert "def poll_events(self)" not in text, "Window.poll_events became the free bz.poll_events()"
    # 0.17: an instance count is an argument of a draw, not a second verb.
    # "def " on purpose: the migration note in draw_indexed's docstring names
    # the old method, and that mention is the point of it.
    assert "def draw_indexed_instanced" not in text, "draw_indexed_instanced became draw_indexed(instances=)"


def test_renamed_and_new_api_is_declared():
    text = stub_text()
    for expected in ("class VertexFormat(", "def on_message", "class LogMessage",
                     "class Severity(", "class Source(", "class Feature(",
                     "class RenderTarget(", "def read_pixels", "class BazaltError",
                     "def flush",
                     # 0.5
                     "class Format(", "class Image", "class Sampler",
                     "def load_image", "def create_image",
                     "def create_sampler", "def set_image",
                     # 0.6
                     "class GraphicsPipelineBuilder:", "class ComputePipelineBuilder:",
                     "def graphics_pipeline", "def compute_pipeline",
                     "class Topology(", "def topology",
                     "class Access(", "def dispatch", "def barrier",
                     "auto_barriers",
                     # 0.14
                     "class Device", "def list_devices", "def begin_frame",
                     "def acquire", "def present", "FIFO_RELAXED",
                     "def poll_events()",
                     # 0.16
                     "class WindowMode(", "def set_mode", "FULLSCREEN_WINDOWED",
                     "def was_key_pressed", "def scroll_dy", "def content_scale",
                     "def set_present_mode", "class BlendMode(", "PREMULTIPLIED",
                     "class PolygonMode(", "def polygon_mode",
                     "clear_depth", "def line_width", "def depth_bias",
                     "include_dirs", "entry_point",
                     # 0.17
                     "def instance_format", "UBYTE4_NORM", "TRIANGLE_STRIP",
                     "instances: int = 1", "binding: int = 0",
                     "class StencilOp(", "def stencil_test", "DEPTH_STENCIL",
                     "clear_stencil", "def color_mask", "def depth_clamp",
                     "def alpha_to_coverage", "def constant", "def copy_image",
                     "def clear_image", "R32_UINT",
                     "class BorderColor(", "CLAMP_TO_BORDER", "mip_lod_bias",
                     "stencil: bool = False", "INDEPENDENT_BLEND"):
        assert expected in text, f"{expected!r} missing from _core.pyi"


def test_stub_does_not_reference_an_undefined_buffer_type():
    """The old stub annotated arrays as `buffer`, which is not a Python type.

    Any type checker flags it, which trains users to ignore the stub.
    """
    assert not re.search(r":\s*buffer\b", stub_text())


def test_exception_hierarchy_matches_the_stub():
    for name in ("BazaltError", "InitializationError", "DeviceLostError",
                 "OutOfMemoryError", "ShaderError", "WindowError", "ResourceError"):
        assert f"class {name}(" in stub_text()
        assert hasattr(bz, name)


def test_version_is_declared():
    assert bz.__version__
