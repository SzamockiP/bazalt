"""The .pyi is hand-written, so it drifts unless something checks it.

These tests are cheap and catch the common failure: an API renamed in C++ and
forgotten in the stub, which silently misleads every user's type checker.
"""

import pathlib
import re

import pytest

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
                     "class RenderTarget(", "class BazaltError",
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
                     "stencil: bool = False", "INDEPENDENT_BLEND",
                     # 0.18
                     "def blit_image", "def copy_buffer", "def fill_buffer",
                     "def occlusion_query", "class OcclusionQuery",
                     "def label", "class LabelScope",
                     "def memory_stats", "class MemoryStats",
                     "def subgroup_size", "shader_printf",
                     "def read_pixels", "capture: bool = False",
                     "wait: bool = True", "def ready", "def upload_progress",
                     # 0.19
                     "TESS_CONTROL", "TESS_EVALUATION", "GEOMETRY = 5",
                     "TESSELLATION = 8", "GEOMETRY_SHADER = 9", "PATCH_LIST = 5",
                     "def tess_control_shader", "def tess_evaluation_shader",
                     "def geometry_shader", "def patch_control_points",
                     "color: Optional[Image | Sequence[Image]]",
                     "def dropped_files", "def set_cursor_position", "def set_icon",
                     "def get_clipboard", "def set_clipboard",
                     "INDIRECT_READ = 5", "def draw_indirect",
                     "def draw_indexed_indirect", "def dispatch_indirect",
                     "def writes", "def writes_unknown", "def prints",
                     "FRAGMENT_STORES = 10", "VERTEX_STAGE_STORES = 11",
                     # 0.22 — the portability rows, which answer True on every
                     # full Vulkan driver and False on a subset such as MoltenVK.
                     "COMPARISON_SAMPLER = 15", "SAMPLER_MIP_LOD_BIAS = 16",
                     "MULTISAMPLE_ARRAYS = 17"):
        assert expected in text, f"{expected!r} missing from _core.pyi"


def test_the_verbs_0_18_removed_are_gone():
    """0.18 collapsed several duplicate paths. Each name below had exactly one
    replacement, and a stub that still declares one is a stub promising an
    attribute the module does not have."""
    text = stub_text()
    for gone, replacement in (("def wait_idle", "ctx.wait()"),
                              ("def wait_for_uploads", "ctx.wait()"),
                              ("def uploads_done", "ctx.upload_progress"),
                              ("def should_close", "not window.is_open()")):
        assert gone not in text, f"{gone!r} is still in _core.pyi; use {replacement}"

    for cls, attr in ((bz.Context, "wait_idle"), (bz.Context, "wait_for_uploads"),
                      (bz.Context, "uploads_done"), (bz.Window, "should_close"),
                      (bz.RenderTarget, "read_pixels"), (bz.RenderTarget, "mip")):
        assert not hasattr(cls, attr), f"{cls.__name__}.{attr} is still bound"

    # The survivors of the same audit. read_pixels stays on the renderer because
    # a screenshot is different work; the two begin/end pairs stay because a
    # recording can be split across functions, which no `with` block spans.
    assert hasattr(bz.SwapchainRenderer, "read_pixels")
    for pair in ("begin_rendering", "end_rendering", "begin_label", "end_label"):
        assert hasattr(bz.CommandBuffer, pair)


def _bound_classes():
    return [obj for name in bz.__all__
            if isinstance(obj := getattr(bz, name), type)]


def test_every_bound_parameter_has_a_name():
    """A `.def()` without `py::arg()` registers no parameter name, so only a
    positional call works — while the stub names the parameter and invites a
    keyword call that raises TypeError. Ten methods were in that state through
    0.19, and nothing noticed because the README, the tests and all 28 examples
    happen to call them positionally.

    pybind11 writes the signature into __doc__ and falls back to `arg0`, `arg1`
    for unnamed parameters, so the whole class of mistake is one scan. This
    replaces a list of ten, which would go stale the next time somebody adds a
    binding in a hurry."""
    unnamed = re.compile(r"\barg\d+:")
    offenders = []
    for cls in _bound_classes():
        for name, attr in vars(cls).items():
            # __enter__/__exit__ and the other dunders are called by the
            # interpreter, always positionally. Naming their parameters would be
            # decoration nobody can use.
            if name.startswith("__"):
                continue
            doc = getattr(attr, "__doc__", None) or ""
            for line in doc.splitlines():
                if unnamed.search(line):
                    offenders.append(f"{cls.__name__}.{name}: {line.strip()}")
    assert not offenders, "py::arg() missing, so these take no keyword arguments:\n" + "\n".join(offenders)


def test_the_bare_enum_names_are_gone():
    """0.20 dropped `.export_values()` from all 15 enums that had it.

    It binds every member a SECOND time as a bare module attribute, so
    `bz.ShaderStage.VERTEX` had a twin `bz.VERTEX` — around 60 of them, none in
    the stub and none in `__all__`. Two collided outright: `bz.VERTEX` resolved
    to ShaderStage and `bz.FLOAT` to VertexFormat, because whichever enum bound
    last silently won. The qualified spelling is the only one."""
    for bare in ("VERTEX", "FLOAT", "NONE", "ERROR", "INFO", "WARNING", "LINE",
                 "POINT", "FILL", "ALPHA", "STATIC", "DYNAMIC", "DEVICE",
                 "SHADER", "WINDOW", "INDEX", "STORAGE", "UNIFORM", "GENERAL",
                 "COMPUTE", "FRAGMENT", "WINDOWED", "CLOCKWISE"):
        assert not hasattr(bz, bare), (
            f"bz.{bare} exists again — some enum got .export_values() back")


def test_stub_does_not_reference_an_undefined_buffer_type():
    """The old stub annotated arrays as `buffer`, which is not a Python type.

    Any type checker flags it, which trains users to ignore the stub.
    """
    assert not re.search(r":\s*buffer\b", stub_text())


def test_exception_hierarchy_matches_the_stub():
    for name in ("BazaltError", "InitializationError", "DeviceLostError",
                 "OutOfMemoryError", "ShaderError", "WindowError", "ResourceError",
                 "StateError", "UnsupportedError"):
        assert f"class {name}(" in stub_text()
        assert hasattr(bz, name)


def test_version_is_declared():
    assert bz.__version__


def test_the_two_version_strings_agree():
    """0.19: the version lives in pyproject.toml AND bazalt/__init__.py, and until now
    nothing compared them — so a release could ship a wheel whose metadata and
    `bz.__version__` disagreed, which is the kind of thing nobody notices until a
    bug report quotes the wrong one.

    Skipped from an installed wheel, where there is no pyproject.toml to read.
    """
    pyproject = pathlib.Path(bz.__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("running against an installed wheel, not the source tree")
    declared = re.search(r'^version = "([^"]+)"', pyproject.read_text(encoding="utf-8"), re.M)
    assert declared, "pyproject.toml has no version line"
    assert declared.group(1) == bz.__version__, (
        f"pyproject.toml says {declared.group(1)} but bazalt.__version__ is {bz.__version__}")
