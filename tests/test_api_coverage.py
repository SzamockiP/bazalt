"""The coverage gate is only worth trusting if its two decisions are right:
what counts as a public symbol, and what counts as untouched. Both are pure
functions, so they are tested here rather than by reading the gate's output.
"""

import api_coverage


def test_the_surface_classifies_each_kind():
    surface = api_coverage.public_surface()

    assert surface["Context.create_buffer"] == "method"
    assert surface["Context.__init__"] == "method"
    assert surface["Image.width"] == "property"
    assert surface["poll_events"] == "function"


def test_the_surface_counts_only_callables():
    """Enum members and exception classes are read, never called, so the only
    evidence a census could collect was a regex scan of the test sources —
    dropped in 0.27 as measurement theater. `test_stubs.py` asserts they exist,
    which is everything a constant can be wrong about."""
    surface = api_coverage.public_surface()

    assert "Format.RGBA8" not in surface
    assert "Key.SPACE" not in surface
    assert "ShaderError" not in surface
    assert all(kind in ("method", "property", "function") for kind in surface.values())


def test_the_surface_skips_an_init_that_only_raises():
    """A class with no py::init still has an `__init__` in its dict — the slot
    wrapper pybind installs to raise TypeError. Counting it asks for a test that
    constructs what 0.23 made unconstructible on purpose, and 23 of the 26
    untouched methods were exactly that."""
    surface = api_coverage.public_surface()

    assert "RenderTarget.__init__" not in surface
    assert "SwapchainRenderer.__init__" not in surface
    assert "Image.__init__" not in surface
    assert surface["Window.__init__"] == "method"


def test_untouched_is_what_no_test_called():
    surface = {"Thing.called": "method", "Thing.never_called": "method"}

    assert api_coverage.untouched(surface, {"Thing.called"}) == ["Thing.never_called"]
