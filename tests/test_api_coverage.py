"""The coverage report is only worth reading if its two decisions are right:
what counts as a public symbol, and what counts as untouched. Both are pure
functions, so they are tested here rather than by reading the report.

This file is excluded from the identifier scan (see api_coverage._SCAN_EXCLUDES)
because it names API symbols in string literals, and every one of them would
otherwise count as a use.
"""

import api_coverage


def test_the_surface_classifies_each_kind():
    surface = api_coverage.public_surface()

    assert surface["Context.create_buffer"] == "method"
    assert surface["Context.__init__"] == "method"
    assert surface["Image.width"] == "property"
    assert surface["Format.RGBA8"] == "enum member"
    assert surface["ShaderError"] == "exception"
    assert surface["poll_events"] == "function"


def test_the_surface_counts_the_keyboard_once():
    """0.23 dropped the KEY_*/MOUSE_*/CURSOR_* integers from the count. They are
    the pre-enum spelling of three enums the census already counts, so keeping
    both reported the keyboard twice — 116 untouched constants beside 99
    untouched Key members, one fact, the largest number in the report.

    `Key.SPACE` staying in is the point: the aliases were dropped, not the
    keyboard."""
    surface = api_coverage.public_surface()

    assert "KEY_SPACE" not in surface
    assert "MOUSE_BUTTON_LEFT" not in surface
    assert surface["Key.SPACE"] == "enum member"


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


def test_the_surface_holds_no_pybind_boilerplate():
    """An enum contributes its members only. pybind11 also puts `name`, `value`
    and `__members__` on every one of them, and they are not bazalt's API."""
    surface = api_coverage.public_surface()

    assert "Format.name" not in surface
    assert "Format.value" not in surface
    assert not [key for key in surface if "__members__" in key]
    assert not [key for key in surface if key.rpartition(".")[2].startswith("_pybind11")]


def test_untouched_asks_the_right_question_per_kind():
    """A callable has to be CALLED, a constant only named: nothing can wrap a
    value that is read rather than invoked."""
    surface = {
        "Thing.called": "method",
        "Thing.never_called": "method",
        "Enum.NAMED": "enum member",
        "Enum.UNNAMED": "enum member",
    }

    missing = api_coverage.untouched(surface, {"Thing.called"}, {"NAMED"})

    assert missing == [("Enum.UNNAMED", "enum member"), ("Thing.never_called", "method")]
