"""The .pyi is hand-written, so it drifts unless something checks it.

These tests are cheap and catch the common failure: an API renamed in C++ and
forgotten in the stub, which silently misleads every user's type checker.

Everything here is DERIVED from the module. 0.27 replaced the hand-typed
substring lists that grew by a block every release: a list checks that the stub
contains strings someone once typed, a derivation checks the stub against what
the module really exports, and only the second survives the next rename.
"""

import inspect
import pathlib
import re
import sys

import pytest

import bazalt as bz

PYI = pathlib.Path(bz.__file__).parent / "_core.pyi"


def stub_text():
    return PYI.read_text(encoding="utf-8")


def test_this_module_needs_no_gpu():
    """This file is the only test CI runs on every wheel (CIBW_TEST_COMMAND),
    and those runners have no Vulkan driver. A test here that asks for the `ctx`
    fixture fails at SETUP with InitializationError and takes all three wheel
    jobs with it — 0.23 shipped exactly that for one commit, and the failure
    reads like a broken wheel rather than a misplaced test.

    The rule is "no parameters" rather than "not ctx" because it needs no
    maintenance: a driver-free fixture that is genuinely wanted here can be
    added to this assertion, and the choice becomes deliberate."""
    for name, function in vars(sys.modules[__name__]).items():
        if not (name.startswith("test_") and inspect.isfunction(function)):
            continue
        taken = list(inspect.signature(function).parameters)
        assert not taken, f"{name} takes {taken}; test_stubs.py must pass with no GPU"


def test_stub_exists_and_package_is_typed():
    assert PYI.is_file()
    assert (PYI.parent / "py.typed").is_file()


def test_every_public_name_is_in_all():
    """`from bazalt import *` should give the same names as `bz.<name>`."""
    for name in bz.__all__:
        assert hasattr(bz, name), f"{name} is in __all__ but missing from the module"


def test_the_stub_declares_every_public_symbol():
    """Every class in __all__ appears as `class Name`, every enum member with
    its exact value, every public method and property as a `def`, and every
    module function as a `def`. A binding added, renamed or removed in
    src/bindings/ without the stub fails here on the next run.

    The method check is a substring on the whole stub, so a method missing from
    one class while a same-named method exists on another can slip through —
    the enum values and class names are exact, and that trade keeps this a scan
    rather than a parser."""
    text = stub_text()
    missing = []
    for name in bz.__all__:
        if name.startswith("__"):
            continue
        obj = getattr(bz, name)

        if not isinstance(obj, type):
            if not re.search(rf"^def {name}\(", text, re.M):
                missing.append(f"function {name}")
            continue

        if not re.search(rf"^class {name}\b", text, re.M):
            missing.append(f"class {name}")
            continue

        members = getattr(obj, "__members__", None)
        if members is not None:
            for member, value in members.items():
                if not re.search(rf"^    {member} = {int(value)}$", text, re.M):
                    missing.append(f"{name}.{member} = {int(value)}")
            continue

        for member, attr in vars(obj).items():
            if member.startswith("_"):
                continue
            # A property may be spelled `def member(` or as a bare
            # `member: Type` annotation — both are valid stub forms.
            if isinstance(attr, property):
                if f"def {member}(" not in text and not re.search(
                        rf"^    {member}: ", text, re.M):
                    missing.append(f"{name}.{member}")
            elif callable(attr):
                if f"def {member}(" not in text:
                    missing.append(f"{name}.{member}")

    assert not missing, "declared by the module, absent from _core.pyi:\n  " + "\n  ".join(missing)


def test_no_enum_member_leaks_as_a_bare_module_name():
    """0.20 dropped `.export_values()` from every enum: it binds each member a
    SECOND time as a bare module attribute (`bz.VERTEX` beside
    `bz.ShaderStage.VERTEX`), and two of those collided outright — whichever
    enum bound last silently won. Derived, so an enum that gets
    `.export_values()` back fails here whatever its members are called."""
    for name in bz.__all__:
        obj = getattr(bz, name, None)
        members = getattr(obj, "__members__", None) if isinstance(obj, type) else None
        if not members:
            continue
        for member in members:
            if member not in bz.__all__:
                assert not hasattr(bz, member), (
                    f"bz.{member} exists — {name} got .export_values() back")


def test_the_target_types_are_not_constructible():
    """0.23: a RenderTarget and a SwapchainRenderer come from the Context, and
    the constructors are GONE rather than kept beside the factories — a second
    spelling of one call is a fork (the 0.18 audit). The classes stay as types,
    which is what isinstance and the annotations name.

    No Context here on purpose — see test_this_module_needs_no_gpu. The refusal
    happens before any argument is looked at, so None is as good as a device,
    and the factory half of the claim is asserted in test_targets.py where a
    Context already exists."""
    for cls in (bz.RenderTarget, bz.SwapchainRenderer):
        with pytest.raises(TypeError):
            cls(None, 16, 16)

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
