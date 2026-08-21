"""Which of the public API does the suite actually touch.

`test_stubs.py` proves the stub and the module agree about what EXISTS. This
answers the other question, the one "What 1.0 means" asks in DESIGN.md: which
public symbols does no test ever use. Run it with `--api-coverage`; the gate in
conftest.py fails on an untouched symbol that is not in the baseline file.

The census counts what can be CALLED — methods, properties, module functions.
Every one is replaced at session start with a wrapper that records the call and
forwards. This is exact: it separates `Buffer.update` from `Image.update`,
which is the distinction a text search cannot make. (The first attempt used
`sys.setprofile`, which is wrong for pybind11: a bound pybind11 method is a
plain `method` around a custom function record, not a `PyCFunction`, so the
interpreter emits no `c_call` event for it.)

What is NOT counted, on purpose: enum members and exception classes. Neither
can be called, so the only available evidence was a regex scan for their names
in the test sources — a mention in a comment counted as a use, and 0.27 dropped
that as measurement theater. `test_stubs.py` already asserts they exist, and an
exception's real test is the behavior test that raises it. A class with no
py::init keeps its raising `__init__` slot wrapper out of the census too:
counting it asks for a test that constructs what cannot be constructed.
"""

import functools
import pathlib

import bazalt as bz

# Everything else that starts with an underscore is implementation.
_DUNDERS = ("__init__", "__enter__", "__exit__")

_BASELINE = pathlib.Path(__file__).parent / "api_coverage_baseline.txt"


def public_surface():
    """{"Owner.member": kind} for every callable reachable from `bazalt.__all__`.

    Derived, never hand-written: a binding added to `src/bindings/` and to
    `__all__` appears here on the next run with no edit.
    """
    surface = {}
    for name in bz.__all__:
        if name.startswith("__"):
            continue
        obj = getattr(bz, name)

        if not isinstance(obj, type):
            surface[name] = "function"
            continue

        # Exceptions and enums contribute nothing — see the module docstring.
        if issubclass(obj, BaseException) or hasattr(obj, "__members__"):
            continue

        for member, value in vars(obj).items():
            if member.startswith("_") and member not in _DUNDERS:
                continue
            # A class with no py::init still has an `__init__` in its dict: the
            # slot wrapper pybind installs to raise TypeError.
            #
            # The discriminator is the slot wrapper, not pybind's own
            # `instancemethod`, because Recorder has usually replaced a real
            # `__init__` with a plain function by the time anything asks.
            if member == "__init__" and type(value).__name__ == "wrapper_descriptor":
                continue
            key = f"{name}.{member}"
            if isinstance(value, property):
                surface[key] = "property"
            elif callable(value):
                surface[key] = "method"
    return surface


class Recorder:
    """Records which callables ran, by installing a wrapper on each one."""

    def __init__(self):
        self.used = set()

    def install(self, surface):
        for key, kind in surface.items():
            owner_name, _, member = key.partition(".")
            owner = getattr(bz, owner_name) if member else bz
            member = member or owner_name

            if kind == "method":
                setattr(owner, member, self._wrap(key, getattr(owner, member)))
            elif kind == "function":
                setattr(bz, member, self._wrap(key, getattr(bz, member)))
            elif kind == "property":
                prop = getattr(owner, member)
                setattr(
                    owner,
                    member,
                    property(
                        self._wrap(key, prop.fget),
                        prop.fset and self._wrap(key, prop.fset),
                        prop.fdel,
                        prop.__doc__))

    def _wrap(self, key, function):
        # functools.wraps keeps __doc__, which pybind11 writes the signature
        # into -- test_stubs.py reads it to find a .def() missing a py::arg().
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            self.used.add(key)
            return function(*args, **kwargs)

        return wrapper


def untouched(surface, used):
    """The symbols no test called."""
    return sorted(key for key in surface if key not in used)


def read_baseline():
    if not _BASELINE.exists():
        return set()
    return {
        line.strip()
        for line in _BASELINE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def write_baseline(missing):
    header = [
        "# Public symbols no test touches yet, as measured by",
        "# `pytest --api-coverage` (see tests/api_coverage.py).",
        "#",
        "# The list is a gate, not a target: a symbol that turns up untouched and is",
        "# NOT here fails the run, which is what catches a binding shipped with no",
        "# test. Shrinking it is 1.0's work. Regenerate with",
        "# BAZALT_WRITE_API_BASELINE=1 pytest --api-coverage.",
    ]
    _BASELINE.write_text("\n".join(header + sorted(missing)) + "\n", encoding="utf-8")
