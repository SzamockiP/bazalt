"""Which of the public API does the suite actually touch.

`test_stubs.py` proves the stub and the module agree about what EXISTS. This
answers the other question, the one "What 1.0 means" asks in DESIGN.md: which
public symbols does no test ever use. Run it with `--api-coverage`, read
`api_coverage.md`, and the untouched list is the test plan.

Two mechanisms, because the surface is two kinds of thing.

Callables — methods, properties, constructors, module functions — are measured
by RUNNING them. Every one is replaced at session start with a wrapper that
records the call and forwards. This is exact: it separates `Buffer.update` from
`Image.update`, which is the distinction a text search cannot make. The first
attempt used `sys.setprofile`, which is wrong for pybind11: a bound pybind11
method is a plain `method` around a custom function record, not a `PyCFunction`,
so the interpreter emits no `c_call` event for it and only module-level
functions would have been seen.

Enum members and the key/button constants are read, never called, so nothing can
wrap them. They are matched by NAME against the identifiers in the test sources.
`test_stubs.py` is excluded from that scan: it carries about 110 API names as
string literals, and every one of them would count as a use.
"""

import functools
import pathlib
import re

import bazalt as bz

# The scan reads identifiers, so a name is "used" wherever it appears. Excluded
# because it names most of the API in string literals rather than using it.
_SCAN_EXCLUDES = {"test_stubs.py", "test_api_coverage.py", "api_coverage.py", "conftest.py"}

# Everything else that starts with an underscore is implementation.
_DUNDERS = ("__init__", "__enter__", "__exit__")

_REPORT = "api_coverage.md"
_BASELINE = pathlib.Path(__file__).parent / "api_coverage_baseline.txt"


def public_surface():
    """{"Owner.member": kind} for everything reachable from `bazalt.__all__`.

    Derived, never hand-written: a binding added to `src/bindings/` and to
    `__all__` appears here on the next run with no edit (the same rule that
    keeps `__init__.py`'s constant list a comprehension).
    """
    surface = {}
    for name in bz.__all__:
        if name.startswith("__"):
            continue
        obj = getattr(bz, name)

        if isinstance(obj, int) and not isinstance(obj, type):
            surface[name] = "constant"
            continue

        if not isinstance(obj, type):
            surface[name] = "function"
            continue

        if issubclass(obj, BaseException):
            surface[name] = "exception"
            continue

        # An enum contributes its members and nothing else. `vars()` on one also
        # holds pybind11's own `name`, `value` and `__members__`, which are not
        # bazalt's API and would add 44 symbols nobody can test.
        members = getattr(obj, "__members__", None)
        if members is not None:
            for member in members:
                surface[f"{name}.{member}"] = "enum member"
            continue

        for member, value in vars(obj).items():
            if member.startswith("_") and member not in _DUNDERS:
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


def names_in_tests(directory):
    """Every identifier that appears in the test sources."""
    identifier = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    found = set()
    for path in pathlib.Path(directory).rglob("*.py"):
        if path.name in _SCAN_EXCLUDES:
            continue
        found.update(identifier.findall(path.read_text(encoding="utf-8", errors="replace")))
    return found


def untouched(surface, used, identifiers):
    """The symbols no test called and no test named."""
    missing = []
    for key, kind in sorted(surface.items()):
        if kind in ("method", "property", "function"):
            if key not in used:
                missing.append((key, kind))
        else:
            # Read, not called: a mention is the only evidence available.
            if key.rpartition(".")[2] not in identifiers:
                missing.append((key, kind))
    return missing


def write_report(path, surface, missing):
    kinds = ["method", "property", "function", "enum member", "constant", "exception"]
    lines = [
        f"# API coverage — bazalt {bz.__version__}",
        "",
        "Written by `pytest --api-coverage`. A symbol counts as touched when a test",
        "calls it (methods, properties, functions) or names it (enum members,",
        "constants, exception classes). See `tests/api_coverage.py` for why the two",
        "halves are measured differently.",
        "",
        "| Kind | Symbols | Touched | Untouched |",
        "| --- | ---: | ---: | ---: |",
    ]
    missing_by_kind = {}
    for key, kind in missing:
        missing_by_kind.setdefault(kind, []).append(key)

    for kind in kinds:
        total = sum(1 for k in surface.values() if k == kind)
        if not total:
            continue
        gone = len(missing_by_kind.get(kind, ()))
        lines.append(f"| {kind} | {total} | {total - gone} | {gone} |")
    total = len(surface)
    lines.append(f"| **all** | **{total}** | **{total - len(missing)}** | **{len(missing)}** |")

    lines += ["", "## Untouched", ""]
    if not missing:
        lines.append("Nothing. Every public symbol is used by a test.")
    else:
        owners = {}
        for key, kind in missing:
            owner = key.rpartition(".")[0] or "module"
            owners.setdefault(owner, []).append((key, kind))
        for owner in sorted(owners):
            lines.append(f"### {owner}")
            lines.append("")
            for key, kind in owners[owner]:
                lines.append(f"- `{key}` ({kind})")
            lines.append("")

    pathlib.Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    body = sorted(key for key, _ in missing)
    _BASELINE.write_text("\n".join(header + body) + "\n", encoding="utf-8")
