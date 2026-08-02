"""Shared fixtures.

The point of this file is `ctx`: every test that touches the GPU fails if the
validation layers said anything. Before headless rendering existed there was no
way to run bazalt without a window, so there were no tests at all and an entire
class of bug was invisible.
"""

import os
import pathlib

# A fast poll keeps the hot-reload tests quick; set before bazalt is imported so
# the watcher (created with the session Context below) reads it. A test/CI knob,
# not public API — the Python surface stays at one kwarg. setdefault lets a real
# CI override it.
os.environ.setdefault("BAZALT_HOT_RELOAD_POLL_MS", "20")

import pytest

import bazalt as bz

SHADER_DIR = pathlib.Path(__file__).parent / "shaders"


# ── API coverage (opt in with --api-coverage) ────────────────────────────
# Off by default: it replaces every public callable with a recording wrapper,
# and the suite has no business running against a patched module when the
# question is not being asked. See tests/api_coverage.py.

_recorder = None


def pytest_addoption(parser):
    parser.addoption(
        "--api-coverage",
        action="store_true",
        help="report which public symbols the suite touches, into api_coverage.md")


def pytest_configure(config):
    global _recorder
    if not config.getoption("--api-coverage"):
        return
    import api_coverage

    _recorder = api_coverage.Recorder()
    _recorder.install(api_coverage.public_surface())


def pytest_sessionfinish(session, exitstatus):
    """Write the report, and fail on a symbol that is untouched and unexpected.

    The measurement is only meaningful for a whole, passing run: a failed test
    stops calling things, `-k` never reaches most of the API, and a SKIPPED test
    is a symbol nobody called on this machine rather than a symbol nobody
    tested. All three write the report and skip the gate, so the number is never
    read as a regression when it is really a partial run.

    The skip rule is what lets CI ask for the report at all: lavapipe has no
    display and not every feature, so it always skips something, while a
    developer GPU runs the suite with no skips and gets the gate.
    """
    if _recorder is None:
        return
    import api_coverage

    surface = api_coverage.public_surface()
    missing = api_coverage.untouched(
        surface, _recorder.used, api_coverage.names_in_tests(pathlib.Path(__file__).parent))
    api_coverage.write_report(api_coverage._REPORT, surface, missing)

    if os.environ.get("BAZALT_WRITE_API_BASELINE") == "1":
        api_coverage.write_baseline(missing)
        return

    # A named file or node id is a subset by definition; the directory CI passes
    # is not. -k and -m are subsets whatever they select.
    selected_files = any(
        argument.endswith(".py") or "::" in argument
        for argument in session.config.invocation_params.args)
    reporter = session.config.pluginmanager.getplugin("terminalreporter")
    skipped = len(reporter.stats.get("skipped", ())) if reporter else 0
    if (exitstatus != 0 or skipped or selected_files
            or session.config.option.keyword or session.config.option.markexpr):
        return

    new = sorted(key for key, _ in missing if key not in api_coverage.read_baseline())
    if new:
        session.exitstatus = 1
        print("\nAPI coverage: these public symbols are untouched by any test and are not in")
        print("tests/api_coverage_baseline.txt — add a test, or accept them into the baseline")
        print("with BAZALT_WRITE_API_BASELINE=1:")
        for key in new:
            print(f"  {key}")


@pytest.fixture(scope="session")
def _session_context():
    """One Context for the whole run.

    Session-scoped because every resource holds its Context alive: a per-test
    Context works only as long as no test leaks a reference to a buffer or a
    pipeline, and session scope removes that trap entirely. (This used to be
    forced — one Context per process — until 0.15 gave each one its own dispatch
    table; tests that want a second Context now just ask for `extra_context`.)
    """
    messages = []
    logger = bz.Logger(min_severity=bz.Severity.INFO)
    logger.on_message(messages.append)
    # hot_reload=True and gpu_timing=True for the whole suite: both run under
    # every test, so validation-as-assert audits them continuously. Hot reload
    # for files that never change never fires; the timestamp path is opt-in so
    # the gpu_time_ms test needs it on here (default apps pay nothing).
    context = bz.Context(logger, validation="auto", hot_reload=True, gpu_timing=True)
    yield context, logger, messages


@pytest.fixture
def ctx(_session_context):
    """The Context, plus an assertion that the validation layers stayed quiet.

    Only Source.VALIDATION counts. Tests that deliberately provoke a bad shader
    or a missing file also log an error, and those are the test working, not the
    library misbehaving — the two are distinguishable only because LogMessage
    carries a structured source instead of a formatted string.

    Severity ERROR and above, not WARNING: the loader emits warnings about
    unrelated third-party layers (OBS, Epic overlay) that have nothing to do with
    bazalt, and failing on those would make the suite depend on what else the
    developer happens to have installed.
    """
    context, logger, messages = _session_context
    start = len(messages)
    yield context

    logger.flush()
    errors = [
        m for m in messages[start:]
        if m.source == bz.Source.VALIDATION and m.severity >= bz.Severity.ERROR
    ]
    assert not errors, "validation errors:\n" + "\n".join(f"  {m.text}" for m in errors)


@pytest.fixture
def extra_context():
    """A factory for Contexts beyond the session one, with the same referee.

    Multi-context (0.15) and sync validation both need a Context of their own,
    and a Context nobody audits is weaker than the rest of the suite — so this
    wires up the same validation-as-assert the `ctx` fixture applies, and checks
    every Context it handed out when the test ends.

        a = extra_context(validation="on")

    Each gets its own Logger: the layers are per-instance, so mixing their
    output would make "which Context complained" unanswerable.
    """
    created = []

    def make(**kwargs):
        msgs = []
        logger = bz.Logger(min_severity=bz.Severity.INFO)
        logger.on_message(msgs.append)
        kwargs.setdefault("validation", "on")
        context = bz.Context(logger, **kwargs)
        created.append((logger, msgs))
        return context

    yield make

    for logger, msgs in created:
        logger.flush()
        errors = [
            m for m in msgs
            if m.source == bz.Source.VALIDATION and m.severity >= bz.Severity.ERROR
        ]
        assert not errors, "validation errors:\n" + "\n".join(f"  {m.text}" for m in errors)


@pytest.fixture
def messages(_session_context):
    """Messages logged during this test, newest run only."""
    _, logger, all_messages = _session_context
    start = len(all_messages)

    class View:
        def __call__(self):
            logger.flush()
            return all_messages[start:]

    return View()


@pytest.fixture
def triangle_shaders(ctx):
    vert = ctx.compile_shader(str(SHADER_DIR / "triangle.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "triangle.frag"), bz.ShaderStage.FRAGMENT)
    return vert, frag


@pytest.fixture
def triangle_buffers(ctx):
    """A triangle covering the centre of the viewport: red top, green left, blue right."""
    vertices = [
        +0.0, -0.5, 0.0, 1.0, 0.0, 0.0,
        -0.5, +0.5, 0.0, 0.0, 1.0, 0.0,
        +0.5, +0.5, 0.0, 0.0, 0.0, 1.0,
    ]
    vbuf = ctx.create_buffer(vertices, bz.BufferType.VERTEX, bz.MemoryUsage.STATIC,
                             bz.DataType.FLOAT)
    ibuf = ctx.create_buffer([0, 1, 2], bz.BufferType.INDEX, bz.MemoryUsage.STATIC,
                             bz.DataType.UINT32)
    return vbuf, ibuf


PRINTF_PROBE = """
#version 450
#extension GL_EXT_debug_printf : enable
layout(local_size_x = 1) in;
void main()
{
    debugPrintfEXT("bazalt probe %d", 1);
}
"""


@pytest.fixture(scope="session")
def printf_compiles():
    """Whether this driver's shader compiler implements debugPrintfEXT.

    Nothing can be asked in advance. MoltenVK advertises
    VK_KHR_shader_non_semantic_info, accepts the SPIR-V, and then fails inside
    the Metal compiler with "use of undeclared identifier 'debugPrintfEXT'", so
    the probe compiles one and looks.

    On a Context of its own, never the `extra_context` factory: the failure IS a
    validation error, and every Context that factory hands out is watched by the
    referee that fails a test for exactly that.
    """
    try:
        context = bz.Context(bz.Logger(), validation="on", shader_printf=True)
    except bz.BazaltError:
        return False
    if not context.shader_printf:
        return False
    try:
        shader = context.compile_shader(source=PRINTF_PROBE, stage=bz.ShaderStage.COMPUTE)
        context.compute_pipeline().shader(shader).build()
        return True
    except bz.BazaltError:
        return False
