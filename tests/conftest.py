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


@pytest.fixture(scope="session")
def _session_context():
    """One Context for the whole run.

    Only one Context can be alive per process (volk binds its global function
    pointers to a single device), so this is session-scoped rather than
    per-test. Creating and destroying one per test would also work, but only as
    long as no test leaks a reference to a buffer or pipeline — every resource
    holds the Context alive. Session scope removes that trap entirely.
    """
    messages = []
    logger = bz.Logger(min_severity=bz.Severity.INFO)
    logger.on_message(messages.append)
    # hot_reload=True and gpu_timing=True for the whole suite: both run under
    # every test, so validation-as-assert audits them continuously, and it's the
    # only way to exercise them given one Context per process. Hot reload for
    # files that never change never fires; the timestamp path is opt-in so the
    # gpu_time_ms test needs it on here (default apps pay nothing).
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
