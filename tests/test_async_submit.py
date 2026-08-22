"""ctx.submit(wait=False), Serial handles, after= and ctx.wait().

Deferred since 0.5. A headless submit blocked on vkQueueWaitIdle, which is right
when the next line reads the result and wrong when it does not: a compute
prototype that submits in a loop left the GPU idle between iterations, so the
loop ran at the speed of the round trip rather than of the work.

Since 0.28 every submit returns a Serial: ctx.wait(serial) waits for that one
submit, and submit(after=serial) orders one submit after another on the GPU
without blocking the CPU.
"""

import os
import pathlib

import pytest

import bazalt as bz

SHADER_DIR = pathlib.Path(__file__).parent / "shaders"


def add_one_pipeline(ctx):
    shader = ctx.compile_shader(str(SHADER_DIR / "add_one.comp"), bz.ShaderStage.COMPUTE)
    return ctx.compute_pipeline().shader(shader).storage_buffer(0).build()


def counting_setup(ctx):
    pipeline = add_one_pipeline(ctx)
    buf = ctx.create_buffer([0.0, 0.0, 0.0, 0.0], bz.BufferType.STORAGE,
                            bz.MemoryUsage.STATIC, bz.DataType.FLOAT)
    pool = ctx.create_descriptor_pool(max_sets=1, storage_buffers=1)
    dset = pool.allocate_set(pipeline, set=0)
    dset.set_buffer(0, buf)
    return pipeline, buf, dset


def counting_graph(ctx, pipeline, dset):
    g = ctx.graph()
    g.add_pass().bind_pipeline(pipeline).bind_descriptor_set(dset, pipeline, 0).dispatch(1)
    return g


def test_async_submits_all_land(ctx):
    """Every dispatch has to be there after ctx.wait(), whatever order the
    pacing let them run in."""
    pipeline, buf, dset = counting_setup(ctx)

    for _ in range(8):
        ctx.submit(counting_graph(ctx, pipeline, dset), wait=False)
    ctx.wait()

    assert buf.read("float32")[0] == 8.0


def test_reusing_one_graph_asynchronously(ctx):
    """The hazard the ring pacing exists for: with frames_in_flight slots, the
    N+2nd submit reuses the command buffer the Nth is still running. Without the
    per-slot wait this overwrites work in flight, which the validation layer
    reports — and the ctx fixture is the referee."""
    pipeline, buf, dset = counting_setup(ctx)

    g = counting_graph(ctx, pipeline, dset)
    for _ in range(10):
        ctx.submit(g, wait=False)
    ctx.wait()

    assert buf.read("float32")[0] == 10.0


def test_wait_is_idempotent(ctx):
    """Calling it with nothing outstanding is not an error, and calling it twice
    is not either — a loop that ends with a wait should not have to know."""
    ctx.wait()
    ctx.wait()


def test_blocking_submit_is_unchanged(ctx):
    """The default stays what it was: the result is readable on the next line."""
    pipeline, buf, dset = counting_setup(ctx)

    ctx.submit(counting_graph(ctx, pipeline, dset))

    assert buf.read("float32")[0] == 1.0


def test_async_then_blocking_submit(ctx):
    """Mixing them must not lose the asynchronous work: a blocking submit waits
    for the queue, which covers everything before it."""
    pipeline, buf, dset = counting_setup(ctx)

    g = counting_graph(ctx, pipeline, dset)
    ctx.submit(g, wait=False)
    ctx.submit(g, wait=False)
    ctx.submit(g)

    assert buf.read("float32")[0] == 3.0


# ── Serial handles (0.28) ─────────────────────────────────────────────────


def test_submit_returns_a_serial_and_wait_takes_it(ctx):
    """The return value IS the signal: every submit signals the timeline at its
    own serial, so a separate signal= parameter would be redundant. wait(serial)
    blocks for that one submit alone."""
    pipeline, buf, dset = counting_setup(ctx)

    s = ctx.submit(counting_graph(ctx, pipeline, dset), wait=False)
    assert isinstance(s, bz.Serial)
    ctx.wait(s)
    assert buf.read("float32")[0] == 1.0
    # Waiting again for an already-finished serial returns immediately.
    ctx.wait(s)


def test_after_orders_submits_on_the_gpu(ctx):
    """after= is the manual ordering between whole submits: the second starts
    after the first completes, with no CPU block. On one queue it accepts a
    single Serial or a list."""
    pipeline, buf, dset = counting_setup(ctx)

    s1 = ctx.submit(counting_graph(ctx, pipeline, dset), wait=False)
    s2 = ctx.submit(counting_graph(ctx, pipeline, dset), wait=False, after=s1)
    s3 = ctx.submit(counting_graph(ctx, pipeline, dset), wait=False, after=[s1, s2])
    ctx.wait(s3)

    assert buf.read("float32")[0] == 3.0


def test_a_serial_is_opaque(ctx):
    """No attributes, no ordering: a Serial from one queue must not be
    comparable arithmetic the day a second queue exists, so it is not today."""
    pipeline, buf, dset = counting_setup(ctx)
    s = ctx.submit(counting_graph(ctx, pipeline, dset))
    assert not hasattr(s, "value")
    with pytest.raises(TypeError):
        s < s  # noqa: B015 — the refusal is the assertion


# ── the compute queue (0.29) ──────────────────────────────────────────────


def test_async_compute_is_reported_honestly(ctx):
    """A queue family that does compute and not graphics is a fact about the
    hardware, so it is a Feature row like any other and nobody has to ask for
    it. On a device without one, Queue.COMPUTE still works — it runs on the
    graphics queue under its own timeline — and this is how a caller finds out
    which of the two they got."""
    answer = ctx.supports(bz.Feature.ASYNC_COMPUTE)
    assert isinstance(answer, bool)
    if os.environ.get("BAZALT_FORCE_SINGLE_QUEUE") != "1":
        # The Device reports the same hardware fact, and list_devices() never
        # reads the test knob.
        matching = [d for d in bz.list_devices() if d.name == ctx.device_name]
        assert matching and matching[0].supports(bz.Feature.ASYNC_COMPUTE) == answer


def test_requiring_async_compute_on_a_single_queue_device_raises(monkeypatch):
    """features= is a refusal, not a preference: a program that only makes sense
    with real overlap says so and gets an error rather than a silent fallback.

    The knob makes this deterministic everywhere — on a GPU that HAS a compute
    family the Context would otherwise be built successfully and prove nothing.
    """
    monkeypatch.setenv("BAZALT_FORCE_SINGLE_QUEUE", "1")
    with pytest.raises(bz.InitializationError, match="ASYNC_COMPUTE"):
        bz.Context(features=[bz.Feature.ASYNC_COMPUTE])

