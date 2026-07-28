"""ctx.submit(wait=False) and ctx.wait().

Deferred since 0.5. A headless submit blocked on vkQueueWaitIdle, which is right
when the next line reads the result and wrong when it does not: a compute
prototype that submits in a loop left the GPU idle between iterations, so the
loop ran at the speed of the round trip rather than of the work.
"""

import pathlib

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


def test_async_submits_all_land(ctx):
    """Every dispatch has to be there after ctx.wait(), whatever order the
    pacing let them run in."""
    pipeline, buf, dset = counting_setup(ctx)

    for _ in range(8):
        cmd = ctx.create_command_buffer()
        cmd.begin().bind_pipeline(pipeline).bind_descriptor_set(dset, pipeline, 0).dispatch(1)
        ctx.submit(cmd, wait=False)
    ctx.wait()

    assert buf.read("float32")[0] == 8.0


def test_reusing_one_command_buffer_asynchronously(ctx):
    """The hazard the ring pacing exists for: with frames_in_flight slots, the
    N+2nd submit reuses the command buffer the Nth is still running. Without the
    per-slot wait this overwrites work in flight, which the validation layer
    reports — and the ctx fixture is the referee."""
    pipeline, buf, dset = counting_setup(ctx)

    cmd = ctx.create_command_buffer()
    cmd.begin().bind_pipeline(pipeline).bind_descriptor_set(dset, pipeline, 0).dispatch(1)
    for _ in range(10):
        ctx.submit(cmd, wait=False)
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

    cmd = ctx.create_command_buffer()
    cmd.begin().bind_pipeline(pipeline).bind_descriptor_set(dset, pipeline, 0).dispatch(1)
    ctx.submit(cmd)

    assert buf.read("float32")[0] == 1.0


def test_async_then_blocking_submit(ctx):
    """Mixing them must not lose the asynchronous work: a blocking submit waits
    for the queue, which covers everything before it."""
    pipeline, buf, dset = counting_setup(ctx)

    cmd = ctx.create_command_buffer()
    cmd.begin().bind_pipeline(pipeline).bind_descriptor_set(dset, pipeline, 0).dispatch(1)
    ctx.submit(cmd, wait=False)
    ctx.submit(cmd, wait=False)
    ctx.submit(cmd)

    assert buf.read("float32")[0] == 3.0
