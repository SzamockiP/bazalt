"""Queue.TRANSFER: a pass of copies on the device's transfer queue.

The cross-queue edges are the point. A copy on TRANSFER that a graphics pass
samples becomes a timeline wait, and the same program runs where the transfer
runtime aliases the graphics queue — BAZALT_FORCE_SINGLE_QUEUE=1 is how the
alias half runs on a machine that has the real family.
"""

import pathlib

import numpy as np
import pytest

import bazalt as bz

SHADER_DIR = pathlib.Path(__file__).parent / "shaders"


def run_transfer_copy_then_sample_case(monkeypatch, forced):
    """A TRANSFER pass copies a red tile into an image; a graphics pass
    samples it into a target. Sync validation plus the pixel are the referees.
    """
    if forced:
        monkeypatch.setenv("BAZALT_FORCE_SINGLE_QUEUE", "1")
    hazards = []
    log = bz.Logger(min_severity=bz.Severity.INFO)

    @log.on_message
    def _(msg):
        if msg.source == bz.Source.VALIDATION and "hazard" in msg.text.lower():
            hazards.append(msg.text)

    context = bz.Context(log, validation="sync")
    vert = context.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = context.compile_shader(str(SHADER_DIR / "textured.frag"), bz.ShaderStage.FRAGMENT)
    target = context.create_render_target(32, 32)
    pipe = (context.graphics_pipeline().vertex_shader(vert).fragment_shader(frag)
            .texture(0, bz.ShaderStage.FRAGMENT).build(target))

    image = context.create_image(8, 8, bz.Format.RGBA8)
    tile = np.zeros((8 * 8, 4), np.uint8)
    tile[:] = (255, 0, 0, 255)
    src = context.create_buffer(tile.ravel(), bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    dset = context.create_descriptor_pool().allocate_set(pipe)
    dset.set_image(0, image)

    g = context.graph()
    g.add_pass(name="stream", queue=bz.Queue.TRANSFER).copy_buffer_to_image(src, image)
    with g.add_pass(target, name="sample") as p:
        p.bind_pipeline(pipe).bind_descriptor_set(dset).draw(3)
    context.submit(g)

    pixels = target.color[0].read()
    log.flush()
    return hazards, pixels


@pytest.mark.parametrize("forced", [False, True], ids=["real-queues", "forced-alias"])
def test_a_transfer_copy_is_sampled_by_a_later_graphics_pass(ctx, monkeypatch, forced):
    hazards, pixels = run_transfer_copy_then_sample_case(monkeypatch, forced)
    assert hazards == []
    assert tuple(pixels[16, 16][:3]) == (255, 0, 0)


def test_a_transfer_readback_after_a_compute_write(ctx):
    """The buffer path across the third queue: a compute pass writes, a
    TRANSFER pass copies the result out, and the fold turns the edge into a
    timeline wait."""
    comp = ctx.compile_shader(str(SHADER_DIR / "double.comp"), bz.ShaderStage.COMPUTE)
    pipeline = ctx.compute_pipeline().shader(comp).storage_buffer(0).build()
    buf = ctx.create_buffer(np.arange(64, dtype=np.float32),
                            bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    out = ctx.create_buffer(64 * 4, bz.BufferUsage.STORAGE, bz.MemoryUsage.STATIC)
    dset = ctx.create_descriptor_pool().allocate_set(pipeline)
    dset.set_buffer(0, buf)

    g = ctx.graph()
    (g.add_pass(name="double", queue=bz.Queue.COMPUTE)
        .bind_pipeline(pipeline).bind_descriptor_set(dset).dispatch(1))
    g.add_pass(name="readback", queue=bz.Queue.TRANSFER).copy_buffer(buf, out)
    ctx.submit(g)

    assert np.allclose(out.read(np.float32), np.arange(64, dtype=np.float32) * 2)
