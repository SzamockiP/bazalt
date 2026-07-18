"""The frame ring: owned by the Context, advanced at frame boundaries.

Headless submits used to sit on ring slot 0 forever — the renderer advanced
the index in end_frame, and the headless path has no renderer. These tests pin
the rotation and its user-facing contract: `update()` writes the slot that the
next submit reads.
"""

import pathlib

import numpy as np
import pytest

import bazalt as bz

SHADER_DIR = pathlib.Path(__file__).parent / "shaders"

CLEAR = [0.1, 0.2, 0.3, 1.0]


@pytest.fixture
def fullscreen_vert(ctx):
    return ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)


def test_frames_in_flight_is_validated_before_touching_vulkan():
    """ValueError, not InitializationError: a bad argument, and it must be
    raised before any Vulkan work happens (a live session Context exists, and
    only one Context may be alive per process)."""
    with pytest.raises(ValueError):
        bz.Context(frames_in_flight=0)
    with pytest.raises(ValueError):
        bz.Context(frames_in_flight=5)


def test_frames_in_flight_is_reported(ctx):
    assert ctx.frames_in_flight == 2


def test_headless_submits_rotate_the_ring(ctx, fullscreen_vert):
    """Two submits must consume two different DynamicBuffer slots.

    The buffer starts red in every slot; update() writes green into the
    CURRENT slot only. The first submit therefore renders green, and the
    second — with no update in between — renders the untouched red of the
    next slot. On 0.4.2 the headless path never advanced the ring, so the
    second submit rendered green again: this test pins the fix.
    """
    frag = ctx.compile_shader(str(SHADER_DIR / "ubo.frag"), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, 64, 64)
    pipeline = (ctx.pipeline_builder()
                .vertex_shader(fullscreen_vert)
                .fragment_shader(frag)
                .uniform_buffer(0, bz.ShaderStage.FRAGMENT, set=0)
                .build(target))

    ubuf = ctx.create_buffer([1.0, 0.0, 0.0, 1.0], bz.BufferType.UNIFORM,
                             bz.MemoryUsage.DYNAMIC)
    pool = ctx.create_descriptor_pool(max_sets=8, uniform_buffers=8)
    dset = pool.allocate_frame_set(pipeline, set=0)
    dset.set_buffer(0, ubuf)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(target, clear_color=CLEAR)
    cmd.bind_pipeline(pipeline)
    cmd.bind_descriptor_set(dset, pipeline, set=0)
    cmd.draw(3)
    cmd.end_rendering(target)

    ubuf.update([0.0, 1.0, 0.0, 1.0])
    ctx.submit(cmd)
    assert np.allclose(target.read_pixels()[32, 32, :3], [0, 255, 0], atol=2), \
        "the submit did not read the slot update() wrote"

    ctx.submit(cmd)
    assert np.allclose(target.read_pixels()[32, 32, :3], [255, 0, 0], atol=2), \
        "the second submit reused the first submit's ring slot"


def test_update_before_each_submit_always_wins(ctx, fullscreen_vert):
    """The idiomatic loop shape: update, submit, repeat.

    Whatever the ring does underneath, an update() made right before a submit
    must be what that submit renders.
    """
    frag = ctx.compile_shader(str(SHADER_DIR / "ubo.frag"), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, 64, 64)
    pipeline = (ctx.pipeline_builder()
                .vertex_shader(fullscreen_vert)
                .fragment_shader(frag)
                .uniform_buffer(0, bz.ShaderStage.FRAGMENT, set=0)
                .build(target))

    ubuf = ctx.create_buffer(16, bz.BufferType.UNIFORM, bz.MemoryUsage.DYNAMIC)
    pool = ctx.create_descriptor_pool(max_sets=8, uniform_buffers=8)
    dset = pool.allocate_frame_set(pipeline, set=0)
    dset.set_buffer(0, ubuf)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(target, clear_color=CLEAR)
    cmd.bind_pipeline(pipeline)
    cmd.bind_descriptor_set(dset, pipeline, set=0)
    cmd.draw(3)
    cmd.end_rendering(target)

    colours = [
        ([1.0, 0.0, 0.0, 1.0], [255, 0, 0]),
        ([0.0, 1.0, 0.0, 1.0], [0, 255, 0]),
        ([0.0, 0.0, 1.0, 1.0], [0, 0, 255]),
    ]
    for value, expected in colours:
        ubuf.update(value)
        ctx.submit(cmd)
        assert np.allclose(target.read_pixels()[32, 32, :3], expected, atol=2), value
