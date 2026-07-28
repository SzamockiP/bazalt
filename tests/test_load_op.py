"""0.16: clear_color=None preserves the attachment instead of clearing it.

The load-op used to be CLEAR unconditionally, so a target could hold the result
of exactly one pass. Preserving is what puts a second pass on the same image:
opaque then transparent, or a UI over a scene.

Everything here runs headless, so CI covers it. The `ctx` fixture is the second
referee: a preserved pass that entered from the wrong layout is a validation
error, not a wrong pixel.
"""

import pathlib
import struct

import numpy as np
import pytest

import bazalt as bz

SHADER_DIR = pathlib.Path(__file__).parent / "shaders"

RED = [1.0, 0.0, 0.0, 1.0]


@pytest.fixture
def push_pipeline(ctx):
    """A fullscreen draw whose colour comes from a push constant, built for
    `target` at either z = 0.0 (near) or z = 0.9 (far)."""
    frag = ctx.compile_shader(str(SHADER_DIR / "push.frag"), bz.ShaderStage.FRAGMENT)

    def make(target, *, far=False, **state):
        name = "fullscreen_far.vert" if far else "fullscreen.vert"
        vert = ctx.compile_shader(str(SHADER_DIR / name), bz.ShaderStage.VERTEX)
        builder = (ctx.graphics_pipeline()
                   .vertex_shader(vert)
                   .fragment_shader(frag)
                   .push_constant(16, bz.ShaderStage.FRAGMENT))
        if state:
            builder = builder.depth_test(state.pop("depth_test", True), **state)
        return builder.build(target)

    return make


def colour(*rgba):
    return struct.pack("4f", *rgba)


def test_second_pass_keeps_what_the_first_one_drew(ctx, push_pipeline):
    """Pass 1 clears red. Pass 2 preserves and repaints only the left half.

    The scissor is what makes the assertion two-sided: the right half can only
    still be red if the load really loaded.
    """
    target = bz.RenderTarget(ctx, 64, 64)
    pipeline = push_pipeline(target)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(target, clear_color=RED)
    cmd.end_rendering(target)

    cmd.begin_rendering(target, clear_color=None)
    cmd.set_scissor(0, 0, 32, 64)
    cmd.bind_pipeline(pipeline)
    cmd.push_constants(pipeline, 0, colour(0.0, 1.0, 0.0, 1.0))
    cmd.draw(3)
    cmd.end_rendering(target)
    ctx.submit(cmd)

    pixels = target.color[0].read()
    assert np.allclose(pixels[32, 8, :3], [0, 255, 0], atol=2), pixels[32, 8]
    assert np.allclose(pixels[32, 56, :3], [255, 0, 0], atol=2), pixels[32, 56]


def test_preserve_survives_a_replay(ctx, push_pipeline):
    """Two submits of the same recording must land on the same pixels.

    A recording is replayed every frame, so a preserved pass reads whatever the
    previous replay left. Pass 1 still clears, which is what keeps the sequence
    idempotent — and this is the case that would drift if it did not.
    """
    target = bz.RenderTarget(ctx, 64, 64)
    pipeline = push_pipeline(target)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(target, clear_color=RED)
    cmd.end_rendering(target)
    cmd.begin_rendering(target, clear_color=None)
    cmd.set_scissor(0, 0, 32, 64)
    cmd.bind_pipeline(pipeline)
    cmd.push_constants(pipeline, 0, colour(0.0, 1.0, 0.0, 1.0))
    cmd.draw(3)
    cmd.end_rendering(target)

    ctx.submit(cmd)
    first = target.color[0].read().copy()
    ctx.submit(cmd)
    second = target.color[0].read()

    assert np.array_equal(first, second)


def test_depth_survives_into_the_second_pass(ctx, push_pipeline):
    """Pass 1 writes depth 0.0 over the whole target. Pass 2 preserves and draws
    at 0.9, which LESS_OR_EQUAL must reject.

    This is the functional half of the store-op change. It is not a proof: a
    DONT_CARE depth is *undefined* between passes, and undefined is free to
    happen to hold the old value on any given driver.
    """
    target = bz.RenderTarget(ctx, 64, 64, depth=bz.Format.D32F)
    near = push_pipeline(target, depth_test=True)
    far = push_pipeline(target, far=True, depth_test=True)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0])
    cmd.bind_pipeline(near)
    cmd.push_constants(near, 0, colour(0.0, 1.0, 0.0, 1.0))
    cmd.draw(3)
    cmd.end_rendering(target)

    cmd.begin_rendering(target, clear_color=None)
    cmd.bind_pipeline(far)
    cmd.push_constants(far, 0, colour(0.0, 0.0, 1.0, 1.0))
    cmd.draw(3)
    cmd.end_rendering(target)
    ctx.submit(cmd)

    pixels = target.color[0].read()
    assert np.allclose(pixels[32, 32, :3], [0, 255, 0], atol=2), \
        f"the far draw was not rejected, so the depth did not survive: {pixels[32, 32]}"


def test_an_empty_clear_list_still_means_black(ctx):
    """None and [] are different answers: preserve versus clear to black.

    They used to collapse into one empty vector, so this pins the distinction.
    """
    target = bz.RenderTarget(ctx, 64, 64)
    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(target, clear_color=[])
    cmd.end_rendering(target)
    ctx.submit(cmd)

    assert np.allclose(target.color[0].read()[32, 32, :3], [0, 0, 0], atol=1)


def test_multisampled_preserve_is_rejected(ctx):
    """MSAA has nothing to preserve: the multisampled image is discarded each
    pass and the result lives in the resolve image."""
    if ctx.max_samples() < 2:
        pytest.skip("GPU reports no MSAA support (max_samples == 1)")

    target = bz.RenderTarget(ctx, 64, 64, samples=min(4, ctx.max_samples()))
    cmd = ctx.create_command_buffer()
    cmd.begin()

    with pytest.raises(bz.ResourceError, match="nothing to preserve"):
        cmd.begin_rendering(target, clear_color=None)

    # The `with` sugar goes through the same guard, and must reject at the call
    # rather than inside __enter__.
    with pytest.raises(bz.ResourceError, match="nothing to preserve"):
        cmd.rendering(target, clear_color=None)


def test_a_multisampled_target_still_clears(ctx):
    """The guard is about preserve only — MSAA plus a clear is untouched."""
    if ctx.max_samples() < 2:
        pytest.skip("GPU reports no MSAA support (max_samples == 1)")

    target = bz.RenderTarget(ctx, 64, 64, samples=min(4, ctx.max_samples()))
    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(target, clear_color=RED)
    cmd.end_rendering(target)
    ctx.submit(cmd)

    assert np.allclose(target.color[0].read()[32, 32, :3], [255, 0, 0], atol=2)
