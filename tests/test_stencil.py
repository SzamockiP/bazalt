"""0.17: the stencil buffer.

stencilTestEnable was hard-coded to FALSE and no format carried a stencil
aspect, so masking a pass by "where something else was drawn" had no spelling
at all. The referee for the aspect masks and the attachment layouts is the
validation-as-assert fixture, not the pixels.

Headless, so CI covers it.
"""

import pathlib
import struct

import numpy as np
import pytest

import bazalt as bz

SHADER_DIR = pathlib.Path(__file__).parent / "shaders"


@pytest.fixture
def solid(ctx):
    """A fullscreen pipeline painting one push-constant colour, with the
    stencil state the caller asks for."""
    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "push.frag"), bz.ShaderStage.FRAGMENT)

    def make(target, **stencil):
        builder = (ctx.graphics_pipeline()
                   .vertex_shader(vert)
                   .fragment_shader(frag)
                   .push_constant(16, bz.ShaderStage.FRAGMENT))
        if stencil:
            builder = builder.stencil_test(True, **stencil)
        return builder.build(target)

    return make


def draw(cmd, pipeline, rgba):
    cmd.bind_pipeline(pipeline)
    cmd.push_constants(pipeline, 0, struct.pack("4f", *rgba))
    cmd.draw(3)


def test_depth_stencil_target_builds_and_renders(ctx):
    """The plain path first: a target with a stencil aspect still renders."""
    target = bz.RenderTarget(ctx, 32, 32, color=bz.Format.RGBA8, depth=bz.Format.DEPTH_STENCIL)
    assert target.depth.format == bz.Format.DEPTH_STENCIL

    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "push.frag"), bz.ShaderStage.FRAGMENT)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .push_constant(16, bz.ShaderStage.FRAGMENT)
                .depth_test(True)
                .build(target))

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0], clear_stencil=0) as c:
        draw(c, pipeline, [0.0, 1.0, 0.0, 1.0])
    ctx.submit(cmd)

    assert np.allclose(target.color[0].read()[16, 16][:3], [0, 255, 0], atol=2)


def test_a_mask_written_by_one_pass_gates_the_next(ctx, solid):
    """Two passes on one target: the first writes stencil 1 over the whole
    image, the second paints only where the stencil is NOT 1.

    Two-sided: the same second pass with compare=EQUAL paints everywhere the
    first one marked, so a stencil test that did nothing would fail one of the
    two halves.
    """
    target = bz.RenderTarget(ctx, 32, 32, color=bz.Format.RGBA8, depth=bz.Format.DEPTH_STENCIL)
    mark = solid(target, compare=bz.CompareOp.ALWAYS, ref=1, pass_op=bz.StencilOp.REPLACE)
    outside = solid(target, compare=bz.CompareOp.NOT_EQUAL, ref=1)
    inside = solid(target, compare=bz.CompareOp.EQUAL, ref=1)

    def run(second):
        cmd = ctx.create_command_buffer()
        cmd.begin()
        with cmd.rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0], clear_stencil=0) as c:
            draw(c, mark, [1.0, 0.0, 0.0, 1.0])
        # The second pass preserves, so it draws over what the first one left.
        with cmd.rendering(target, clear_color=None) as c:
            draw(c, second, [0.0, 0.0, 1.0, 1.0])
        ctx.submit(cmd)
        return target.color[0].read()[16, 16][:3]

    assert np.allclose(run(outside), [255, 0, 0], atol=2), "NOT_EQUAL painted a marked pixel"
    assert np.allclose(run(inside), [0, 0, 255], atol=2), "EQUAL skipped a marked pixel"


def test_write_mask_zero_writes_nothing(ctx, solid):
    """write_mask=0 keeps the test and drops the write, so the second pass sees
    the cleared value and paints."""
    target = bz.RenderTarget(ctx, 32, 32, color=bz.Format.RGBA8, depth=bz.Format.DEPTH_STENCIL)
    mark = solid(target, compare=bz.CompareOp.ALWAYS, ref=1,
                 pass_op=bz.StencilOp.REPLACE, write_mask=0)
    outside = solid(target, compare=bz.CompareOp.NOT_EQUAL, ref=1)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0], clear_stencil=0) as c:
        draw(c, mark, [1.0, 0.0, 0.0, 1.0])
    with cmd.rendering(target, clear_color=None) as c:
        draw(c, outside, [0.0, 0.0, 1.0, 1.0])
    ctx.submit(cmd)

    assert np.allclose(target.color[0].read()[16, 16][:3], [0, 0, 255], atol=2)


def test_clear_stencil_sets_the_starting_value(ctx, solid):
    """clear_stencil=1 makes the pass start where the mask already passes."""
    target = bz.RenderTarget(ctx, 32, 32, color=bz.Format.RGBA8, depth=bz.Format.DEPTH_STENCIL)
    equal_one = solid(target, compare=bz.CompareOp.EQUAL, ref=1)

    def run(clear_stencil):
        cmd = ctx.create_command_buffer()
        cmd.begin()
        with cmd.rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0],
                           clear_stencil=clear_stencil) as c:
            draw(c, equal_one, [0.0, 1.0, 0.0, 1.0])
        ctx.submit(cmd)
        return target.color[0].read()[16, 16][:3]

    assert np.allclose(run(1), [0, 255, 0], atol=2)
    assert np.all(run(0) == 0), "the test passed against a stencil cleared to 0"


def test_stencil_test_without_a_stencil_attachment_is_refused(ctx, solid):
    """Asking for the test on a plain depth target names the fix instead of
    drawing nothing."""
    target = bz.RenderTarget(ctx, 32, 32, color=bz.Format.RGBA8, depth=bz.Format.D32F)
    with pytest.raises(bz.ShaderError, match="DEPTH_STENCIL"):
        solid(target, compare=bz.CompareOp.EQUAL, ref=1)


def test_a_window_can_carry_a_stencil(ctx):
    """`SwapchainRenderer(..., stencil=True)` is what makes a masked pass
    reachable on screen rather than only offscreen.

    Needs a display, so CI skips it — but the bug it pins is real and was found
    by running example 23: a depth buffer WITH a stencil aspect may not be
    transitioned to DEPTH_ATTACHMENT_OPTIMAL, which is the layout every target
    used to name unconditionally.
    """
    if ctx.headless:
        pytest.skip("no swapchain support (headless Context)")
    try:
        window = bz.Window(64, 64, "bazalt stencil")
    except bz.WindowError:
        pytest.skip("no display available")
    try:
        renderer = bz.SwapchainRenderer(window, ctx, stencil=True)
        vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
        frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
        pipeline = (ctx.graphics_pipeline()
                    .vertex_shader(vert)
                    .fragment_shader(frag)
                    .depth_test(True)
                    .stencil_test(True, compare=bz.CompareOp.ALWAYS, ref=1,
                                  pass_op=bz.StencilOp.REPLACE)
                    .build(renderer))

        cmd = ctx.create_command_buffer()
        cmd.begin()
        with cmd.rendering(renderer, clear_color=[0.0, 0.0, 0.0, 1.0], clear_stencil=0) as c:
            c.bind_pipeline(pipeline).draw(3)

        bz.poll_events()
        ctx.begin_frame()
        if renderer.acquire():
            renderer.present(cmd)
        ctx.wait()
    finally:
        del renderer
        del window


def test_a_combined_format_is_not_readable(ctx):
    """One texel is depth AND stencil, so there is no array shape for it."""
    target = bz.RenderTarget(ctx, 32, 32, color=bz.Format.RGBA8, depth=bz.Format.DEPTH_STENCIL)
    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0]):
        pass
    ctx.submit(cmd)

    with pytest.raises(bz.ResourceError, match="DEPTH_STENCIL"):
        target.depth.read()
