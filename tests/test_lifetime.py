"""Deferred destruction: dropping a resource on the CPU must be safe even if
an in-flight frame still references its handle.

Resource destructors enqueue their vkDestroy calls on the Context's deletion
queue instead of running them inline; the queue drains once the GPU provably
passed the frame that could last have referenced the handle. Validation-as-
assert in the ctx fixture is the real check here — a premature destroy shows
up as a validation error.
"""

import gc
import pathlib
import sys

import numpy as np
import pytest

import bazalt as bz

SHADER_DIR = pathlib.Path(__file__).parent / "shaders"

CLEAR = [0.1, 0.2, 0.3, 1.0]


def test_dropping_everything_between_submits_is_safe(ctx, triangle_shaders):
    """Create, render, drop, render something else, repeat."""
    vert, frag = triangle_shaders
    target = ctx.create_render_target(64, 64)

    for _ in range(3):
        vertices = [
            +0.0, -0.5, 0.0, 1.0, 0.0, 0.0,
            -0.5, +0.5, 0.0, 0.0, 1.0, 0.0,
            +0.5, +0.5, 0.0, 0.0, 0.0, 1.0,
        ]
        vbuf = ctx.create_buffer(vertices, bz.BufferType.VERTEX,
                                 bz.MemoryUsage.STATIC, bz.DataType.FLOAT)
        ibuf = ctx.create_buffer([0, 1, 2], bz.BufferType.INDEX,
                                 bz.MemoryUsage.STATIC, bz.DataType.UINT32)
        pipeline = (ctx.graphics_pipeline()
                    .vertex_shader(vert)
                    .fragment_shader(frag)
                    .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                    .build(target))

        cmd = ctx.create_command_buffer()
        cmd.begin()
        cmd.begin_rendering(target, clear_color=CLEAR)
        cmd.bind_pipeline(pipeline)
        cmd.bind_vertex_buffer(vbuf)
        cmd.bind_index_buffer(ibuf)
        cmd.draw_indexed(3)
        cmd.end_rendering(target)
        ctx.submit(cmd)

        centre = target.color[0].read()[32, 32, :3]
        assert not np.allclose(centre, np.array(CLEAR[:3]) * 255, atol=2)

        # Drop every resource this iteration created. Their handles go through
        # the deletion queue; the next iteration submits over them.
        del cmd, pipeline, vbuf, ibuf
        gc.collect()


def test_rerecording_a_command_buffer_drops_old_resources_safely(ctx, triangle_shaders, triangle_buffers):
    """cmd.begin() clears the recorded lambdas — the shared_ptrs they held were
    the only thing keeping dropped-from-Python resources alive. The frees this
    triggers must be deferred, not inline."""
    vert, frag = triangle_shaders
    vbuf, ibuf = triangle_buffers
    target = ctx.create_render_target(64, 64)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                .build(target))

    cmd = ctx.create_command_buffer()

    def record():
        cmd.begin()
        cmd.begin_rendering(target, clear_color=CLEAR)
        cmd.bind_pipeline(pipeline)
        cmd.bind_vertex_buffer(vbuf)
        cmd.bind_index_buffer(ibuf)
        cmd.draw_indexed(3)
        cmd.end_rendering(target)

    record()
    ctx.submit(cmd)
    record()  # clears the previous recording while nothing else is pending
    ctx.submit(cmd)
    assert target.color[0].read() is not None


def test_a_renderer_keeps_its_window_alive(ctx):
    """A SwapchainRenderer reads the Window on every present, and the Python
    reference that says so is the caller's alone unless the binding ties them.

    Window.get_surface_provider() hands the renderer three lambdas that capture
    the raw GLFWwindow* and a pointer to the Window's own resize flag, and the
    renderer keeps them for its whole life. So `del window` used to leave both
    pointers dangling and present() read freed memory."""
    if ctx.headless:
        pytest.skip("no swapchain support (headless Context)")
    try:
        window = bz.Window(64, 64, "bazalt window lifetime")
    except bz.WindowError:
        pytest.skip("no display available")

    renderer = None
    try:
        before = sys.getrefcount(window)
        renderer = ctx.create_renderer(window)
        assert sys.getrefcount(window) == before + 1, "the renderer holds no reference to the window"

        vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
        frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
        pipeline = ctx.graphics_pipeline().vertex_shader(vert).fragment_shader(frag).build(renderer)

        cmd = ctx.create_command_buffer()
        cmd.begin()
        cmd.begin_rendering(renderer, clear_color=CLEAR)
        cmd.bind_pipeline(pipeline)
        cmd.draw(3)
        cmd.end_rendering(renderer)

        # The caller drops its only reference. Everything after this reads the
        # Window through the renderer's captured pointers.
        window = None
        gc.collect()

        for _ in range(ctx.frames_in_flight + 2):
            bz.poll_events()
            ctx.begin_frame()
            if renderer.acquire():
                renderer.present(cmd)
    finally:
        renderer = None
        window = None
