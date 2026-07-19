"""Chained recording and the `with cmd.rendering(...)` scope.

Chaining is the same API, not a second one: every recording method returns
the command buffer itself, so `cmd.a().b()` and `cmd.a(); cmd.b()` record
identically. Pinned by rendering the same scene both ways and comparing
pixels exactly.
"""

import numpy as np
import pytest

import bazalt as bz

CLEAR = [0.1, 0.2, 0.3, 1.0]


def test_every_recording_method_returns_the_same_object(ctx, triangle_shaders, triangle_buffers):
    vert, frag = triangle_shaders
    vbuf, ibuf = triangle_buffers
    target = bz.RenderTarget(ctx, 16, 16)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                .build(target))

    cmd = ctx.create_command_buffer()
    assert cmd.begin() is cmd
    assert cmd.begin_rendering(target) is cmd
    assert cmd.bind_pipeline(pipeline) is cmd
    assert cmd.bind_vertex_buffer(vbuf) is cmd
    assert cmd.bind_index_buffer(ibuf) is cmd
    assert cmd.set_viewport(0, 0, 16, 16) is cmd
    assert cmd.set_scissor(0, 0, 16, 16) is cmd
    assert cmd.draw_indexed(3) is cmd
    assert cmd.end_rendering(target) is cmd


def test_chained_and_statement_styles_render_identically(ctx, triangle_shaders, triangle_buffers):
    vert, frag = triangle_shaders
    vbuf, ibuf = triangle_buffers

    def make_pipeline(target):
        return (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                .build(target))

    statement_target = bz.RenderTarget(ctx, 64, 64)
    pipeline = make_pipeline(statement_target)
    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(statement_target, clear_color=CLEAR)
    cmd.bind_pipeline(pipeline)
    cmd.bind_vertex_buffer(vbuf)
    cmd.bind_index_buffer(ibuf)
    cmd.draw_indexed(3)
    cmd.end_rendering(statement_target)
    ctx.submit(cmd)

    chained_target = bz.RenderTarget(ctx, 64, 64)
    pipeline2 = make_pipeline(chained_target)
    cmd2 = ctx.create_command_buffer()
    (cmd2.begin()
         .begin_rendering(chained_target, clear_color=CLEAR)
         .bind_pipeline(pipeline2)
         .bind_vertex_buffer(vbuf)
         .bind_index_buffer(ibuf)
         .draw_indexed(3)
         .end_rendering(chained_target))
    ctx.submit(cmd2)

    np.testing.assert_array_equal(chained_target.read_pixels(),
                                  statement_target.read_pixels())


def test_rendering_scope_records_the_pair(ctx, triangle_shaders, triangle_buffers):
    """`with cmd.rendering(...)` is begin_rendering + end_rendering, exactly."""
    vert, frag = triangle_shaders
    vbuf, ibuf = triangle_buffers
    target = bz.RenderTarget(ctx, 64, 64)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                .build(target))

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(target, clear_color=CLEAR) as c:
        assert c is cmd
        c.bind_pipeline(pipeline).bind_vertex_buffer(vbuf).bind_index_buffer(ibuf).draw_indexed(3)
    ctx.submit(cmd)

    pixels = target.read_pixels()
    assert not np.allclose(pixels[32, 32, :3], np.array(CLEAR[:3]) * 255, atol=2)


def test_rendering_scope_closes_on_exception(ctx, triangle_shaders, triangle_buffers):
    """end_rendering is recorded even when the block raises — the command
    buffer is left in a consistent state and can be re-recorded and used."""
    vert, frag = triangle_shaders
    vbuf, ibuf = triangle_buffers
    target = bz.RenderTarget(ctx, 64, 64)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                .build(target))

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with pytest.raises(RuntimeError, match="user code"):
        with cmd.rendering(target, clear_color=CLEAR):
            raise RuntimeError("user code")

    # The pair is balanced (begin + end were both recorded), so the recording
    # is submittable as-is...
    ctx.submit(cmd)

    # ...and a fresh recording over the same cmd works normally.
    cmd.begin()
    with cmd.rendering(target, clear_color=CLEAR) as c:
        c.bind_pipeline(pipeline).bind_vertex_buffer(vbuf).bind_index_buffer(ibuf).draw_indexed(3)
    ctx.submit(cmd)
    assert target.read_pixels() is not None
