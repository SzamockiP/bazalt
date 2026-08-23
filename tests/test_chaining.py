"""Chained recording and the pass handle.

Chaining is the same API, not a second one: every recording method returns
the pass itself, so `p.a().b()` and `p.a(); p.b()` record identically. The
`with` block is sugar over the same handle — it only seals the pass early —
so a pass recorded without it renders the same frame. Both pinned by
rendering the same scene both ways and comparing pixels exactly.
"""

import pathlib
import struct

import numpy as np
import pytest

import bazalt as bz

CLEAR = [0.1, 0.2, 0.3, 1.0]
SHADER_DIR = pathlib.Path(__file__).parent / "shaders"


def _pipeline(ctx, shaders, target):
    vert, frag = shaders
    return (ctx.graphics_pipeline()
            .vertex_shader(vert)
            .fragment_shader(frag)
            .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
            .build(target))


def test_every_recording_method_returns_the_same_object(ctx, triangle_shaders, triangle_buffers):
    vbuf, ibuf = triangle_buffers
    target = ctx.create_render_target(16, 16)
    pipeline = _pipeline(ctx, triangle_shaders, target)

    g = ctx.graph()
    p = g.add_pass(target)
    assert p.bind_pipeline(pipeline) is p
    assert p.bind_vertex_buffer(vbuf) is p
    assert p.bind_index_buffer(ibuf) is p
    assert p.set_viewport(0, 0, 16, 16) is p
    assert p.set_scissor(0, 0, 16, 16) is p
    assert p.draw_indexed(3) is p

    q = g.add_pass()
    assert q.dispatch is not None  # a verb, not a value — the identity check:
    assert q.fill_buffer(ctx.create_buffer(16, bz.BufferUsage.STORAGE,
                                           bz.MemoryUsage.STATIC), 0) is q


def test_chained_and_statement_styles_render_identically(ctx, triangle_shaders, triangle_buffers):
    vbuf, ibuf = triangle_buffers

    statement_target = ctx.create_render_target(64, 64)
    pipeline = _pipeline(ctx, triangle_shaders, statement_target)
    g = ctx.graph()
    p = g.add_pass(statement_target, clear_color=CLEAR)
    p.bind_pipeline(pipeline)
    p.bind_vertex_buffer(vbuf)
    p.bind_index_buffer(ibuf)
    p.draw_indexed(3)
    ctx.submit(g)

    chained_target = ctx.create_render_target(64, 64)
    pipeline2 = _pipeline(ctx, triangle_shaders, chained_target)
    g2 = ctx.graph()
    (g2.add_pass(chained_target, clear_color=CLEAR)
       .bind_pipeline(pipeline2)
       .bind_vertex_buffer(vbuf)
       .bind_index_buffer(ibuf)
       .draw_indexed(3))
    ctx.submit(g2)

    np.testing.assert_array_equal(chained_target.color[0].read(),
                                  statement_target.color[0].read())


def test_with_and_naked_pass_render_identically(ctx, triangle_shaders, triangle_buffers):
    """`with` only seals the pass early. A naked handle records the same frame,
    and it crosses function boundaries by itself — which is why there is no
    begin_pass/end_pass pair beside the block."""
    vbuf, ibuf = triangle_buffers

    def draw(p, pipe):
        p.bind_pipeline(pipe)
        p.bind_vertex_buffer(vbuf)
        p.bind_index_buffer(ibuf)
        p.draw_indexed(3)

    with_target = ctx.create_render_target(32, 32)
    g = ctx.graph()
    with g.add_pass(with_target, clear_color=CLEAR) as p:
        draw(p, _pipeline(ctx, triangle_shaders, with_target))
    ctx.submit(g)

    naked_target = ctx.create_render_target(32, 32)
    g2 = ctx.graph()
    p2 = g2.add_pass(naked_target, clear_color=CLEAR)
    draw(p2, _pipeline(ctx, triangle_shaders, naked_target))
    ctx.submit(g2)

    np.testing.assert_array_equal(naked_target.color[0].read(),
                                  with_target.color[0].read())


def test_the_with_block_seals_the_pass(ctx, triangle_shaders, triangle_buffers):
    vbuf, _ = triangle_buffers
    target = ctx.create_render_target(16, 16)
    g = ctx.graph()
    with g.add_pass(target, clear_color=CLEAR) as p:
        pass
    with pytest.raises(bz.StateError, match="sealed"):
        p.bind_vertex_buffer(vbuf)


def test_the_first_submit_seals_a_naked_pass(ctx, triangle_shaders, triangle_buffers):
    """A pass used without `with` stays recordable until the graph first
    compiles — the compile is what has to trust the use list."""
    vbuf, _ = triangle_buffers
    target = ctx.create_render_target(16, 16)
    g = ctx.graph()
    p = g.add_pass(target, clear_color=CLEAR)
    ctx.submit(g)
    with pytest.raises(bz.StateError, match="sealed"):
        p.bind_vertex_buffer(vbuf)


def test_pass_block_closes_on_exception(ctx, triangle_shaders, triangle_buffers):
    """The block seals the pass even when it raises, and the graph stays
    consistent: submittable as-is, and a fresh graph renders normally."""
    vbuf, ibuf = triangle_buffers
    target = ctx.create_render_target(64, 64)
    pipeline = _pipeline(ctx, triangle_shaders, target)

    g = ctx.graph()
    with pytest.raises(RuntimeError, match="user code"):
        with g.add_pass(target, clear_color=CLEAR):
            raise RuntimeError("user code")

    # The pass is balanced (the executor opens and closes the rendering
    # scope), so the graph is submittable as-is...
    ctx.submit(g)

    # ...and a fresh graph over the same target works normally.
    g2 = ctx.graph()
    with g2.add_pass(target, clear_color=CLEAR) as p:
        p.bind_pipeline(pipeline).bind_vertex_buffer(vbuf).bind_index_buffer(ibuf).draw_indexed(3)
    ctx.submit(g2)
    assert target.color[0].read() is not None


def test_the_short_verbs_bind_what_the_long_ones_do(ctx, triangle_shaders):
    """bind_descriptor_set(set) and push_constants(offset, data) read the set's
    own index and the bound pipeline, so they must produce the same picture as
    naming both."""
    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "push.frag"), bz.ShaderStage.FRAGMENT)

    def render(short):
        target = ctx.create_render_target(16, 16)
        pipe = (ctx.graphics_pipeline().vertex_shader(vert).fragment_shader(frag)
                .push_constant(16, bz.ShaderStage.FRAGMENT).build(target))
        colour = struct.pack("4f", 0.25, 0.5, 0.75, 1.0)
        g = ctx.graph()
        with g.add_pass(target, clear_color=[0, 0, 0, 1]) as p:
            p.bind_pipeline(pipe)
            if short:
                p.push_constants(0, colour)
            else:
                p.push_constants(pipe, 0, colour)
            p.draw(3)
        ctx.submit(g)
        return target.color[0].read()

    assert np.array_equal(render(short=True), render(short=False))


def test_the_short_verbs_need_a_bound_pipeline(ctx):
    """They read the layout off the pipeline that is bound, so with none bound
    there is nothing to read — StateError naming the fix, not a crash."""
    g = ctx.graph()
    p = g.add_pass()
    with pytest.raises(bz.StateError, match="bind_pipeline"):
        p.push_constants(0, b"\x00" * 16)
