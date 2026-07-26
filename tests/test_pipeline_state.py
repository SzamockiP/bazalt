"""0.16: the pipeline state that used to be hard-coded.

blend() was wired to alpha, depthWriteEnable was glued to depth_test, the
compare op was always LESS_OR_EQUAL and the polygon mode was always FILL. Each
test here is two-sided: it renders the same geometry with and without the new
setting, so a knob that silently does nothing fails.

Headless throughout, so CI covers it.
"""

import pathlib
import struct

import numpy as np
import pytest

import bazalt as bz

SHADER_DIR = pathlib.Path(__file__).parent / "shaders"


@pytest.fixture
def fullscreen_push(ctx):
    """Build a fullscreen push-constant-coloured pipeline for `target`.

    far=True moves the draw to z = 0.9, which is how two draws disagree about
    depth. Everything else is forwarded to the builder verb it names.
    """
    frag = ctx.compile_shader(str(SHADER_DIR / "push.frag"), bz.ShaderStage.FRAGMENT)

    def make(target, *, far=False, blend=None, depth=None):
        name = "fullscreen_far.vert" if far else "fullscreen.vert"
        vert = ctx.compile_shader(str(SHADER_DIR / name), bz.ShaderStage.VERTEX)
        builder = (ctx.graphics_pipeline()
                   .vertex_shader(vert)
                   .fragment_shader(frag)
                   .push_constant(16, bz.ShaderStage.FRAGMENT))
        if blend is not None:
            builder = builder.blend(True, mode=blend)
        if depth is not None:
            builder = builder.depth_test(True, **depth)
        return builder.build(target)

    return make


def draw_over(ctx, target, pipeline, clear, rgba):
    """Clear, draw one fullscreen quad of `rgba`, submit, read the centre pixel."""
    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(target, clear_color=clear)
    cmd.bind_pipeline(pipeline)
    cmd.push_constants(pipeline, 0, struct.pack("4f", *rgba))
    cmd.draw(3)
    cmd.end_rendering(target)
    ctx.submit(cmd)
    return target.read_pixels()[32, 32]


# ── blend modes ───────────────────────────────────────────────────────────


def test_additive_accumulates_where_alpha_replaces(ctx, fullscreen_push):
    """Quarter-red over quarter-red: additive reaches a half, alpha stays a
    quarter because a fully opaque source replaces the destination."""
    target = bz.RenderTarget(ctx, 64, 64)
    quarter = [0.25, 0.0, 0.0, 1.0]

    additive = fullscreen_push(target, blend=bz.BlendMode.ADDITIVE)
    assert np.allclose(draw_over(ctx, target, additive, quarter, quarter)[:3],
                       [128, 0, 0], atol=3)

    alpha = fullscreen_push(target, blend=bz.BlendMode.ALPHA)
    assert np.allclose(draw_over(ctx, target, alpha, quarter, quarter)[:3],
                       [64, 0, 0], atol=3)


def test_premultiplied_does_not_scale_the_source(ctx, fullscreen_push):
    """(0.5, 0, 0, 0.5) over half-blue.

    Premultiplied keeps the source at 0.5 red and attenuates only the
    destination. Alpha multiplies the source by its own 0.5 as well, so the red
    lands at a quarter instead of a half.
    """
    target = bz.RenderTarget(ctx, 64, 64)
    half_blue = [0.0, 0.0, 0.5, 1.0]
    source = [0.5, 0.0, 0.0, 0.5]

    premultiplied = fullscreen_push(target, blend=bz.BlendMode.PREMULTIPLIED)
    assert np.allclose(draw_over(ctx, target, premultiplied, half_blue, source)[:3],
                       [128, 0, 64], atol=3)

    alpha = fullscreen_push(target, blend=bz.BlendMode.ALPHA)
    assert np.allclose(draw_over(ctx, target, alpha, half_blue, source)[:3],
                       [64, 0, 64], atol=3)


def test_blend_off_ignores_the_mode(ctx, fullscreen_push):
    """blend(False, mode=ADDITIVE) must still replace, not add."""
    target = bz.RenderTarget(ctx, 64, 64)
    frag = ctx.compile_shader(str(SHADER_DIR / "push.frag"), bz.ShaderStage.FRAGMENT)
    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .push_constant(16, bz.ShaderStage.FRAGMENT)
                .blend(False, mode=bz.BlendMode.ADDITIVE)
                .build(target))

    quarter = [0.25, 0.0, 0.0, 1.0]
    assert np.allclose(draw_over(ctx, target, pipeline, quarter, quarter)[:3],
                       [64, 0, 0], atol=3)


# ── depth write and compare ───────────────────────────────────────────────


def test_depth_write_false_leaves_the_buffer_alone(ctx, fullscreen_push):
    """A near draw that does not write depth cannot occlude a later far draw.

    The control is the same pair with write=True, where the far draw loses.
    """
    target = bz.RenderTarget(ctx, 64, 64, depth=bz.Format.D32F)

    def render(write):
        near = fullscreen_push(target, depth={"write": write})
        far = fullscreen_push(target, far=True, depth={})
        cmd = ctx.create_command_buffer()
        cmd.begin()
        cmd.begin_rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0])
        cmd.bind_pipeline(near)
        cmd.push_constants(near, 0, struct.pack("4f", 0.0, 1.0, 0.0, 1.0))
        cmd.draw(3)
        cmd.bind_pipeline(far)
        cmd.push_constants(far, 0, struct.pack("4f", 0.0, 0.0, 1.0, 1.0))
        cmd.draw(3)
        cmd.end_rendering(target)
        ctx.submit(cmd)
        return target.read_pixels()[32, 32]

    assert np.allclose(render(False)[:3], [0, 0, 255], atol=2), \
        "write=False still wrote depth, so the far draw was rejected"
    assert np.allclose(render(True)[:3], [0, 255, 0], atol=2), \
        "write=True did not write depth, so the far draw was not rejected"


def test_depth_compare_never_rejects_everything(ctx, fullscreen_push):
    """compare= reaches depthCompareOp: NEVER leaves the clear colour intact,
    where the default LESS_OR_EQUAL paints over it."""
    target = bz.RenderTarget(ctx, 64, 64, depth=bz.Format.D32F)
    red = [1.0, 0.0, 0.0, 1.0]
    green = [0.0, 1.0, 0.0, 1.0]

    never = fullscreen_push(target, depth={"compare": bz.CompareOp.NEVER})
    assert np.allclose(draw_over(ctx, target, never, red, green)[:3], [255, 0, 0], atol=2)

    default = fullscreen_push(target, depth={})
    assert np.allclose(draw_over(ctx, target, default, red, green)[:3], [0, 255, 0], atol=2)


def test_depth_test_off_still_writes_nothing(ctx, fullscreen_push):
    """depth_test(False) has always meant "nothing to do with depth", and the
    new write= default must not quietly turn writes back on: a far draw after a
    near one must survive."""
    target = bz.RenderTarget(ctx, 64, 64, depth=bz.Format.D32F)
    frag = ctx.compile_shader(str(SHADER_DIR / "push.frag"), bz.ShaderStage.FRAGMENT)

    def plain(name):
        vert = ctx.compile_shader(str(SHADER_DIR / name), bz.ShaderStage.VERTEX)
        return (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .push_constant(16, bz.ShaderStage.FRAGMENT)
                .depth_test(False)
                .build(target))

    near = plain("fullscreen.vert")
    far = plain("fullscreen_far.vert")

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0])
    cmd.bind_pipeline(near)
    cmd.push_constants(near, 0, struct.pack("4f", 0.0, 1.0, 0.0, 1.0))
    cmd.draw(3)
    cmd.bind_pipeline(far)
    cmd.push_constants(far, 0, struct.pack("4f", 0.0, 0.0, 1.0, 1.0))
    cmd.draw(3)
    cmd.end_rendering(target)
    ctx.submit(cmd)

    assert np.allclose(target.read_pixels()[32, 32, :3], [0, 0, 255], atol=2)


# ── polygon mode ──────────────────────────────────────────────────────────


def test_line_mode_needs_the_wireframe_feature(ctx, triangle_shaders):
    """Anything but FILL is fillModeNonSolid, so it is negotiated like every
    other optional capability. The session Context does not ask for it."""
    if ctx.supports(bz.Feature.WIREFRAME):
        pytest.skip("session Context happens to have WIREFRAME enabled")

    vert, frag = triangle_shaders
    target = bz.RenderTarget(ctx, 64, 64)
    with pytest.raises(bz.ShaderError, match="WIREFRAME"):
        (ctx.graphics_pipeline()
         .vertex_shader(vert)
         .fragment_shader(frag)
         .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
         .polygon_mode(bz.PolygonMode.LINE)
         .build(target))


def test_line_mode_draws_edges_and_leaves_the_interior(extra_context):
    """The wireframe view: LINE paints the silhouette of the centre triangle and
    not its middle, and touches far fewer pixels than FILL.

    Its own Context, because the feature has to be asked for at device creation
    and the session one does not. `extra_context` applies the same
    validation-as-assert referee.
    """
    ctx = extra_context(optional=[bz.Feature.WIREFRAME])
    if not ctx.supports(bz.Feature.WIREFRAME):
        pytest.skip("GPU reports no fillModeNonSolid")

    vert = ctx.compile_shader(str(SHADER_DIR / "triangle.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "triangle.frag"), bz.ShaderStage.FRAGMENT)
    vertices = [
        +0.0, -0.5, 0.0, 1.0, 0.0, 0.0,
        -0.5, +0.5, 0.0, 0.0, 1.0, 0.0,
        +0.5, +0.5, 0.0, 0.0, 0.0, 1.0,
    ]
    vbuf = ctx.create_buffer(vertices, bz.BufferType.VERTEX, bz.MemoryUsage.STATIC, bz.DataType.FLOAT)
    ibuf = ctx.create_buffer([0, 1, 2], bz.BufferType.INDEX, bz.MemoryUsage.STATIC, bz.DataType.UINT32)
    target = bz.RenderTarget(ctx, 64, 64)

    def render(mode):
        pipeline = (ctx.graphics_pipeline()
                    .vertex_shader(vert)
                    .fragment_shader(frag)
                    .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                    .polygon_mode(mode)
                    .build(target))
        cmd = ctx.create_command_buffer()
        cmd.begin()
        cmd.begin_rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0])
        cmd.bind_pipeline(pipeline)
        cmd.bind_vertex_buffer(vbuf)
        cmd.bind_index_buffer(ibuf)
        cmd.draw_indexed(3)
        cmd.end_rendering(target)
        ctx.submit(cmd)
        return target.read_pixels().copy()

    filled = render(bz.PolygonMode.FILL)
    lined = render(bz.PolygonMode.LINE)

    # The triangle spans pixels (32,16), (16,48), (48,48), so its centroid sits
    # at about row 37, column 32 — inside the fill, away from every edge.
    assert filled[37, 32, :3].any(), "the filled triangle missed its own centroid"
    assert not lined[37, 32, :3].any(), "LINE painted the interior"

    painted = lambda px: int(np.count_nonzero(px[:, :, :3].any(axis=2)))
    assert 0 < painted(lined) < painted(filled) // 2, \
        f"line {painted(lined)} px vs fill {painted(filled)} px"
