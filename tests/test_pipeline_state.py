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


def test_reversed_depth_needs_both_the_compare_and_the_clear(ctx, fullscreen_push):
    """compare=GREATER is unusable while the depth clear is fixed at 1.0, since
    nothing is ever greater than the far plane. clear_depth= is the other half:
    clear to 0.0 and a reversed-depth buffer works, which is what the pair is
    for.
    """
    target = bz.RenderTarget(ctx, 64, 64, depth=bz.Format.D32F)
    red = [1.0, 0.0, 0.0, 1.0]
    green = [0.0, 1.0, 0.0, 1.0]
    greater = fullscreen_push(target, far=True, depth={"compare": bz.CompareOp.GREATER})

    def render(clear_depth):
        cmd = ctx.create_command_buffer()
        cmd.begin()
        cmd.begin_rendering(target, clear_color=red, clear_depth=clear_depth)
        cmd.bind_pipeline(greater)
        cmd.push_constants(greater, 0, struct.pack("4f", *green))
        cmd.draw(3)
        cmd.end_rendering(target)
        ctx.submit(cmd)
        return target.read_pixels()[32, 32]

    # 0.9 > 1.0 is false: the default clear rejects every fragment.
    assert np.allclose(render(1.0)[:3], [255, 0, 0], atol=2)
    # 0.9 > 0.0 is true.
    assert np.allclose(render(0.0)[:3], [0, 255, 0], atol=2)


def test_clear_depth_is_ignored_when_the_pass_preserves(ctx, fullscreen_push):
    """clear_depth is the clear VALUE, not a second preserve switch. A preserving
    pass keeps the depth the previous pass wrote whatever it says."""
    target = bz.RenderTarget(ctx, 64, 64, depth=bz.Format.D32F)
    near = fullscreen_push(target, depth={})
    far = fullscreen_push(target, far=True, depth={})

    cmd = ctx.create_command_buffer()
    cmd.begin()
    # Pass 1 writes depth 0.0 over the whole target.
    cmd.begin_rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0])
    cmd.bind_pipeline(near)
    cmd.push_constants(near, 0, struct.pack("4f", 0.0, 1.0, 0.0, 1.0))
    cmd.draw(3)
    cmd.end_rendering(target)
    # Pass 2 preserves. A clear_depth of 0.0 would change nothing either way,
    # so ask for 1.0: if it were honoured, the far draw would pass and win.
    cmd.begin_rendering(target, clear_color=None, clear_depth=1.0)
    cmd.bind_pipeline(far)
    cmd.push_constants(far, 0, struct.pack("4f", 0.0, 0.0, 1.0, 1.0))
    cmd.draw(3)
    cmd.end_rendering(target)
    ctx.submit(cmd)

    assert np.allclose(target.read_pixels()[32, 32, :3], [0, 255, 0], atol=2)


def test_depth_bias_pushes_the_written_depth(ctx, fullscreen_push):
    """A bias big enough to matter changes which of two coincident surfaces
    wins, which is the mechanism behind fixing shadow acne.

    Both draws sit at z = 0.9. Equal depth passes LESS_OR_EQUAL, so the second
    normally wins; a NEGATIVE bias on the first pulls its written depth nearer
    and rejects the second.

    The magnitude is the trap. On a FLOATING-POINT depth buffer Vulkan scales
    depthBiasConstantFactor by 2^(exponent - 23) of the largest depth in the
    primitive, which is about 6e-8 at z = 0.9. So the 0.001-ish constants from a
    D24_UNORM tutorial move the depth by nothing at all, and a visible offset
    needs five or six digits.
    """
    target = bz.RenderTarget(ctx, 64, 64, depth=bz.Format.D32F)
    frag = ctx.compile_shader(str(SHADER_DIR / "push.frag"), bz.ShaderStage.FRAGMENT)
    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen_far.vert"), bz.ShaderStage.VERTEX)

    def build(bias):
        builder = (ctx.graphics_pipeline()
                   .vertex_shader(vert)
                   .fragment_shader(frag)
                   .push_constant(16, bz.ShaderStage.FRAGMENT)
                   .depth_test(True))
        if bias is not None:
            builder = builder.depth_bias(bias)
        return builder.build(target)

    def render(bias):
        first = build(bias)
        second = build(None)
        cmd = ctx.create_command_buffer()
        cmd.begin()
        cmd.begin_rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0])
        cmd.bind_pipeline(first)
        cmd.push_constants(first, 0, struct.pack("4f", 0.0, 1.0, 0.0, 1.0))
        cmd.draw(3)
        cmd.bind_pipeline(second)
        cmd.push_constants(second, 0, struct.pack("4f", 0.0, 0.0, 1.0, 1.0))
        cmd.draw(3)
        cmd.end_rendering(target)
        ctx.submit(cmd)
        return target.read_pixels()[32, 32]

    # No bias: both are at 0.9, LESS_OR_EQUAL lets the second one through.
    assert np.allclose(render(None)[:3], [0, 0, 255], atol=2)
    # -200000 * 6e-8 is about -0.012, so the first draw writes nearer than it
    # rasterizes and the second one loses.
    assert np.allclose(render(-200000.0)[:3], [0, 255, 0], atol=2), \
        "depth_bias did not move the written depth"


def test_line_width_needs_the_wide_lines_feature(ctx, fullscreen_push):
    """A driver may support exactly one line width, so anything but 1.0 is
    negotiated. 1.0 must stay free."""
    target = bz.RenderTarget(ctx, 64, 64)
    frag = ctx.compile_shader(str(SHADER_DIR / "push.frag"), bz.ShaderStage.FRAGMENT)
    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)

    def build(width):
        return (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .push_constant(16, bz.ShaderStage.FRAGMENT)
                .line_width(width)
                .build(target))

    assert build(1.0) is not None

    if ctx.supports(bz.Feature.WIDE_LINES):
        pytest.skip("session Context happens to have WIDE_LINES enabled")
    with pytest.raises(bz.ShaderError, match="WIDE_LINES"):
        build(3.0)


def test_wide_lines_thicken_the_wireframe(extra_context):
    """The pairing that makes line_width worth having: a 1-pixel wireframe is
    nearly invisible, so a wider one must paint strictly more pixels."""
    ctx = extra_context(optional=[bz.Feature.WIREFRAME, bz.Feature.WIDE_LINES])
    if not (ctx.supports(bz.Feature.WIREFRAME) and ctx.supports(bz.Feature.WIDE_LINES)):
        pytest.skip("GPU lacks fillModeNonSolid or wideLines")

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

    def painted(width):
        pipeline = (ctx.graphics_pipeline()
                    .vertex_shader(vert)
                    .fragment_shader(frag)
                    .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                    .polygon_mode(bz.PolygonMode.LINE)
                    .line_width(width)
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
        px = target.read_pixels()
        return int(np.count_nonzero(px[:, :, :3].any(axis=2)))

    assert painted(3.0) > painted(1.0)


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
