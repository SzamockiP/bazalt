"""Tessellation and geometry stages (0.19).

Every test here builds its own Context, because both stages are optional device
features that have to be asked for at device creation and the session Context does
not ask. `extra_context` applies the same validation-as-assert referee.

Two things are being pinned down, and the second one is easy to overlook:

1. That the stages *do something*. Each pixel test is two-sided — the same
   geometry rendered with and without the new stage — because a knob that
   silently does nothing passes a one-sided test.
2. That a barrier on a Context with these features enabled is legal. The shader
   stage mask stopped being a compile-time constant in 0.19 (see
   Context::all_shader_stages): Vulkan forbids naming the tessellation or
   geometry stage in a barrier unless the feature is on, and it equally forbids
   omitting a stage that really did the read. The session Context covers the
   narrow mask; only a test here can cover the wide one.
"""

import struct

import numpy as np
import pytest

import bazalt as bz

from conftest import SHADER_DIR

# An equilateral triangle whose corners sit exactly on the circle of radius 0.8
# that disc.tese projects onto — which is what makes level 1 reproduce the input.
TRIANGLE_ON_CIRCLE = [
    +0.0000, +0.8000,
    -0.6928, -0.4000,
    +0.6928, -0.4000,
]


def tess_context(extra_context):
    ctx = extra_context(optional=[bz.Feature.TESSELLATION])
    if not ctx.supports(bz.Feature.TESSELLATION):
        pytest.skip("GPU reports no tessellationShader")
    return ctx


def geometry_context(extra_context):
    ctx = extra_context(optional=[bz.Feature.GEOMETRY_SHADER])
    if not ctx.supports(bz.Feature.GEOMETRY_SHADER):
        pytest.skip("GPU reports no geometryShader")
    return ctx


def painted(target):
    px = target.color[0].read()
    return int(np.count_nonzero(px[:, :, :3].any(axis=2)))


# ── Tessellation ──────────────────────────────────────────────────────────────

def test_tessellation_level_changes_what_is_drawn(extra_context):
    """The two-sided test. disc.tese pushes every generated vertex onto a circle
    the three input corners already sit on, so level 1 paints the input triangle
    and level 16 fills out toward the disc around it.

    One pipeline, two draws: the level is a push constant, so what differs
    between the two sides is only the number the tessellator read.
    """
    ctx = tess_context(extra_context)

    vert = ctx.compile_shader(str(SHADER_DIR / "pos2.vert"), bz.ShaderStage.VERTEX)
    tesc = ctx.compile_shader(str(SHADER_DIR / "disc.tesc"), bz.ShaderStage.TESS_CONTROL)
    tese = ctx.compile_shader(str(SHADER_DIR / "disc.tese"), bz.ShaderStage.TESS_EVALUATION)
    frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)

    vbuf = ctx.create_buffer(TRIANGLE_ON_CIRCLE, bz.BufferType.VERTEX,
                             bz.MemoryUsage.STATIC, bz.DataType.FLOAT)
    target = bz.RenderTarget(ctx, 128, 128)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .tess_control_shader(tesc)
                .tess_evaluation_shader(tese)
                .fragment_shader(frag)
                .vertex_format([bz.VertexFormat.FLOAT2])
                .topology(bz.Topology.PATCH_LIST)
                .patch_control_points(3)
                .push_constant(4, bz.ShaderStage.TESS_CONTROL)
                .cull_mode(bz.CullMode.NONE, bz.FrontFace.COUNTER_CLOCKWISE)
                .build(target))

    def draw(level):
        cmd = ctx.create_command_buffer()
        cmd.begin()
        cmd.begin_rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0])
        cmd.bind_pipeline(pipeline)
        cmd.push_constants(pipeline, 0, struct.pack("f", level))
        cmd.bind_vertex_buffer(vbuf)
        cmd.draw(3)
        cmd.end_rendering(target)
        ctx.submit(cmd)
        return painted(target)

    flat = draw(1.0)
    subdivided = draw(16.0)

    # An inscribed equilateral triangle is ~1.30 r^2 against the disc's ~3.14 r^2,
    # so the real ratio is about 2.4. A loose bound keeps this about "tessellation
    # happened" rather than about the tessellator's exact vertex placement.
    assert flat > 0
    assert subdivided > flat * 1.5


def test_patch_list_without_tessellation_shaders_is_refused(extra_context):
    """The two halves of one mistake. A patch has nothing to tessellate it, so
    bazalt names the missing stages instead of letting the layers talk about
    topology."""
    ctx = tess_context(extra_context)
    vert = ctx.compile_shader(str(SHADER_DIR / "pos2.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, 32, 32)

    with pytest.raises(bz.ShaderError, match="PATCH_LIST"):
        (ctx.graphics_pipeline()
         .vertex_shader(vert)
         .fragment_shader(frag)
         .vertex_format([bz.VertexFormat.FLOAT2])
         .topology(bz.Topology.PATCH_LIST)
         .build(target))


def test_one_tessellation_stage_without_the_other_is_refused(extra_context):
    """Tessellation is a pair with a fixed-function tessellator between the two
    halves, so one alone is not a partial pipeline but an invalid one."""
    ctx = tess_context(extra_context)
    vert = ctx.compile_shader(str(SHADER_DIR / "pos2.vert"), bz.ShaderStage.VERTEX)
    tesc = ctx.compile_shader(str(SHADER_DIR / "disc.tesc"), bz.ShaderStage.TESS_CONTROL)
    frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, 32, 32)

    with pytest.raises(bz.ShaderError, match="BOTH stages"):
        (ctx.graphics_pipeline()
         .vertex_shader(vert)
         .tess_control_shader(tesc)
         .fragment_shader(frag)
         .vertex_format([bz.VertexFormat.FLOAT2])
         .topology(bz.Topology.PATCH_LIST)
         .patch_control_points(3)
         .build(target))


def test_patch_control_points_must_be_in_range(extra_context):
    """0 is as wrong as a million, and the message names the device's limit
    rather than leaving the caller to look it up."""
    ctx = tess_context(extra_context)
    vert = ctx.compile_shader(str(SHADER_DIR / "pos2.vert"), bz.ShaderStage.VERTEX)
    tesc = ctx.compile_shader(str(SHADER_DIR / "disc.tesc"), bz.ShaderStage.TESS_CONTROL)
    tese = ctx.compile_shader(str(SHADER_DIR / "disc.tese"), bz.ShaderStage.TESS_EVALUATION)
    frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, 32, 32)

    def build(count):
        builder = (ctx.graphics_pipeline()
                   .vertex_shader(vert)
                   .tess_control_shader(tesc)
                   .tess_evaluation_shader(tese)
                   .fragment_shader(frag)
                   .vertex_format([bz.VertexFormat.FLOAT2])
                   .topology(bz.Topology.PATCH_LIST))
        if count is not None:
            builder = builder.patch_control_points(count)
        return builder.build(target)

    # Never set at all is the same error as explicitly zero: both mean "no patch
    # size", and the default cannot be a working guess.
    with pytest.raises(bz.ShaderError, match="patch_control_points"):
        build(None)
    with pytest.raises(bz.ShaderError, match="patch_control_points"):
        build(0)
    with pytest.raises(bz.ShaderError, match="patch_control_points"):
        build(100000)


def test_tessellation_needs_its_feature_at_compile_time(ctx):
    """The refusal happens at compile_shader, not at build().

    That is not an ergonomic preference. SPIR-V for these stages declares
    OpCapability Tessellation, and vkCreateShaderModule rejects a capability whose
    feature is off (VUID-VkShaderModuleCreateInfo-pCode-08740) — so a compile that
    "succeeded" would already have produced an invalid module and a validation
    error, some distance from the line that caused it. The session Context never
    asked for TESSELLATION, so this is the ordinary shape of the mistake.
    """
    with pytest.raises(bz.ShaderError, match="TESSELLATION"):
        ctx.compile_shader(str(SHADER_DIR / "disc.tesc"), bz.ShaderStage.TESS_CONTROL)
    with pytest.raises(bz.ShaderError, match="TESSELLATION"):
        ctx.compile_shader(str(SHADER_DIR / "disc.tese"), bz.ShaderStage.TESS_EVALUATION)


def test_the_pipeline_gate_catches_a_module_from_another_context(ctx, extra_context):
    """The second gate, and why it is not a duplicate of the first.

    A module compiled on a Context that HAS tessellation, built into a pipeline on
    one that does not, passes the compile-time check by construction. The builder
    refuses it before the foreign handle is ever touched, which turns what would
    be a cross-device handle error into a sentence naming the feature.
    """
    tess = extra_context(optional=[bz.Feature.TESSELLATION])
    if not tess.supports(bz.Feature.TESSELLATION):
        pytest.skip("GPU reports no tessellationShader")

    tesc = tess.compile_shader(str(SHADER_DIR / "disc.tesc"), bz.ShaderStage.TESS_CONTROL)
    tese = tess.compile_shader(str(SHADER_DIR / "disc.tese"), bz.ShaderStage.TESS_EVALUATION)

    vert = ctx.compile_shader(str(SHADER_DIR / "pos2.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, 32, 32)

    with pytest.raises(bz.ShaderError, match="TESSELLATION"):
        (ctx.graphics_pipeline()
         .vertex_shader(vert)
         .tess_control_shader(tesc)
         .tess_evaluation_shader(tese)
         .fragment_shader(frag)
         .vertex_format([bz.VertexFormat.FLOAT2])
         .topology(bz.Topology.PATCH_LIST)
         .patch_control_points(3)
         .build(target))


# ── Geometry ──────────────────────────────────────────────────────────────────

def test_geometry_shader_turns_points_into_surfaces(extra_context):
    """The two-sided test, and the argument for the stage existing: the same
    POINT_LIST draw paints one pixel per point on its own, and a quad per point
    through point_quad.geom. Changing the primitive type is the thing
    tessellation cannot do."""
    ctx = geometry_context(extra_context)

    # Two vertex shaders, and the difference is not incidental: with no geometry
    # stage the vertex shader is the last pre-rasterization stage and POINT_LIST
    # requires it to write gl_PointSize, while with one the geometry shader takes
    # that role and emits triangles instead.
    plain_vert = ctx.compile_shader(str(SHADER_DIR / "point.vert"), bz.ShaderStage.VERTEX)
    geom_vert = ctx.compile_shader(str(SHADER_DIR / "pos2.vert"), bz.ShaderStage.VERTEX)
    geom = ctx.compile_shader(str(SHADER_DIR / "point_quad.geom"), bz.ShaderStage.GEOMETRY)
    frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)

    vbuf = ctx.create_buffer([0.0, 0.0], bz.BufferType.VERTEX,
                             bz.MemoryUsage.STATIC, bz.DataType.FLOAT)
    target = bz.RenderTarget(ctx, 128, 128)

    def draw(with_geometry):
        builder = (ctx.graphics_pipeline()
                   .vertex_shader(geom_vert if with_geometry else plain_vert)
                   .fragment_shader(frag)
                   .vertex_format([bz.VertexFormat.FLOAT2])
                   .topology(bz.Topology.POINT_LIST)
                   .cull_mode(bz.CullMode.NONE, bz.FrontFace.COUNTER_CLOCKWISE))
        if with_geometry:
            builder = builder.geometry_shader(geom)
        pipeline = builder.build(target)

        cmd = ctx.create_command_buffer()
        cmd.begin()
        cmd.begin_rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0])
        cmd.bind_pipeline(pipeline)
        cmd.bind_vertex_buffer(vbuf)
        cmd.draw(1)
        cmd.end_rendering(target)
        ctx.submit(cmd)
        return painted(target)

    bare = draw(False)
    amplified = draw(True)

    # One point is one pixel; the quad is 0.5 x 0.5 of clip space, so a quarter
    # of a 128x128 target either side of centre — about 1024 pixels.
    assert bare == 1
    assert amplified > 500


def test_geometry_needs_its_feature(ctx):
    """Same reach contract, same compile-time refusal. Also the reason geometry is
    a Feature and not a free stage: Metal has no geometry shaders at all, so
    MoltenVK reports geometryShader false."""
    with pytest.raises(bz.ShaderError, match="GEOMETRY_SHADER"):
        ctx.compile_shader(str(SHADER_DIR / "point_quad.geom"), bz.ShaderStage.GEOMETRY)


# ── The stage mask ────────────────────────────────────────────────────────────

def test_a_barrier_is_legal_on_a_tessellating_context(extra_context):
    """The referee for Context::all_shader_stages, and the reason the old
    constexpr kAllShaderStages had to go.

    A manual barrier on a Context with tessellation and geometry enabled names
    those stages in its mask. That is legal here and a validation error on a
    Context without the features, which is why the mask is per-Context. The
    assertion is the fixture: if the mask were wrong in either direction this
    test reports a validation error rather than a wrong pixel.
    """
    ctx = extra_context(optional=[bz.Feature.TESSELLATION, bz.Feature.GEOMETRY_SHADER])
    if not ctx.supports(bz.Feature.TESSELLATION):
        pytest.skip("GPU reports no tessellationShader")

    vert = ctx.compile_shader(str(SHADER_DIR / "pos2.vert"), bz.ShaderStage.VERTEX)
    tesc = ctx.compile_shader(str(SHADER_DIR / "disc.tesc"), bz.ShaderStage.TESS_CONTROL)
    tese = ctx.compile_shader(str(SHADER_DIR / "disc.tese"), bz.ShaderStage.TESS_EVALUATION)
    frag = ctx.compile_shader(str(SHADER_DIR / "ssbo_solid.frag"), bz.ShaderStage.FRAGMENT)

    buf = ctx.create_buffer([0.0, 1.0, 0.0, 1.0], bz.BufferType.STORAGE,
                            bz.MemoryUsage.STATIC, bz.DataType.FLOAT)
    vbuf = ctx.create_buffer(TRIANGLE_ON_CIRCLE, bz.BufferType.VERTEX,
                             bz.MemoryUsage.STATIC, bz.DataType.FLOAT)
    target = bz.RenderTarget(ctx, 64, 64)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .tess_control_shader(tesc)
                .tess_evaluation_shader(tese)
                .fragment_shader(frag)
                .vertex_format([bz.VertexFormat.FLOAT2])
                .topology(bz.Topology.PATCH_LIST)
                .patch_control_points(3)
                .push_constant(4, bz.ShaderStage.TESS_CONTROL)
                .storage_buffer(0, bz.ShaderStage.FRAGMENT, set=0)
                .cull_mode(bz.CullMode.NONE, bz.FrontFace.COUNTER_CLOCKWISE)
                .build(target))
    pool = ctx.create_descriptor_pool(max_sets=4, storage_buffers=4)
    dset = pool.allocate_set(pipeline, set=0)
    dset.set_buffer(0, buf)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    # The manual half: cmd.barrier() spells SHADER_READ/WRITE with the Context's
    # mask, which now names the tessellation and geometry stages.
    cmd.barrier(buf, bz.Access.SHADER_WRITE, bz.Access.SHADER_READ)
    # The automatic half, and the one that matters more. fill_buffer is a transfer
    # write; the draw below reads the same buffer in a shader, so track_draw_ emits
    # a RAW barrier whose destination mask is all_shader_stages(). A mask carrying
    # a stage whose feature is off fails here.
    # 0x3F800000 is 1.0f, so every component becomes 1.0 and the draw paints
    # white. Filling with 0 would leave the shader reading black and the pixel
    # assertion below could not tell "drew nothing" from "drew the fill".
    cmd.fill_buffer(buf, 0x3F800000)
    cmd.begin_rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0])
    cmd.bind_pipeline(pipeline)
    cmd.bind_descriptor_set(dset, pipeline, set=0)
    cmd.push_constants(pipeline, 0, struct.pack("f", 8.0))
    cmd.bind_vertex_buffer(vbuf)
    cmd.draw(3)
    cmd.end_rendering(target)
    ctx.submit(cmd)
    ctx.wait()

    # It really drew, so the barriers were around real work and not a no-op.
    assert painted(target) > 0


def test_tessellation_stages_reach_a_specialization_constant(extra_context):
    """constant() used to test FRAGMENT and send everything else to the vertex
    list, so a TESS_CONTROL constant would have been baked into the vertex
    shader. The map keyed by stage is what fixes that, and this is the test that
    would have caught it: the level comes from a constant, not a push, and the
    two sides differ only in the value baked in.
    """
    ctx = tess_context(extra_context)

    vert = ctx.compile_shader(str(SHADER_DIR / "pos2.vert"), bz.ShaderStage.VERTEX)
    tesc = ctx.compile_shader(str(SHADER_DIR / "disc_spec.tesc"), bz.ShaderStage.TESS_CONTROL)
    tese = ctx.compile_shader(str(SHADER_DIR / "disc.tese"), bz.ShaderStage.TESS_EVALUATION)
    frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)

    vbuf = ctx.create_buffer(TRIANGLE_ON_CIRCLE, bz.BufferType.VERTEX,
                             bz.MemoryUsage.STATIC, bz.DataType.FLOAT)
    target = bz.RenderTarget(ctx, 128, 128)

    def draw(level):
        pipeline = (ctx.graphics_pipeline()
                    .vertex_shader(vert)
                    .tess_control_shader(tesc)
                    .tess_evaluation_shader(tese)
                    .fragment_shader(frag)
                    .vertex_format([bz.VertexFormat.FLOAT2])
                    .topology(bz.Topology.PATCH_LIST)
                    .patch_control_points(3)
                    .constant(0, level, bz.ShaderStage.TESS_CONTROL)
                    .cull_mode(bz.CullMode.NONE, bz.FrontFace.COUNTER_CLOCKWISE)
                    .build(target))
        cmd = ctx.create_command_buffer()
        cmd.begin()
        cmd.begin_rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0])
        cmd.bind_pipeline(pipeline)
        cmd.bind_vertex_buffer(vbuf)
        cmd.draw(3)
        cmd.end_rendering(target)
        ctx.submit(cmd)
        return painted(target)

    assert draw(16) > draw(1) * 1.5
