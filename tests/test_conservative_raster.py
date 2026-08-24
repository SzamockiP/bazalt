"""Conservative rasterization (0.30): the pipeline knob and what it changes.

The referee here is a pixel count rather than validation alone, because the
whole point of the feature is a primitive that ordinary rasterization DROPS.
`shaders/sliver.vert` draws a triangle inside one pixel and away from that
pixel's centre: OFF paints nothing, OVERESTIMATE paints the pixel. A test that
only built the pipeline would pass on a driver that ignored the knob.

Every test that needs the capability asks for its own Context, because the
session one negotiates no optional features on purpose. Skips are expected in
CI: MoltenVK has no VK_EXT_conservative_rasterization at all.
"""

import numpy as np
import pytest

import bazalt as bz

from conftest import SHADER_DIR


@pytest.fixture
def cons_ctx(extra_context):
    """A Context with CONSERVATIVE_RASTER, or a skip where the GPU has none."""
    context = extra_context(optional=[bz.Feature.CONSERVATIVE_RASTER])
    if not context.supports(bz.Feature.CONSERVATIVE_RASTER):
        pytest.skip("GPU lacks VK_EXT_conservative_rasterization")
    return context


def _painted(ctx, mode, extra=0.0):
    """Pixels the sliver covers, under one rasterization mode."""
    vert = ctx.compile_shader(str(SHADER_DIR / "sliver.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    target = ctx.create_render_target(16, 16)
    builder = (ctx.graphics_pipeline()
               .vertex_shader(vert)
               .fragment_shader(frag)
               # The winding of a sub-pixel triangle is not the subject here,
               # and culling it would hide the result behind a second reason.
               .cull_mode(bz.CullMode.NONE))
    if mode is not bz.ConservativeRaster.OFF:
        builder = builder.conservative_raster(mode, extra)
    pipeline = builder.build(target)

    g = ctx.graph()
    with g.add_pass(target, clear_color=[0.0, 0.0, 0.0, 1.0]) as p:
        p.bind_pipeline(pipeline)
        p.draw(3)
    ctx.submit(g)
    return int(np.count_nonzero(target.color[0].read()[:, :, :3].any(axis=2)))


# ── What the mode actually changes ────────────────────────────────────────


def test_the_sliver_misses_every_sample_without_conservative_raster(cons_ctx):
    """The premise the rest of the file rests on. Without it a green
    OVERESTIMATE test would prove nothing: a triangle the normal path already
    draws is drawn either way."""
    assert _painted(cons_ctx, bz.ConservativeRaster.OFF) == 0


def test_overestimate_paints_the_pixel_the_sliver_touches(cons_ctx):
    assert _painted(cons_ctx, bz.ConservativeRaster.OVERESTIMATE) > 0


def test_underestimate_drops_the_sliver_too(cons_ctx):
    """The other direction: a triangle that does not CONTAIN a whole pixel
    produces nothing, which is what makes an underestimated pass a guarantee."""
    if not cons_ctx.limits.conservative_underestimation:
        pytest.skip("GPU reports primitiveUnderestimation false")
    assert _painted(cons_ctx, bz.ConservativeRaster.UNDERESTIMATE) == 0


def test_extra_overestimation_does_not_shrink_the_coverage(cons_ctx):
    """A dilation on top of OVERESTIMATE. Asserted as "no less" rather than
    "more", because one granularity step on a triangle this small may not reach
    the next pixel — the claim under test is that the driver takes the value at
    all, and the layers are the referee for whether it was legal."""
    granularity = cons_ctx.limits.extra_overestimation_granularity
    if granularity <= 0.0 or cons_ctx.limits.max_extra_overestimation < granularity:
        pytest.skip("GPU allows no extra overestimation")
    plain = _painted(cons_ctx, bz.ConservativeRaster.OVERESTIMATE)
    assert _painted(cons_ctx, bz.ConservativeRaster.OVERESTIMATE, granularity) >= plain


# ── What build() refuses ──────────────────────────────────────────────────


def test_conservative_raster_needs_the_feature(ctx, fullscreen_vert):
    """The knob is per pipeline, but the extension is per device."""
    if ctx.supports(bz.Feature.CONSERVATIVE_RASTER):
        pytest.skip("session Context happens to have CONSERVATIVE_RASTER enabled")
    frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    target = ctx.create_render_target(16, 16)
    with pytest.raises(bz.UnsupportedError, match="CONSERVATIVE_RASTER"):
        (ctx.graphics_pipeline()
         .vertex_shader(fullscreen_vert)
         .fragment_shader(frag)
         .conservative_raster(bz.ConservativeRaster.OVERESTIMATE)
         .build(target))


def test_extra_overestimation_above_the_maximum_is_refused(cons_ctx):
    """Refused rather than clamped: the driver would silently take the maximum,
    and a grid built on a dilation it did not get is wrong with nothing said."""
    vert = cons_ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = cons_ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    target = cons_ctx.create_render_target(16, 16)
    too_much = cons_ctx.limits.max_extra_overestimation + 1.0
    with pytest.raises(bz.UnsupportedError, match="extra_overestimation"):
        (cons_ctx.graphics_pipeline()
         .vertex_shader(vert)
         .fragment_shader(frag)
         .conservative_raster(bz.ConservativeRaster.OVERESTIMATE, too_much)
         .build(target))


def test_extra_overestimation_without_overestimate_is_refused(cons_ctx):
    """There is no area to grow when the mode is OFF or UNDERESTIMATE, so the
    call is a mistake rather than a no-op."""
    vert = cons_ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = cons_ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    target = cons_ctx.create_render_target(16, 16)
    with pytest.raises(bz.ShaderError, match="extra_overestimation"):
        (cons_ctx.graphics_pipeline()
         .vertex_shader(vert)
         .fragment_shader(frag)
         .conservative_raster(bz.ConservativeRaster.OFF, 0.25)
         .build(target))


def test_underestimate_is_refused_where_the_property_says_no(cons_ctx):
    """A property, not a feature, so the Context could not negotiate it away —
    build() is the only place that can answer."""
    if cons_ctx.limits.conservative_underestimation:
        pytest.skip("GPU supports primitiveUnderestimation")
    vert = cons_ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = cons_ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    target = cons_ctx.create_render_target(16, 16)
    with pytest.raises(bz.UnsupportedError, match="primitiveUnderestimation"):
        (cons_ctx.graphics_pipeline()
         .vertex_shader(vert)
         .fragment_shader(frag)
         .conservative_raster(bz.ConservativeRaster.UNDERESTIMATE)
         .build(target))


def test_the_device_reports_the_same_answer_as_the_context(cons_ctx):
    """`Device.supports()` is what a program picks a GPU with, so it must agree
    with what the Context it then builds reports."""
    matching = [d for d in bz.list_devices() if d.name == cons_ctx.device_name]
    assert matching
    assert matching[0].supports(bz.Feature.CONSERVATIVE_RASTER) is True
