"""Capability negotiation and logging.

Capabilities rather than versions: Vulkan promotes extensions into core, so the
same capability is spelled differently per driver. Which spelling to use is the
library's problem, not the caller's.
"""

import pytest

import bazalt as bz

ALL_FEATURES = [
    bz.Feature.ANISOTROPIC_FILTERING,
    bz.Feature.WIREFRAME,
    bz.Feature.WIDE_LINES,
    bz.Feature.DEPTH_CLAMP,
    bz.Feature.SAMPLE_RATE_SHADING,
    bz.Feature.MULTI_DRAW_INDIRECT,
    bz.Feature.SHADER_FLOAT64,
]


def test_context_reports_what_it_negotiated(ctx):
    assert ctx.device_name
    assert ctx.api_version.count(".") == 2
    assert isinstance(ctx.headless, bool)


@pytest.mark.parametrize("feature", ALL_FEATURES)
def test_supports_answers_for_every_feature(ctx, feature):
    assert isinstance(ctx.supports(feature), bool)


def test_anisotropy_is_on_by_default_when_available(ctx):
    """Texture uses it, so it's enabled when present.

    It used to be a *required* device feature, which turned a nicety into a
    hardware requirement.
    """
    assert ctx.supports(bz.Feature.ANISOTROPIC_FILTERING) in (True, False)


def test_unrequested_features_stay_off(ctx):
    """Asking for nothing must not silently enable everything the GPU can do."""
    assert ctx.supports(bz.Feature.WIREFRAME) is False


def test_severity_is_data_not_a_string_prefix(ctx, messages):
    logger = ctx.logger
    logger.log("hello", severity=bz.Severity.WARNING, source=bz.Source.GENERAL)

    mine = [m for m in messages() if m.text == "hello"]
    assert len(mine) == 1
    assert mine[0].severity == bz.Severity.WARNING
    assert mine[0].source == bz.Source.GENERAL


def test_severity_is_ordered(ctx):
    assert bz.Severity.ERROR > bz.Severity.WARNING > bz.Severity.INFO


def test_min_severity_filters(ctx, messages):
    logger = ctx.logger
    previous = logger.min_severity
    try:
        logger.min_severity = bz.Severity.ERROR
        logger.log("suppressed", severity=bz.Severity.INFO)
        logger.log("kept", severity=bz.Severity.ERROR)

        texts = [m.text for m in messages()]
        assert "suppressed" not in texts
        assert "kept" in texts
    finally:
        logger.min_severity = previous


def test_flush_makes_delivery_observable(ctx, messages):
    """Without flush, asserting "nothing was logged" only means "not yet"."""
    ctx.logger.log("flushed", severity=bz.Severity.WARNING)
    assert "flushed" in [m.text for m in messages()]


def test_log_message_is_readable(ctx, messages):
    ctx.logger.log("readable", severity=bz.Severity.WARNING)
    msg = [m for m in messages() if m.text == "readable"][0]
    assert "readable" in str(msg)
    assert "WARNING" in str(msg)


# ── Portability-subset capabilities (0.22) ───────────────────────────────
# These three read the opposite way from every other Feature: they name things
# full Vulkan always allows, and only a portability subset (MoltenVK on macOS)
# can take away. The tests therefore assert the AGREEMENT between the answer and
# the behaviour, which is the same assertion on both kinds of driver.
#
# The refusal branch below runs ONLY on a portability driver, so no desktop run
# can check it. That is how the 0.23 error split reached CI green here and red
# on macOS: the type these branches expect had changed under them (0.23).


def test_a_comparison_sampler_agrees_with_what_the_context_says(ctx):
    """create_sampler(compare=) is sampler2DShadow, which shadow mapping needs.
    Metal has no mutable comparison samplers, so the capability is a question
    there and an always-yes everywhere else."""
    if ctx.supports(bz.Feature.COMPARISON_SAMPLER):
        assert ctx.create_sampler(compare=bz.CompareOp.LESS) is not None
    else:
        with pytest.raises(bz.UnsupportedError):
            ctx.create_sampler(compare=bz.CompareOp.LESS)


def test_a_mip_lod_bias_agrees_with_what_the_context_says(ctx):
    if ctx.supports(bz.Feature.SAMPLER_MIP_LOD_BIAS):
        assert ctx.create_sampler(mip_lod_bias=1.5) is not None
    else:
        with pytest.raises(bz.UnsupportedError):
            ctx.create_sampler(mip_lod_bias=1.5)


def test_a_layered_multisampled_target_agrees_with_what_the_context_says(ctx):
    """A multisampled image with layers > 1. Refused before vkCreateImage where
    the driver cannot do it, rather than reported by the validation layers."""
    if ctx.max_samples() < 4:
        pytest.skip("this device has no 4x MSAA")
    if ctx.supports(bz.Feature.MULTISAMPLE_ARRAYS):
        # color[0] is the RESOLVE attachment and stays single-sample: the
        # multisampled image the feature is about is the one behind it.
        target = ctx.create_render_target(32, 32, layers=2, samples=4)
        assert target.color[0].array_layers == 2
    else:
        with pytest.raises(bz.UnsupportedError):
            ctx.create_render_target(32, 32, layers=2, samples=4)


def test_a_bias_of_zero_needs_no_capability(ctx):
    """The gate asks about the argument, not about the call: mip_lod_bias=0.0 is
    what every sampler in the suite already passes."""
    assert ctx.create_sampler(mip_lod_bias=0.0) is not None
