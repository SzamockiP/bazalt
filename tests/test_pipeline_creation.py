"""0.17: what pipeline creation used to hard-code.

Specialization constants (one module, several pipelines), the per-attachment
blend and colour mask, depth clamp and alpha-to-coverage. The pipeline cache is
here too — it changes no result, so what a test can prove is that turning it on
did not change one.

Headless, so CI covers it.
"""

import pathlib

import numpy as np
import pytest

import bazalt as bz

SHADER_DIR = pathlib.Path(__file__).parent / "shaders"


# ── specialization constants ──────────────────────────────────────────────


@pytest.fixture
def spec_pipeline(ctx):
    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "spec_color.frag"), bz.ShaderStage.FRAGMENT)

    def make(target, constants):
        builder = ctx.graphics_pipeline().vertex_shader(vert).fragment_shader(frag)
        for cid, value in constants.items():
            builder = builder.constant(cid, value, bz.ShaderStage.FRAGMENT)
        return builder.build(target)

    return make


def draw(ctx, target, pipeline):
    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0]) as c:
        c.bind_pipeline(pipeline).draw(3)
    ctx.submit(cmd)
    return target.read_pixels()[16, 16]


def test_one_shader_two_pipelines_two_results(ctx, spec_pipeline):
    """The same ShaderModule, specialized differently, paints differently."""
    target = bz.RenderTarget(ctx, 32, 32)

    assert np.allclose(draw(ctx, target, spec_pipeline(target, {0: 1.0, 1: 2}))[:3],
                       [255, 128, 0], atol=2)
    assert np.allclose(draw(ctx, target, spec_pipeline(target, {0: 0.25, 1: 4}))[:3],
                       [64, 255, 0], atol=2)


def test_the_defaults_in_the_shader_stand_when_nothing_is_specialized(ctx, spec_pipeline):
    target = bz.RenderTarget(ctx, 32, 32)
    assert np.allclose(draw(ctx, target, spec_pipeline(target, {}))[:3], [0, 64, 0], atol=2)


def test_a_bool_constant_is_not_an_int(ctx, spec_pipeline):
    """Python's bool IS an int, so the binding has to test it first; otherwise
    True arrives as the integer 1 and a `bool` constant reads the wrong thing."""
    target = bz.RenderTarget(ctx, 32, 32)
    assert draw(ctx, target, spec_pipeline(target, {2: True}))[2] == 255
    assert draw(ctx, target, spec_pipeline(target, {2: False}))[2] == 0


def test_a_constant_must_be_a_number(ctx, spec_pipeline):
    target = bz.RenderTarget(ctx, 32, 32)
    with pytest.raises(bz.ResourceError, match="bool, an int or a float"):
        spec_pipeline(target, {0: "one"})


def test_compute_takes_constants_without_a_stage(ctx):
    """The compute builder's constant() has no stage argument, for the same
    reason nothing else there does.

    The workgroup size itself is NOT specialized here: local_size_x_id compiles
    to OpExecutionMode LocalSizeId, which needs Vulkan 1.3 / maintenance4, and
    the baseline is 1.2.
    """
    shader = ctx.compile_shader(str(SHADER_DIR / "spec_add.comp"), bz.ShaderStage.COMPUTE)
    pipeline = (ctx.compute_pipeline()
                .shader(shader)
                .storage_buffer(0)
                .constant(1, 5)   # ADDEND
                .build())

    data = np.zeros(8, dtype=np.int32)
    buf = ctx.create_buffer(data, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    pool = ctx.create_descriptor_pool(max_sets=1, storage_buffers=1)
    dset = pool.allocate_set(pipeline, set=0)
    dset.set_buffer(0, buf)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.bind_pipeline(pipeline)
    cmd.bind_descriptor_set(dset, pipeline, set=0)
    cmd.dispatch(1)
    ctx.submit(cmd)

    assert np.array_equal(buf.read(np.int32), np.full(8, 5, dtype=np.int32))


def test_a_hot_reload_keeps_the_constants(ctx, spec_pipeline):
    """A rebuilt pipeline must re-apply the values: they are part of the
    pipeline, not of the SPIR-V, so a recompile cannot recover them."""
    target = bz.RenderTarget(ctx, 32, 32)
    pipeline = spec_pipeline(target, {0: 1.0, 1: 2})
    before = draw(ctx, target, pipeline)

    # The same path hot reload takes, without waiting for a file to change.
    ctx.begin_frame()
    assert np.array_equal(draw(ctx, target, pipeline), before)


# ── per-attachment blend and colour mask ──────────────────────────────────


@pytest.fixture
def mrt_pipeline(ctx):
    """Writes the same push-constant colour into two attachments."""
    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "mrt.frag"), bz.ShaderStage.FRAGMENT)

    def make(target, configure=lambda b: b):
        return configure(ctx.graphics_pipeline().vertex_shader(vert).fragment_shader(frag)).build(target)

    return make


def test_blend_can_differ_per_attachment(extra_context):
    """Additive on attachment 0, replace on attachment 1, one pipeline.

    Two passes: the second preserves, so an additive attachment accumulates over
    the first draw and a replacing one does not. Needs INDEPENDENT_BLEND — see
    the test below for what happens without it.
    """
    ctx = extra_context(optional=[bz.Feature.INDEPENDENT_BLEND])
    if not ctx.supports(bz.Feature.INDEPENDENT_BLEND):
        pytest.skip("this GPU has no independentBlend")

    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "mrt.frag"), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, 32, 32, color=[bz.Format.RGBA8, bz.Format.RGBA8])
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .blend(True, mode=bz.BlendMode.ADDITIVE, attachment=0)
                .build(target))

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0]) as c:
        c.bind_pipeline(pipeline).draw(3)
    with cmd.rendering(target, clear_color=None) as c:
        c.bind_pipeline(pipeline).draw(3)
    ctx.submit(cmd)

    # mrt.frag writes (0.25, 0.5, 0.75) into attachment 0 every pass: additive
    # doubles it, and the unblended attachment 1 keeps one pass worth.
    accumulated = target.color[0].read()[16, 16].astype(int)
    replaced = target.color[1].read()[16, 16].astype(int)
    assert np.allclose(accumulated[:3], [128, 255, 255], atol=3), (
        f"attachment 0 came out {accumulated} — additive did not accumulate")
    assert np.allclose(replaced[:3], [255, 0, 255], atol=3)


def test_a_differing_attachment_needs_independent_blend(ctx):
    """Without the feature every attachment must blend identically, and the
    driver is entitled to reject the pipeline — so bazalt says so first."""
    if ctx.supports(bz.Feature.INDEPENDENT_BLEND):
        pytest.skip("session Context happens to have INDEPENDENT_BLEND enabled")
    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "mrt.frag"), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, 32, 32, color=[bz.Format.RGBA8, bz.Format.RGBA8])
    with pytest.raises(bz.ShaderError, match="INDEPENDENT_BLEND"):
        (ctx.graphics_pipeline()
         .vertex_shader(vert)
         .fragment_shader(frag)
         .blend(True, mode=bz.BlendMode.ADDITIVE, attachment=0)
         .build(target))


def test_color_mask_drops_a_channel(ctx, mrt_pipeline):
    """Masking green off leaves the cleared value in that channel."""
    target = bz.RenderTarget(ctx, 32, 32, color=[bz.Format.RGBA8, bz.Format.RGBA8])

    unmasked = mrt_pipeline(target)
    # No attachment= — the mask applies to both, so this needs no feature.
    masked = mrt_pipeline(target, lambda b: b.color_mask(True, False, True, True))

    def run(pipeline):
        cmd = ctx.create_command_buffer()
        cmd.begin()
        with cmd.rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0]) as c:
            c.bind_pipeline(pipeline).draw(3)
        ctx.submit(cmd)
        return target.color[0].read()[16, 16].astype(int)

    full = run(unmasked)
    dropped = run(masked)
    assert dropped[1] == 0, "the masked channel was still written"
    assert full[0] == dropped[0] and full[2] == dropped[2], "an unmasked channel changed"


# ── rasterizer knobs that needed a Feature or a target ────────────────────


def test_depth_clamp_needs_its_feature(ctx):
    """The session Context asks for no optional features, so this is the
    message a user gets, not a driver-dependent pipeline."""
    if ctx.supports(bz.Feature.DEPTH_CLAMP):
        pytest.skip("session Context happens to have DEPTH_CLAMP enabled")
    target = bz.RenderTarget(ctx, 32, 32, color=bz.Format.RGBA8, depth=bz.Format.D32F)
    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "push.frag"), bz.ShaderStage.FRAGMENT)
    with pytest.raises(bz.ShaderError, match="DEPTH_CLAMP"):
        (ctx.graphics_pipeline()
         .vertex_shader(vert)
         .fragment_shader(frag)
         .push_constant(16, bz.ShaderStage.FRAGMENT)
         .depth_clamp(True)
         .build(target))


def test_depth_clamp_keeps_geometry_behind_the_near_plane(extra_context):
    """Two-sided: the same draw at z beyond the near plane is clipped away
    without the clamp and survives with it."""
    ctx = extra_context(optional=[bz.Feature.DEPTH_CLAMP])
    if not ctx.supports(bz.Feature.DEPTH_CLAMP):
        pytest.skip("this GPU has no depthClamp")

    target = bz.RenderTarget(ctx, 32, 32, color=bz.Format.RGBA8, depth=bz.Format.D32F)
    vert = ctx.compile_shader(str(SHADER_DIR / "behind_near.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "push.frag"), bz.ShaderStage.FRAGMENT)

    def run(clamp):
        pipeline = (ctx.graphics_pipeline()
                    .vertex_shader(vert)
                    .fragment_shader(frag)
                    .push_constant(16, bz.ShaderStage.FRAGMENT)
                    .depth_test(True)
                    .depth_clamp(clamp)
                    .build(target))
        cmd = ctx.create_command_buffer()
        cmd.begin()
        with cmd.rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0]) as c:
            import struct
            c.bind_pipeline(pipeline)
            c.push_constants(pipeline, 0, struct.pack("4f", 1.0, 0.0, 0.0, 1.0))
            c.draw(3)
        ctx.submit(cmd)
        return int(np.count_nonzero(target.read_pixels()[..., 0]))

    assert run(False) == 0, "the geometry was in front of the near plane after all"
    assert run(True) > 0, "depth_clamp did not keep the clipped geometry"


def test_alpha_to_coverage_softens_an_msaa_edge(ctx):
    """A half-alpha fullscreen draw covers half the samples, so the resolve
    lands mid-grey instead of full white."""
    if ctx.max_samples() < 4:
        pytest.skip("this GPU has no 4x MSAA")
    target = bz.RenderTarget(ctx, 32, 32, samples=4)
    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "push.frag"), bz.ShaderStage.FRAGMENT)

    def run(enable):
        pipeline = (ctx.graphics_pipeline()
                    .vertex_shader(vert)
                    .fragment_shader(frag)
                    .push_constant(16, bz.ShaderStage.FRAGMENT)
                    .alpha_to_coverage(enable)
                    .build(target))
        cmd = ctx.create_command_buffer()
        cmd.begin()
        with cmd.rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0]) as c:
            import struct
            c.bind_pipeline(pipeline)
            c.push_constants(pipeline, 0, struct.pack("4f", 1.0, 1.0, 1.0, 0.5))
            c.draw(3)
        ctx.submit(cmd)
        return int(target.read_pixels()[16, 16][0])

    assert run(False) == 255, "alpha changed the colour without coverage"
    assert 64 <= run(True) <= 192, "alpha_to_coverage did not drop any samples"


# ── the pipeline cache ────────────────────────────────────────────────────


def test_the_cache_changes_no_result(ctx):
    """It is on for every Context, so what is testable is that two builds of
    the same description still render identically — the second one comes out of
    the cache."""
    target = bz.RenderTarget(ctx, 32, 32)
    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "spec_color.frag"), bz.ShaderStage.FRAGMENT)

    def build():
        return (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .constant(0, 1.0, bz.ShaderStage.FRAGMENT)
                .build(target))

    assert np.array_equal(draw(ctx, target, build()), draw(ctx, target, build()))
