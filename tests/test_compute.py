"""Compute needs no images: dispatch -> SSBO -> numpy assert.

This is why compute tests are the first thing CI on a software rasterizer
runs — no golden images, just arithmetic.
"""

import struct

import numpy as np
import pytest

import bazalt as bz

from conftest import SHADER_DIR


def test_compute_shader_compiles(ctx):
    comp = ctx.compile_shader(str(SHADER_DIR / "double.comp"), bz.ShaderStage.COMPUTE)
    assert comp is not None


def test_compute_pipeline_builds_without_a_target(ctx):
    comp = ctx.compile_shader(str(SHADER_DIR / "double.comp"), bz.ShaderStage.COMPUTE)
    pipeline = ctx.compute_pipeline().shader(comp).storage_buffer(0).build()
    assert pipeline is not None


def test_compute_pipeline_without_shader_raises(ctx):
    with pytest.raises(bz.ShaderError):
        ctx.compute_pipeline().build()


def test_bad_compute_shader_raises_shader_error(ctx, tmp_path):
    bad = tmp_path / "bad.comp"
    bad.write_text("#version 450\nlayout(local_size_x = 1) in;\nvoid main() { nonsense }\n")

    with pytest.raises(bz.ShaderError) as info:
        ctx.compile_shader(str(bad), bz.ShaderStage.COMPUTE)
    assert info.value.path == str(bad)


# ── dispatch → readback ───────────────────────────────────────────────────


def test_dispatch_doubles_a_storage_buffer(ctx):
    comp = ctx.compile_shader(str(SHADER_DIR / "double.comp"), bz.ShaderStage.COMPUTE)
    pipeline = ctx.compute_pipeline().shader(comp).storage_buffer(0).build()

    data = np.arange(128, dtype=np.float32)
    sbuf = ctx.create_buffer(data, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    pool = ctx.create_descriptor_pool(max_sets=8, storage_buffers=8)
    dset = pool.allocate_set(pipeline, set=0)
    dset.set_buffer(0, sbuf)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.bind_pipeline(pipeline)
    cmd.bind_descriptor_set(dset, pipeline, set=0)
    cmd.dispatch(2)  # 128 / local_size_x=64
    ctx.submit(cmd)

    # The headless submit waits idle, so no barrier is needed between the
    # dispatch and this readback.
    assert np.allclose(sbuf.read(np.float32), data * 2)


def test_push_constants_reach_a_compute_shader(ctx):
    comp = ctx.compile_shader(str(SHADER_DIR / "push_add.comp"), bz.ShaderStage.COMPUTE)
    pipeline = (ctx.compute_pipeline()
                .shader(comp)
                .storage_buffer(0)
                .push_constant(4)
                .build())

    data = np.arange(64, dtype=np.float32)
    sbuf = ctx.create_buffer(data, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    pool = ctx.create_descriptor_pool(max_sets=8, storage_buffers=8)
    dset = pool.allocate_set(pipeline, set=0)
    dset.set_buffer(0, sbuf)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.bind_pipeline(pipeline)
    cmd.bind_descriptor_set(dset, pipeline, set=0)
    cmd.push_constants(pipeline, 0, struct.pack("<f", 5.0))
    cmd.dispatch(1)
    ctx.submit(cmd)

    assert np.allclose(sbuf.read(np.float32), data + 5.0)


def test_uniform_buffer_reaches_a_compute_shader(ctx):
    comp = ctx.compile_shader(str(SHADER_DIR / "ubo_scale.comp"), bz.ShaderStage.COMPUTE)
    pipeline = (ctx.compute_pipeline()
                .shader(comp)
                .storage_buffer(0)
                .uniform_buffer(1)
                .build())

    data = np.arange(64, dtype=np.float32)
    sbuf = ctx.create_buffer(data, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    ubuf = ctx.create_buffer(np.array([3.0], dtype=np.float32),
                             bz.BufferType.UNIFORM, bz.MemoryUsage.DYNAMIC)
    pool = ctx.create_descriptor_pool(max_sets=8, storage_buffers=8, uniform_buffers=8)
    dset = pool.allocate_frame_set(pipeline, set=0)
    dset.set_buffer(0, sbuf)
    dset.set_buffer(1, ubuf)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.bind_pipeline(pipeline)
    cmd.bind_descriptor_set(dset, pipeline, set=0)
    cmd.dispatch(1)
    ctx.submit(cmd)

    assert np.allclose(sbuf.read(np.float32), data * 3.0)


# ── a sampled texture in compute (0.21) ───────────────────────────────────────


def test_a_compute_shader_can_sample_a_texture(ctx):
    """`texture()` on the compute builder. Until 0.21 a compute shader could only
    reach an image as a storage image — imageLoad by integer coordinate, with no
    filtering, no mip selection and no address mode. Nothing downstream was
    missing: set_image, the pool and the barrier tracker all handled a sampler
    binding on a compute set already.

    The test is two-sided through the filter rather than through a second
    pipeline: a 2x2 source magnified 16x with a LINEAR sampler must produce
    intermediate values, which is exactly what imageLoad cannot do."""
    src_pixels = np.zeros((2, 2, 4), dtype=np.uint8)
    src_pixels[0, 0, :3] = (255, 255, 255)
    src_pixels[0, 1, :3] = (0, 0, 0)
    src_pixels[1, 0, :3] = (0, 0, 0)
    src_pixels[1, 1, :3] = (255, 255, 255)
    src_pixels[:, :, 3] = 255
    src = ctx.create_image(src_pixels)
    dst = ctx.create_image(32, 32, bz.Format.RGBA8)

    comp = ctx.compile_shader(str(SHADER_DIR / "sample_texture.comp"), bz.ShaderStage.COMPUTE)
    pipe = (ctx.compute_pipeline()
            .shader(comp)
            .texture(0, set=0)
            .storage_image(1, set=0)
            .build())

    pool = ctx.create_descriptor_pool(max_sets=1, samplers=1, storage_images=1)
    dset = pool.allocate_set(pipe, set=0)
    dset.set_image(0, src, sampler=ctx.create_sampler(filter=bz.Filter.LINEAR,
                                                      address_mode=bz.AddressMode.CLAMP))
    dset.set_storage_image(1, dst)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.bind_pipeline(pipe)
    cmd.bind_descriptor_set(dset, pipe, set=0)
    cmd.dispatch(4, 4)
    ctx.submit(cmd)

    out = dst.read()[:, :, 0].astype(int)
    # The corners keep the source's own values, so the sampler read the image.
    assert out[1, 1] > 200 and out[1, 30] < 55
    # And the middle is interpolated: a value that is neither of the two the
    # source contains can only come from filtering.
    interpolated = int(np.count_nonzero((out > 40) & (out < 215)))
    assert interpolated > 100, f"only {interpolated} filtered pixels — imageLoad would give 0"


def test_update_after_bind_on_a_plain_binding(extra_context):
    """The opt-in half of update_after_bind. A single descriptor defaults to OFF,
    because a plain binding is written once at setup in nearly every program and
    the flag puts the descriptor in a separate budget. Naming it turns it on, and
    then rewriting the descriptor while a submit that binds the set is still in
    flight is legal instead of undefined.

    The referee is the validation-as-assert fixture: without the flag this is
    VUID-vkUpdateDescriptorSets-None-03047, confirmed by hand on the same
    sequence with the flag off."""
    ctx = extra_context(optional=[bz.Feature.BINDLESS])
    if not ctx.supports(bz.Feature.BINDLESS):
        pytest.skip("GPU reports no descriptorIndexing")

    def solid(rgb):
        px = np.zeros((4, 4, 4), dtype=np.uint8)
        px[:, :, :3] = rgb
        px[:, :, 3] = 255
        return ctx.create_image(px)

    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "textured.frag"), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, 32, 32)
    pipe = (ctx.graphics_pipeline()
            .vertex_shader(vert).fragment_shader(frag)
            .texture(0, bz.ShaderStage.FRAGMENT, set=0, update_after_bind=True)
            .build(target))

    pool = ctx.create_descriptor_pool(max_sets=1, samplers=4)
    dset = pool.allocate_set(pipe, set=0)
    dset.set_image(0, solid((255, 0, 0)))

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(target, clear_color=[0, 0, 0, 1]) as c:
        c.bind_pipeline(pipe).bind_descriptor_set(dset, pipe, set=0).draw(3)
    ctx.submit(cmd, wait=False)

    # The submit is still pending here, which is the whole point.
    dset.set_image(0, solid((0, 255, 0)))
    ctx.wait()

    ctx.submit(cmd)
    assert np.allclose(target.color[0].read()[16, 16, :3], (0, 255, 0), atol=2)


def test_update_after_bind_needs_the_feature(ctx):
    """Silently ignoring the request would leave the caller rewriting descriptors
    a submit still reads, which is the undefined behaviour they asked to be rid
    of."""
    if ctx.supports(bz.Feature.BINDLESS):
        pytest.skip("the session Context enabled BINDLESS after all")
    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "textured.frag"), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, 16, 16)
    with pytest.raises(bz.UnsupportedError, match="BINDLESS"):
        (ctx.graphics_pipeline()
         .vertex_shader(vert).fragment_shader(frag)
         .texture(0, bz.ShaderStage.FRAGMENT, set=0, update_after_bind=True)
         .build(target))
