"""3D textures (0.23): create, upload, read, update, mip, copy, blit, sample.

A volume is one more axis on the machinery 2D images already have, so most of
these tests are the 2D contract restated with a depth: a round trip must be
byte-exact, a mip chain must agree with full_mip_count, the region form of
update must fit or refuse. The two shader tests are the pictures the feature
exists for — a compute image3D fill and a fragment sampler3D read — with the
validation-as-assert fixture as the barrier referee.
"""

import numpy as np
import pytest

import bazalt as bz
from conftest import SHADER_DIR


def volume_rgba8(d, h, w, seed=0):
    """A (d, h, w, 4) uint8 volume where every texel value is predictable."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(d, h, w, 4), dtype=np.uint8)


# ── creation and round trip ───────────────────────────────────────────────


def test_empty_volume_reports_its_geometry(ctx):
    vol = ctx.create_image(16, 8, bz.Format.RGBA8, depth=4)
    assert (vol.width, vol.height, vol.depth) == (16, 8, 4)
    assert vol.array_layers == 1
    assert not vol.is_cube


def test_a_2d_image_reports_depth_1(ctx):
    img = ctx.create_image(8, 8, bz.Format.RGBA8)
    assert img.depth == 1


def test_uint8_volume_round_trips_exactly(ctx):
    pixels = volume_rgba8(4, 8, 16)
    vol = ctx.create_image(pixels)
    assert (vol.width, vol.height, vol.depth) == (16, 8, 4)
    out = vol.read()
    assert out.shape == (4, 8, 16, 4)
    assert np.array_equal(out, pixels)


def test_single_channel_float_volume_round_trips(ctx):
    pixels = np.linspace(0, 1, 4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)[..., None]
    vol = ctx.create_image(pixels)
    out = vol.read()
    # Single-channel reads back without the trailing axis, like 2D does.
    assert out.shape == (4, 4, 4)
    assert np.allclose(out, pixels[..., 0])


# ── refusals ──────────────────────────────────────────────────────────────


def test_depth_with_layers_is_refused(ctx):
    with pytest.raises(bz.ResourceError, match="one array layer|cannot have layers"):
        ctx.create_image(8, 8, bz.Format.RGBA8, depth=4, layers=2)


def test_depth_with_cube_is_refused(ctx):
    with pytest.raises(bz.ResourceError, match="cubemap"):
        ctx.create_image(8, 8, bz.Format.RGBA8, depth=4, cube=True)


def test_a_3dim_array_still_means_channels(ctx):
    """(d, h, w) raw volumes must not silently become 64-channel 2D images;
    the message teaches the arr[..., None] spelling."""
    with pytest.raises(bz.ResourceError, match=r"arr\[\.\.\., None\]"):
        ctx.create_image(np.zeros((64, 32, 32), dtype=np.float32))


def test_a_list_of_volumes_is_refused(ctx):
    with pytest.raises(bz.ResourceError, match="ONE"):
        ctx.create_image([volume_rgba8(2, 4, 4), volume_rgba8(2, 4, 4)])


def test_three_channel_volume_is_refused_with_the_padding_hint(ctx):
    with pytest.raises(bz.ResourceError, match="channel"):
        ctx.create_image(np.zeros((4, 4, 4, 3), dtype=np.uint8))


# ── mipmaps ───────────────────────────────────────────────────────────────


def test_volume_mip_chain_counts_the_depth_axis(ctx):
    """A 1x1x64 volume has 7 levels: depth is an axis of the chain, not a
    passenger."""
    vol = ctx.create_image(1, 1, bz.Format.RGBA8, depth=64, mip_levels=7)
    assert vol.mip_levels == 7
    with pytest.raises(bz.ResourceError, match="mip_levels"):
        ctx.create_image(1, 1, bz.Format.RGBA8, depth=64, mip_levels=8)


def test_volume_mipmaps_halve_the_depth(ctx):
    """Mip 1 of a constant 4x4x4 volume is 2x2x2 with the same value — the blit
    averages equal texels, so any drift is a wrong region, not filtering."""
    pixels = np.full((4, 4, 4, 4), 200, dtype=np.uint8)
    vol = ctx.create_image(pixels, mipmaps=True)
    assert vol.mip_levels == 3
    mip1 = vol.read(mip=1)
    assert mip1.shape == (2, 2, 2, 4)
    assert np.allclose(mip1, 200, atol=1)


# ── update ────────────────────────────────────────────────────────────────


def test_update_replaces_the_whole_volume(ctx):
    vol = ctx.create_image(volume_rgba8(4, 4, 4, seed=1))
    replacement = volume_rgba8(4, 4, 4, seed=2)
    vol.update(replacement)
    assert np.array_equal(vol.read(), replacement)


def test_update_writes_a_3d_region(ctx):
    base = np.zeros((4, 4, 4, 4), dtype=np.uint8)
    vol = ctx.create_image(base)
    patch = np.full((2, 2, 2, 4), 255, dtype=np.uint8)
    vol.update(patch, region=(1, 1, 1, 2, 2, 2))
    out = vol.read()
    assert np.all(out[1:3, 1:3, 1:3] == 255)
    assert np.all(out[0, :, :] == 0) and np.all(out[:, 0, :] == 0) and np.all(out[:, :, 0] == 0)


def test_update_region_out_of_the_volume_is_refused(ctx):
    vol = ctx.create_image(volume_rgba8(4, 4, 4))
    with pytest.raises(bz.ResourceError, match="does not fit"):
        vol.update(np.zeros((2, 2, 2, 4), dtype=np.uint8), region=(3, 0, 3, 2, 2, 2))


def test_update_region_length_follows_the_image_kind(ctx):
    """A 4-tuple on a volume and a 6-tuple on a 2D image are each malformed on
    their own — ValueError, the 0.20 rule."""
    vol = ctx.create_image(volume_rgba8(4, 4, 4))
    with pytest.raises(ValueError):
        vol.update(np.zeros((2, 2, 4), dtype=np.uint8), region=(0, 0, 2, 2))
    img = ctx.create_image(np.zeros((4, 4, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        img.update(np.zeros((2, 2, 4), dtype=np.uint8), region=(0, 0, 0, 2, 2, 1))


# ── copy and blit ─────────────────────────────────────────────────────────


def test_copy_image_carries_every_volume_mip(ctx):
    pixels = np.full((4, 4, 4, 4), 120, dtype=np.uint8)
    src = ctx.create_image(pixels, mipmaps=True)
    dst = ctx.create_image(4, 4, bz.Format.RGBA8, depth=4, mip_levels=3)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.copy_image(src, dst)
    ctx.submit(cmd)

    assert np.array_equal(dst.read(), pixels)
    assert np.allclose(dst.read(mip=2), 120, atol=1)


def test_copy_between_2d_and_3d_is_refused(ctx):
    vol = ctx.create_image(volume_rgba8(4, 4, 4))
    img = ctx.create_image(np.zeros((4, 4, 4), dtype=np.uint8))
    cmd = ctx.create_command_buffer()
    cmd.begin()
    with pytest.raises(bz.ResourceError, match="match in size"):
        cmd.copy_image(vol, img)


def test_blit_scales_a_volume_into_a_volume(ctx):
    src = ctx.create_image(np.full((4, 4, 4, 4), 80, dtype=np.uint8))
    dst = ctx.create_image(2, 2, bz.Format.RGBA8, depth=2)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.blit_image(src, dst)
    ctx.submit(cmd)

    assert np.allclose(dst.read(), 80, atol=1)


def test_blit_between_2d_and_3d_is_refused(ctx):
    vol = ctx.create_image(volume_rgba8(4, 4, 4))
    img = ctx.create_image(np.zeros((8, 8, 4), dtype=np.uint8))
    cmd = ctx.create_command_buffer()
    cmd.begin()
    with pytest.raises(bz.ResourceError, match="3D"):
        cmd.blit_image(vol, img)


# ── shaders: the pictures the feature exists for ──────────────────────────


def test_fragment_shader_samples_the_right_slice(ctx):
    """Distinct color per slice, NEAREST sampling at each slice's center: the
    pixel that comes back is the slice that went in."""
    depth = 8
    pixels = np.zeros((depth, 4, 4, 4), dtype=np.uint8)
    for z in range(depth):
        pixels[z, :, :, 0] = z * 30
        pixels[z, :, :, 3] = 255
    vol = ctx.create_image(pixels)

    target = bz.RenderTarget(ctx, 8, 8)
    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "sample_volume.frag"), bz.ShaderStage.FRAGMENT)
    pipe = (ctx.graphics_pipeline()
            .vertex_shader(vert)
            .fragment_shader(frag)
            .texture(0, bz.ShaderStage.FRAGMENT, set=0)
            .push_constant(4, bz.ShaderStage.FRAGMENT)
            .build(target))

    sampler = ctx.create_sampler(filter=bz.Filter.NEAREST)
    pool = ctx.create_descriptor_pool(max_sets=1, textures=1)
    dset = pool.allocate_set(pipe, set=0)
    dset.set_image(0, vol, sampler)

    for z in (0, 3, 7):
        cmd = ctx.create_command_buffer()
        cmd.begin()
        cmd.begin_rendering(target, clear_color=[0, 0, 0, 1])
        cmd.bind_pipeline(pipe)
        cmd.bind_descriptor_set(dset, pipe, set=0)
        cmd.push_constants(pipe, 0, np.float32((z + 0.5) / depth).tobytes())
        cmd.draw(3)
        cmd.end_rendering(target)
        ctx.submit(cmd)
        got = target.color[0].read()[4, 4]
        assert np.allclose(got, [z * 30, 0, 0, 255], atol=2), f"slice {z}: {got}"


def test_compute_fills_a_volume_through_image3d(ctx):
    """imageStore into a 3D storage image, then read back: the coordinate
    pattern must land at the coordinates that wrote it. The auto-barrier
    between the dispatch and the readback is what validation referees."""
    comp = ctx.compile_shader(str(SHADER_DIR / "fill_volume.comp"), bz.ShaderStage.COMPUTE)
    pipe = ctx.compute_pipeline().shader(comp).storage_image(0).build()

    vol = ctx.create_image(8, 8, bz.Format.RGBA8, depth=8)
    pool = ctx.create_descriptor_pool(max_sets=1, storage_images=1)
    dset = pool.allocate_set(pipe, set=0)
    dset.set_storage_image(0, vol)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.bind_pipeline(pipe)
    cmd.bind_descriptor_set(dset, pipe, set=0)
    cmd.dispatch(2, 2, 2)
    ctx.submit(cmd)

    out = vol.read()
    assert out.shape == (8, 8, 8, 4)
    # out[z, y, x] = (x/7, y/7, z/7, 1) in UNORM.
    assert np.allclose(out[5, 3, 6], [int(6 / 7 * 255), int(3 / 7 * 255), int(5 / 7 * 255), 255], atol=2)
    assert np.allclose(out[0, 0, 0], [0, 0, 0, 255], atol=2)
