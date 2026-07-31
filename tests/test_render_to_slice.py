"""Render-to-slice (0.23): target.layer(z) on a RenderTarget over a 3D image.

The slice view is a 2D view of a 3D image, which full Vulkan always allows and
a portability driver may not (Feature.IMAGE_VIEW_2D_ON_3D; MoltenVK answers
False). Every desktop driver and lavapipe answer True, so these tests run
everywhere the suite runs except MoltenVK CI — the gate names the observed
answer when it skips.

The layout consequence worth pinning: Vulkan tracks a volume's layout per mip
with exactly one array layer, so rendering ONE slice marks the WHOLE mip.
Sampling after a partial render is therefore legal (never a validation error);
the unwritten slices simply hold what they held.
"""

import numpy as np
import pytest

import bazalt as bz


@pytest.fixture(autouse=True)
def needs_slice_views(ctx):
    if not ctx.supports(bz.Feature.IMAGE_VIEW_2D_ON_3D):
        pytest.skip("ctx.supports(IMAGE_VIEW_2D_ON_3D) is False on this driver")


def test_each_slice_gets_its_own_clear(ctx):
    """A clear-only pass per slice: what comes back from read() proves each
    pass landed in the Z slice it named — the whole feature in one claim."""
    depth = 8
    vol = ctx.create_image(4, 4, bz.Format.RGBA8, depth=depth)
    target = bz.RenderTarget(ctx, color=[vol])

    for z in range(depth):
        cmd = ctx.create_command_buffer()
        cmd.begin()
        slice_target = target.layer(z)
        cmd.begin_rendering(slice_target, clear_color=[z * 30 / 255.0, 0, 0, 1])
        cmd.end_rendering(slice_target)
        ctx.submit(cmd)

    out = vol.read()
    for z in range(depth):
        assert np.allclose(out[z, :, :, 0], z * 30, atol=1), f"slice {z}"


def test_partial_slice_render_then_read_is_clean(ctx):
    """Rendering three slices of eight and reading the volume back must not
    trip validation: one slice render marks the whole mip, by the volume's own
    layout granularity."""
    vol = ctx.create_image(4, 4, bz.Format.RGBA8, depth=8)
    target = bz.RenderTarget(ctx, color=[vol])
    for z in range(3):
        cmd = ctx.create_command_buffer()
        cmd.begin()
        st = target.layer(z)
        cmd.begin_rendering(st, clear_color=[1, 1, 1, 1])
        cmd.end_rendering(st)
        ctx.submit(cmd)
    out = vol.read()
    assert np.all(out[:3, :, :, 0] == 255)


def test_slice_of_a_mip_scales_the_bound(ctx):
    """Level 1 of a depth-4 volume has 2 slices; slice 3 exists only at mip 0."""
    vol = ctx.create_image(8, 8, bz.Format.RGBA8, depth=4, mip_levels=2)
    target = bz.RenderTarget(ctx, color=[vol])
    st = target.layer(1, mip=1)
    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(st, clear_color=[0, 1, 0, 1])
    cmd.end_rendering(st)
    ctx.submit(cmd)
    with pytest.raises(bz.ResourceError, match="out of range"):
        target.layer(3, mip=1)


def test_whole_3d_target_is_refused_with_the_fix(ctx):
    vol = ctx.create_image(4, 4, bz.Format.RGBA8, depth=4)
    target = bz.RenderTarget(ctx, color=[vol])
    cmd = ctx.create_command_buffer()
    cmd.begin()
    with pytest.raises(bz.ResourceError, match=r"layer\(z\)"):
        cmd.begin_rendering(target, clear_color=[0, 0, 0, 1])


def test_all_layers_on_a_3d_target_is_refused(ctx):
    vol = ctx.create_image(4, 4, bz.Format.RGBA8, depth=4)
    target = bz.RenderTarget(ctx, color=[vol])
    with pytest.raises(bz.ResourceError, match="one array layer"):
        target.all_layers()


def test_depth_attachment_beside_a_volume_is_refused(ctx):
    """A 2D depth buffer is caught by the equal-Z rule; the dedicated guard
    behind it covers the exotic case of a depth-format volume."""
    vol = ctx.create_image(4, 4, bz.Format.RGBA8, depth=4)
    zbuf = ctx.create_image(4, 4, bz.Format.D32F)
    with pytest.raises(bz.ResourceError, match="deep"):
        bz.RenderTarget(ctx, color=[vol], depth=zbuf)


def test_mixing_2d_and_3d_attachments_is_refused(ctx):
    vol = ctx.create_image(4, 4, bz.Format.RGBA8, depth=4)
    flat = ctx.create_image(4, 4, bz.Format.RGBA8)
    with pytest.raises(bz.ResourceError, match="deep"):
        bz.RenderTarget(ctx, color=[vol, flat])


def test_msaa_on_a_volume_is_refused(ctx):
    vol = ctx.create_image(4, 4, bz.Format.RGBA8, depth=4)
    with pytest.raises(bz.ResourceError, match="3D"):
        bz.RenderTarget(ctx, color=[vol], samples=4)
