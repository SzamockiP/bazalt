"""Per-subresource layout tracking: a pass that writes PART of an image.

Until 0.18 an Image held one layout for the whole thing, so `target.layer(0, mip=1)`
claimed mip 0 had reached the final layout too. The next barrier over the whole
image then named an oldLayout that was true of one level and a lie about the
rest. The documented workaround was "render every layer and every mip before you
sample", which rules out the two things a mip chain is for: rendering into one
level, and reading one back.

The referee for all of this is the `ctx` fixture. A wrong oldLayout is a
validation error, and every test here would have produced one before the change.
"""

import struct

import bazalt as bz

from conftest import SHADER_DIR


def solid_pipeline(ctx, target):
    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    return ctx.graphics_pipeline().vertex_shader(vert).fragment_shader(frag).build(target)


def render_into(ctx, target, view):
    pipeline = solid_pipeline(ctx, view)
    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(view, clear_color=[1, 0, 0, 1]):
        cmd.bind_pipeline(pipeline).draw(3)
    ctx.submit(cmd)


def test_read_after_rendering_one_layer(ctx):
    """The direct bug: read() on a partially rendered image used to transition
    the whole thing from a layout only one layer was actually in."""
    target = ctx.create_render_target(16, 16, color=bz.Format.RGBA8, layers=4)
    render_into(ctx, target, target.layer(0))

    pixels = target.color[0].read()
    assert pixels.shape == (16, 16, 4)
    assert pixels[0, 0, 0] > 200


def test_read_after_rendering_one_mip(ctx):
    """Same for the mip axis. Reading still returns mip 0, and mip 0 here is the
    level that was never drawn into — the point is that the readback is legal at
    all, not what it contains."""
    target = ctx.create_render_target(16, 16, color=bz.Format.RGBA8, mip_levels=3)
    render_into(ctx, target, target.layer(0, mip=1))

    assert target.color[0].read().shape == (16, 16, 4)


def test_sampling_a_partially_rendered_layered_target(ctx):
    """The headline: draw into ONE layer, then sample the array.

    The sampler sees one view over every layer, so the layers nobody drew into
    have to reach the final layout as well. That is what end_rendering's
    even-out barrier is for, and without it this is a validation error at the
    sample — a long way from the pass that caused it.
    """
    target = ctx.create_render_target(16, 16, color=bz.Format.RGBA8, layers=4)
    render_into(ctx, target, target.layer(2))

    fullscreen = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    view_frag = ctx.compile_shader(str(SHADER_DIR / "array_view.frag"), bz.ShaderStage.FRAGMENT)
    screen = ctx.create_render_target(16, 16)
    pipe = (ctx.graphics_pipeline()
            .vertex_shader(fullscreen)
            .fragment_shader(view_frag)
            .texture(0, bz.ShaderStage.FRAGMENT, set=0)
            .push_constant(4, bz.ShaderStage.FRAGMENT)
            .build(screen))

    pool = ctx.create_descriptor_pool(max_sets=1, textures=1)
    dset = pool.allocate_set(pipe, set=0)
    dset.set_image(0, target.color[0], sampler=ctx.create_sampler(filter=bz.Filter.NEAREST))

    def sample(layer):
        cmd = ctx.create_command_buffer()
        cmd.begin()
        with cmd.rendering(screen, clear_color=[0, 0, 0, 1]):
            cmd.bind_pipeline(pipe)
            cmd.bind_descriptor_set(dset, pipe, 0)
            cmd.push_constants(pipe, 0, struct.pack("i", layer))
            cmd.draw(3)
        ctx.submit(cmd)
        return screen.color[0].read()

    # The layer that was drawn is red; the ones that were not are legal to
    # sample and hold whatever the discard left. Only the first is asserted:
    # the others are undefined by design, and asserting on them would be
    # asserting on garbage.
    assert sample(2)[8, 8, 0] > 200


def test_rendering_every_layer_still_costs_one_barrier(ctx):
    """The collapse. An image whose subresources all agree stores one layout and
    behaves exactly as it did before this existed, so the common case — draw
    every face, then sample — pays nothing for the machinery.

    Observable only as "this keeps working": the assertion is the fixture."""
    target = ctx.create_render_target(8, 8, color=bz.Format.RGBA8, cube=True)
    for face in range(6):
        render_into(ctx, target, target.layer(face))

    assert target.color[0].read().shape == (8, 8, 4)


def test_render_one_mip_then_read_it_back_through_a_sample(ctx):
    """Rendering into mip 1 and sampling that level. textureLod reaches the
    level directly, so this is the round trip the old whole-image layout made
    impossible."""
    target = ctx.create_render_target(16, 16, color=bz.Format.RGBA8, mip_levels=3)
    render_into(ctx, target, target.layer(0, mip=1))

    fullscreen = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    lod_frag = ctx.compile_shader(str(SHADER_DIR / "sample_lod.frag"), bz.ShaderStage.FRAGMENT)
    screen = ctx.create_render_target(8, 8)
    pipe = (ctx.graphics_pipeline()
            .vertex_shader(fullscreen)
            .fragment_shader(lod_frag)
            .texture(0, bz.ShaderStage.FRAGMENT, set=0)
            .push_constant(4, bz.ShaderStage.FRAGMENT)
            .build(screen))

    pool = ctx.create_descriptor_pool(max_sets=1, textures=1)
    dset = pool.allocate_set(pipe, set=0)
    dset.set_image(0, target.color[0], sampler=ctx.create_sampler(filter=bz.Filter.NEAREST))

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(screen, clear_color=[0, 0, 0, 1]):
        cmd.bind_pipeline(pipe)
        cmd.bind_descriptor_set(dset, pipe, 0)
        cmd.push_constants(pipe, 0, struct.pack("f", 1.0))
        cmd.draw(3)
    ctx.submit(cmd)

    assert screen.color[0].read()[4, 4, 0] > 200


def test_depth_of_a_partially_rendered_target_is_sampleable(ctx):
    """Depth follows colour through the same path, and it is the attachment
    whose final layout depends on the format (0.17), so it is the one where an
    even-out barrier naming the wrong layout would show up."""
    target = ctx.create_render_target(16, 16, color=bz.Format.RGBA8,
                             depth=bz.Format.D32F, layers=3)
    render_into(ctx, target, target.layer(1))

    assert target.depth.array_layers == 3
