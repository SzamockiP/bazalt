"""A RenderTarget on images you already own (0.19).

`bz.RenderTarget(ctx, color=[image])` renders into images from `create_image`
instead of attachments the target allocates for itself. There is no width or
height in that signature, and no samples, layers, cube or mip_levels either: every
one of them is a property of the images now, and a second answer could disagree
with them.

Each test below is a shape that was impossible while a target insisted on owning
its attachments — not a variation on one shape:

  1. a graphics ping-pong, where a pass samples what the previous pass drew,
  2. drawing over a texture a compute pass baked,
  3. drawing into an image brought from another Context.

The automatic barriers are the interesting part of all three, and the
validation-as-assert fixture is what checks them: a borrowed image goes into the
tracker exactly like an allocated one, so `end_rendering` leaves it in
SHADER_READ_ONLY and the sample that follows needs no `cmd.barrier()`.
"""

import numpy as np
import pytest

import bazalt as bz

from conftest import SHADER_DIR


@pytest.fixture
def fullscreen_vert(ctx):
    return ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)


def textured_pipeline(ctx, target, fullscreen_vert):
    """A fullscreen pass that samples binding 0 and writes it out."""
    frag = ctx.compile_shader(str(SHADER_DIR / "textured.frag"), bz.ShaderStage.FRAGMENT)
    return (ctx.graphics_pipeline()
            .vertex_shader(fullscreen_vert)
            .fragment_shader(frag)
            .texture(0, bz.ShaderStage.FRAGMENT, set=0)
            .build(target))


def solid_pipeline(ctx, target, fullscreen_vert):
    frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    return (ctx.graphics_pipeline()
            .vertex_shader(fullscreen_vert)
            .fragment_shader(frag)
            .build(target))


# ── the shapes that were unreachable ──────────────────────────────────────────

def test_a_pass_samples_what_the_previous_pass_drew(ctx, fullscreen_vert):
    """The graphics ping-pong. Two owned images, two targets, one recording: the
    first pass paints image A, the second samples A while drawing into B.

    Nothing here calls cmd.barrier(). The transition from colour attachment to
    sampled texture is the RenderTarget contract, and a borrowed image is tracked
    the same way an allocated one is — which is the whole claim of the feature.
    """
    a = ctx.create_image(64, 64, bz.Format.RGBA8, name="ping")
    b = ctx.create_image(64, 64, bz.Format.RGBA8, name="pong")
    target_a = bz.RenderTarget(ctx, color=[a])
    target_b = bz.RenderTarget(ctx, color=[b])

    paint = solid_pipeline(ctx, target_a, fullscreen_vert)
    copy = textured_pipeline(ctx, target_b, fullscreen_vert)

    sampler = ctx.create_sampler()
    pool = ctx.create_descriptor_pool(max_sets=4, samplers=4)
    dset = pool.allocate_set(copy, set=0)
    dset.set_image(0, a, sampler)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(target_a, clear_color=[0.0, 0.0, 0.0, 1.0]) as c:
        c.bind_pipeline(paint)
        c.draw(3)
    with cmd.rendering(target_b, clear_color=[0.0, 0.0, 0.0, 1.0]) as c:
        c.bind_pipeline(copy)
        c.bind_descriptor_set(dset, copy, set=0)
        c.draw(3)
    ctx.submit(cmd)
    ctx.wait()

    # solid_red.frag paints (255, 0, 0); B holds what it sampled out of A.
    px = b.read()
    assert px[32, 32, 0] == 255
    assert px[32, 32, 1] == 0
    assert px[32, 32, 2] == 0


def test_drawing_over_a_compute_baked_texture(ctx, fullscreen_vert):
    """Compute writes the image, then a graphics pass draws INTO the same image.

    Before this, a compute-baked texture could be sampled but never drawn over: a
    target allocated its own attachments, so there was no way to name an existing
    image as one. The overwrite is what proves the target is really rendering into
    that image and not into a copy of it.
    """
    img = ctx.create_image(64, 64, bz.Format.RGBA8, name="baked")

    comp = ctx.compile_shader(str(SHADER_DIR / "store_const.comp"), bz.ShaderStage.COMPUTE)
    bake = (ctx.compute_pipeline()
            .shader(comp)
            .storage_image(0, set=0)
            .build())
    pool = ctx.create_descriptor_pool(max_sets=4, storage_images=4)
    dset = pool.allocate_set(bake, set=0)
    dset.set_storage_image(0, img)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.bind_pipeline(bake)
    cmd.bind_descriptor_set(dset, bake, set=0)
    cmd.dispatch(8, 8)
    ctx.submit(cmd)
    ctx.wait()

    # store_const.comp writes (0.25, 0.5, 0.75, 1.0).
    baked = img.read()
    assert baked[32, 32, 1] == pytest.approx(128, abs=2)

    target = bz.RenderTarget(ctx, color=[img])
    paint = solid_pipeline(ctx, target, fullscreen_vert)
    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0]) as c:
        c.bind_pipeline(paint)
        c.draw(3)
    ctx.submit(cmd)
    ctx.wait()

    over = img.read()
    assert over[32, 32, 0] == 255
    assert over[32, 32, 1] == 0


def test_drawing_into_an_image_from_another_context(ctx, extra_context, fullscreen_vert):
    """An image carried across Contexts, then used as an attachment.

    ctx.create_image(other_image) has existed since 0.15, but the result could only
    ever be sampled. Rendering into it needed a target that does not allocate.
    """
    other = extra_context()
    source = other.create_image(np.full((32, 32, 4), 40, dtype=np.uint8), name="from elsewhere")
    other.wait()

    carried = ctx.create_image(source)
    target = bz.RenderTarget(ctx, color=[carried])
    paint = solid_pipeline(ctx, target, fullscreen_vert)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0]) as c:
        c.bind_pipeline(paint)
        c.draw(3)
    ctx.submit(cmd)
    ctx.wait()

    px = carried.read()
    assert px[16, 16, 0] == 255


def test_an_image_from_another_context_is_refused_as_an_attachment(ctx, extra_context):
    """The cross-Context guard, in the binding layer like every other one: a
    bazalt error naming the mistake instead of a driver crash."""
    other = extra_context()
    foreign = other.create_image(32, 32, bz.Format.RGBA8)

    with pytest.raises(bz.ResourceError, match="different Context"):
        bz.RenderTarget(ctx, color=[foreign])
    with pytest.raises(bz.ResourceError, match="different Context"):
        bz.RenderTarget(ctx, depth=foreign)


# ── the properties the images decide ──────────────────────────────────────────

def test_size_layers_and_mips_come_off_the_images(ctx):
    """No width, height, layers or mip_levels in this signature, so the target
    reports what the images say."""
    img = ctx.create_image(128, 64, bz.Format.RGBA8, mip_levels=3)
    target = bz.RenderTarget(ctx, color=[img])

    assert (target.width, target.height) == (128, 64)
    # The subresource machinery reads the same numbers, so layer/mip slicing works
    # on a borrowed image exactly as on an allocated one.
    assert target.layer(0, mip=2) is not None
    with pytest.raises(bz.ResourceError):
        target.layer(0, mip=3)


def test_the_target_holds_the_images_it_borrows(ctx, fullscreen_vert):
    """Dropping the Python reference must not take the attachment with it.

    The target holds each image by shared_ptr exactly as it holds the ones it
    allocates, so the only thing keeping this honest is that ownership was never
    special-cased. If it were, this would be a use-after-free rather than a
    failed assertion.
    """
    img = ctx.create_image(64, 64, bz.Format.RGBA8)
    target = bz.RenderTarget(ctx, color=[img])
    paint = solid_pipeline(ctx, target, fullscreen_vert)

    del img

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0]) as c:
        c.bind_pipeline(paint)
        c.draw(3)
    ctx.submit(cmd)
    ctx.wait()

    assert target.color[0].read()[32, 32, 0] == 255


def test_mismatched_attachments_are_refused(ctx):
    """One extent, one layer count and one mip count for the whole target, because
    the render area, the viewport and every subresource view come from a single set
    of numbers. Reported rather than silently intersected."""
    a = ctx.create_image(64, 64, bz.Format.RGBA8)
    small = ctx.create_image(32, 32, bz.Format.RGBA8)
    mipped = ctx.create_image(64, 64, bz.Format.RGBA8, mip_levels=2)
    layered = ctx.create_image(64, 64, bz.Format.RGBA8, layers=2)

    with pytest.raises(bz.ResourceError, match="same size"):
        bz.RenderTarget(ctx, color=[a, small])
    with pytest.raises(bz.ResourceError, match="same mip count"):
        bz.RenderTarget(ctx, color=[a, mipped])
    with pytest.raises(bz.ResourceError, match="same layer count"):
        bz.RenderTarget(ctx, color=[a, layered])


def test_a_depth_image_in_color_is_refused_and_the_other_way_round(ctx):
    colour = ctx.create_image(64, 64, bz.Format.RGBA8)
    depth = ctx.create_image(64, 64, bz.Format.D32F)

    with pytest.raises(bz.ResourceError, match="cannot go in color"):
        bz.RenderTarget(ctx, color=[depth])
    with pytest.raises(bz.ResourceError, match="depth format"):
        bz.RenderTarget(ctx, depth=colour)
    with pytest.raises(bz.ResourceError, match="at least one attachment"):
        bz.RenderTarget(ctx)


def test_the_two_signatures_do_not_get_confused(ctx):
    """pybind picks between the two __init__ overloads on arity, and neither can
    fall through to the other once matched — so each says what to do instead."""
    img = ctx.create_image(64, 64, bz.Format.RGBA8)

    # Images plus a size: the allocating overload matched, and cannot hand off.
    with pytest.raises(bz.ResourceError, match="drop width and height"):
        bz.RenderTarget(ctx, 64, 64, color=[img])
    with pytest.raises(bz.ResourceError, match="drop width and height"):
        bz.RenderTarget(ctx, 64, 64, color=img)

    # Formats with no size: the borrowing overload matched, same problem mirrored.
    with pytest.raises(bz.ResourceError, match="no width and height"):
        bz.RenderTarget(ctx, color=[bz.Format.RGBA8])


def test_a_depth_attachment_from_an_owned_image_works(ctx, fullscreen_vert):
    """Depth is borrowed the same way colour is, and the layout it retires to is
    derived from its format — the 0.17 rule, unchanged by where the image came
    from."""
    colour = ctx.create_image(64, 64, bz.Format.RGBA8)
    depth = ctx.create_image(64, 64, bz.Format.D32F)
    target = bz.RenderTarget(ctx, color=[colour], depth=depth)

    frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(fullscreen_vert)
                .fragment_shader(frag)
                .depth_test(True)
                .build(target))

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0]) as c:
        c.bind_pipeline(pipeline)
        c.draw(3)
    ctx.submit(cmd)
    ctx.wait()

    assert colour.read()[32, 32, 0] == 255
    # The depth image is sampleable afterwards, so it really was transitioned.
    assert depth.read().shape[:2] == (64, 64)
