"""Descriptor arrays — bindless (0.21).

`texture(binding, stage, set, count=N)` declares one binding holding N textures,
`set_image(binding, image, index=i)` writes element i, and the shader picks one
per draw or per fragment. One pipeline and one draw then serve many materials.

Every test builds its own Context: BINDLESS is an optional device feature that
has to be asked for at device creation, and the session Context does not ask.
`extra_context` applies the same validation-as-assert referee, which is what
actually checks the layout flags — a missing PARTIALLY_BOUND or
UPDATE_AFTER_BIND is a validation error, not a wrong pixel.

The pixel tests are all "each slot shows ITS OWN texture", never "something was
drawn": an array that silently samples slot 0 everywhere still renders.
"""

import struct

import numpy as np
import pytest

import bazalt as bz

from conftest import SHADER_DIR

# Four flat colours, one per array slot. Distinct in every channel so a
# read-back pixel names the slot it came from with no arithmetic.
SLOT_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
]


def bindless_context(extra_context):
    ctx = extra_context(optional=[bz.Feature.BINDLESS])
    if not ctx.supports(bz.Feature.BINDLESS):
        pytest.skip("GPU reports no descriptorIndexing")
    return ctx


def solid(ctx, rgb):
    """A 4x4 image of one colour."""
    pixels = np.zeros((4, 4, 4), dtype=np.uint8)
    pixels[:, :, :3] = rgb
    pixels[:, :, 3] = 255
    return ctx.create_image(pixels)


def array_pipeline(ctx, target, frag_name, push_bytes=0, count=4):
    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / frag_name), bz.ShaderStage.FRAGMENT)
    builder = (ctx.graphics_pipeline()
               .vertex_shader(vert)
               .fragment_shader(frag)
               .texture(0, bz.ShaderStage.FRAGMENT, set=0, count=count))
    if push_bytes:
        builder = builder.push_constant(push_bytes, bz.ShaderStage.FRAGMENT)
    return builder.build(target)


def draw_with(ctx, target, pipeline, dset, push=None):
    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(target, clear_color=[0, 0, 0, 1]) as c:
        c.bind_pipeline(pipeline)
        c.bind_descriptor_set(dset, pipeline, set=0)
        if push is not None:
            c.push_constants(pipeline, 0, push)
        c.draw(3)
    ctx.submit(cmd)
    return target.color[0].read()


# ── the array works ───────────────────────────────────────────────────────────


def test_each_slot_holds_its_own_texture(extra_context):
    """The push-constant index is the same for every invocation, so this is the
    dynamically uniform case. Four draws, four indices, four colours — and the
    pixel has to be the colour of the slot that was asked for, which is what
    separates a working array from one that always reads element 0."""
    ctx = bindless_context(extra_context)
    target = bz.RenderTarget(ctx, 32, 32)
    pipeline = array_pipeline(ctx, target, "bindless_push.frag", push_bytes=4)

    images = [solid(ctx, rgb) for rgb in SLOT_COLORS]
    pool = ctx.create_descriptor_pool(max_sets=1, samplers=len(images))
    dset = pool.allocate_set(pipeline, set=0)
    for i, image in enumerate(images):
        dset.set_image(0, image, index=i)

    for i, rgb in enumerate(SLOT_COLORS):
        pixels = draw_with(ctx, target, pipeline, dset, push=struct.pack("i", i))
        assert np.allclose(pixels[16, 16, :3], rgb, atol=2), f"slot {i}: {pixels[16, 16]}"


def test_one_draw_samples_four_textures(extra_context):
    """The non-uniform case, and the reason bindless exists: the index differs
    per fragment, so ONE draw paints four different textures. Needs
    nonuniformEXT in the shader and shaderSampledImageArrayNonUniformIndexing on
    the device, both of which arrive with BINDLESS."""
    ctx = bindless_context(extra_context)
    target = bz.RenderTarget(ctx, 32, 32)
    pipeline = array_pipeline(ctx, target, "bindless_quadrant.frag")

    images = [solid(ctx, rgb) for rgb in SLOT_COLORS]
    pool = ctx.create_descriptor_pool(max_sets=1, samplers=len(images))
    dset = pool.allocate_set(pipeline, set=0)
    for i, image in enumerate(images):
        dset.set_image(0, image, index=i)

    pixels = draw_with(ctx, target, pipeline, dset)
    # uv (0,0) is the top-left, and row 0 of the read-back is the top row.
    quadrants = {0: (8, 8), 1: (8, 24), 2: (24, 8), 3: (24, 24)}
    for slot, (row, col) in quadrants.items():
        assert np.allclose(pixels[row, col, :3], SLOT_COLORS[slot], atol=2), \
            f"quadrant {slot} at ({row},{col}): {pixels[row, col]}"


def test_a_partially_written_array_is_legal(extra_context):
    """Slots 1 and 3 are never written. Nobody samples them either, which is
    what VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT makes legal — without the
    flag, reading the set at all is undefined and the layers say so. The
    fixture is the assertion here; the pixel only proves the draw happened."""
    ctx = bindless_context(extra_context)
    target = bz.RenderTarget(ctx, 32, 32)
    pipeline = array_pipeline(ctx, target, "bindless_push.frag", push_bytes=4)

    pool = ctx.create_descriptor_pool(max_sets=1, samplers=2)
    try:
        dset = pool.allocate_set(pipeline, set=0)
    except bz.ResourceError as exc:
        # An under-sized pool for a partially bound array is legal Vulkan and
        # MoltenVK still refuses it (VK_ERROR_FRAGMENTED_POOL): Metal reserves
        # the whole argument-buffer slot count whether or not you write it. The
        # sibling test above covers the fully written array, so partial binding
        # keeps its positive case on every driver.
        pytest.skip(f"this driver needs pool room for the whole array: {exc}")
    dset.set_image(0, solid(ctx, SLOT_COLORS[0]), index=0)
    dset.set_image(0, solid(ctx, SLOT_COLORS[2]), index=2)

    for slot in (0, 2):
        pixels = draw_with(ctx, target, pipeline, dset, push=struct.pack("i", slot))
        assert np.allclose(pixels[16, 16, :3], SLOT_COLORS[slot], atol=2)


def test_a_slot_can_be_rewritten_while_a_draw_is_in_flight(extra_context):
    """Update-after-bind, and the submit has to be asynchronous for the test to
    mean anything: rewriting a descriptor of a set that a PENDING command buffer
    binds is what VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT makes legal
    (VUID-vkUpdateDescriptorSets-None-03047), and a blocking submit has already
    finished by the time the next line runs. Swapping a texture at run time is
    the prototyping case, so it has to be legal rather than usually working.

    The referee is the fixture, not the pixel: without the flag this is a
    validation error."""
    ctx = bindless_context(extra_context)
    target = bz.RenderTarget(ctx, 32, 32)
    pipeline = array_pipeline(ctx, target, "bindless_push.frag", push_bytes=4)

    pool = ctx.create_descriptor_pool(max_sets=1, samplers=8)
    dset = pool.allocate_set(pipeline, set=0)
    dset.set_image(0, solid(ctx, SLOT_COLORS[0]), index=0)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(target, clear_color=[0, 0, 0, 1]) as c:
        c.bind_pipeline(pipeline).bind_descriptor_set(dset, pipeline, set=0)
        c.push_constants(pipeline, 0, struct.pack("i", 0)).draw(3)
    ctx.submit(cmd, wait=False)

    dset.set_image(0, solid(ctx, SLOT_COLORS[1]), index=0)
    ctx.wait()

    second = draw_with(ctx, target, pipeline, dset, push=struct.pack("i", 0))
    assert np.allclose(second[16, 16, :3], SLOT_COLORS[1], atol=2), second[16, 16]


def test_rewriting_a_slot_does_not_grow_the_set(extra_context):
    """The same (binding, index) written many times is ONE descriptor, so the
    set must hold one image for it, not one per write. Before 0.21 every write
    appended: the old images stayed alive for the set's whole life and the
    record-time tracker walked a list that only grew. Asserted through
    behaviour — the last write is what gets sampled, and the images the writes
    replaced are collectable."""
    ctx = bindless_context(extra_context)
    target = bz.RenderTarget(ctx, 32, 32)
    pipeline = array_pipeline(ctx, target, "bindless_push.frag", push_bytes=4)

    pool = ctx.create_descriptor_pool(max_sets=1, samplers=64)
    dset = pool.allocate_set(pipeline, set=0)

    import weakref
    replaced = []
    for rgb in SLOT_COLORS:
        image = solid(ctx, rgb)
        replaced.append(weakref.ref(image))
        dset.set_image(0, image, index=0)
        del image

    pixels = draw_with(ctx, target, pipeline, dset, push=struct.pack("i", 0))
    assert np.allclose(pixels[16, 16, :3], SLOT_COLORS[-1], atol=2)
    # Every image but the last one lost its only owner when it was replaced.
    assert [r() for r in replaced[:-1]] == [None, None, None]


# ── refusals ──────────────────────────────────────────────────────────────────


def test_index_outside_the_declared_count_is_refused(extra_context):
    ctx = bindless_context(extra_context)
    target = bz.RenderTarget(ctx, 32, 32)
    pipeline = array_pipeline(ctx, target, "bindless_push.frag", push_bytes=4)
    pool = ctx.create_descriptor_pool(max_sets=1, samplers=4)
    dset = pool.allocate_set(pipeline, set=0)

    with pytest.raises(bz.ResourceError) as e:
        dset.set_image(0, solid(ctx, SLOT_COLORS[0]), index=4)
    assert "count=4" in str(e.value)


def test_index_on_a_plain_binding_is_refused(extra_context):
    """count defaults to 1, so index=1 is out of range on an ordinary texture —
    the same check, and the reason it needs no separate rule."""
    ctx = bindless_context(extra_context)
    target = bz.RenderTarget(ctx, 32, 32)
    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "textured.frag"), bz.ShaderStage.FRAGMENT)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .texture(0, bz.ShaderStage.FRAGMENT, set=0)
                .build(target))
    pool = ctx.create_descriptor_pool(max_sets=1, samplers=1)
    dset = pool.allocate_set(pipeline, set=0)

    with pytest.raises(bz.ResourceError):
        dset.set_image(0, solid(ctx, SLOT_COLORS[0]), index=1)


def test_two_counts_for_one_binding_fail_at_build(extra_context):
    """Declaring a binding twice is how a resource read by two stages is
    spelled, so the stages merge. Two different counts cannot merge: the layout
    holds one number and one of the two is wrong."""
    ctx = bindless_context(extra_context)
    target = bz.RenderTarget(ctx, 32, 32)
    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "bindless_push.frag"), bz.ShaderStage.FRAGMENT)

    with pytest.raises(bz.ShaderError) as e:
        (ctx.graphics_pipeline()
         .vertex_shader(vert)
         .fragment_shader(frag)
         .texture(0, bz.ShaderStage.FRAGMENT, set=0, count=4)
         .texture(0, bz.ShaderStage.VERTEX, set=0, count=8)
         .push_constant(4, bz.ShaderStage.FRAGMENT)
         .build(target))
    assert "different counts" in str(e.value)


def test_count_without_the_feature_is_refused(ctx):
    """The session Context never asked for BINDLESS, which makes it the right
    place to check the gate. count>1 is core Vulkan for a dynamically uniform
    index, and it is still refused: without descriptorIndexing an unwritten slot
    and a per-fragment index are both undefined, so the unguarded version works
    here and returns garbage on the next machine."""
    if ctx.supports(bz.Feature.BINDLESS):
        pytest.skip("the session Context enabled BINDLESS after all")
    target = bz.RenderTarget(ctx, 32, 32)
    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "bindless_push.frag"), bz.ShaderStage.FRAGMENT)

    with pytest.raises(bz.UnsupportedError) as e:
        (ctx.graphics_pipeline()
         .vertex_shader(vert)
         .fragment_shader(frag)
         .texture(0, bz.ShaderStage.FRAGMENT, set=0, count=4)
         .push_constant(4, bz.ShaderStage.FRAGMENT)
         .build(target))
    assert "BINDLESS" in str(e.value)
