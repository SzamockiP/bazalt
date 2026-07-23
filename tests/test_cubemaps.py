"""Cubemaps and texture arrays: layered images (arrayLayers > 1).

One VkImage, two views — CUBE for sampling (samplerCube) and 2D_ARRAY for
compute storage writes (a CUBE view is illegal as storage). Headless like the
rest of the suite: build a layered image from numpy or fill it in compute, then
sample or read it back and assert. The multi-layer barriers (the
GENERAL->SHADER_READ_ONLY transition across all six faces especially) are also
audited by the sync-validation-as-assert `ctx` fixture, not just the numbers.
"""

import struct

import numpy as np
import pytest

import bazalt as bz

from conftest import SHADER_DIR

# Vulkan cube layer order: +X, -X, +Y, -Y, +Z, -Z. FACE_DIRS[i] samples face i.
FACE_DIRS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


@pytest.fixture
def fullscreen_vert(ctx):
    return ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)


def _cube_sampler(ctx, fullscreen_vert, target):
    """A graphics pipeline that samples a bound cubemap in a pushed direction."""
    frag = ctx.compile_shader(str(SHADER_DIR / "skybox.frag"), bz.ShaderStage.FRAGMENT)
    return (ctx.graphics_pipeline()
            .vertex_shader(fullscreen_vert)
            .fragment_shader(frag)
            .texture(0, bz.ShaderStage.FRAGMENT, set=0)
            .push_constant(16, bz.ShaderStage.FRAGMENT)
            .build(target))


# ── data path: a list of numpy arrays -> a layered image ───────────────────


def test_texture_array_from_arrays(ctx):
    """A list of arrays builds a texture array; read() returns layer 0."""
    layers = [np.full((8, 8, 4), v, np.uint8) for v in (10, 20, 30)]
    arr = ctx.create_image(layers)

    assert arr.array_layers == 3
    assert not arr.is_cube
    px = arr.read()
    assert px.shape == (8, 8, 4)
    assert np.all(px == 10)  # layer 0


def test_cubemap_from_arrays_samples_every_face(ctx, fullscreen_vert):
    """Six arrays build a cubemap; sampling each axis direction returns that
    face's colour — proving all six layers uploaded, in the right order."""
    faces = [np.full((8, 8, 4), (i * 40, 0, 0, 255), np.uint8) for i in range(6)]
    cube = ctx.create_image(faces, cube=True)
    assert cube.is_cube and cube.array_layers == 6

    target = bz.RenderTarget(ctx, 8, 8)
    pipeline = _cube_sampler(ctx, fullscreen_vert, target)
    pool = ctx.create_descriptor_pool(max_sets=1, samplers=1)
    dset = pool.allocate_set(pipeline, set=0)
    dset.set_image(0, cube)

    for i, direction in enumerate(FACE_DIRS):
        cmd = ctx.create_command_buffer()
        cmd.begin()
        cmd.begin_rendering(target)
        cmd.bind_pipeline(pipeline)
        cmd.bind_descriptor_set(dset, pipeline, set=0)
        cmd.push_constants(pipeline, 0, struct.pack("4f", *direction, 0.0))
        cmd.draw(3)
        cmd.end_rendering(target)
        ctx.submit(cmd)

        got = target.read_pixels()[4, 4, :3]
        assert np.allclose(got, [i * 40, 0, 0], atol=2), f"face {i}: {got}"


# ── empty + compute: a storage image fills the layers on the GPU ───────────


def test_empty_texture_array_filled_by_compute(ctx):
    """An empty texture array is a valid compute storage target (2D_ARRAY);
    a dispatch writes every layer and read() sees layer 0."""
    comp = ctx.compile_shader(str(SHADER_DIR / "store_array.comp"), bz.ShaderStage.COMPUTE)
    pipeline = ctx.compute_pipeline().shader(comp).storage_image(0).build()

    arr = ctx.create_image(16, 16, bz.Format.RGBA8, layers=4)
    pool = ctx.create_descriptor_pool(max_sets=1, storage_images=1)
    dset = pool.allocate_set(pipeline, set=0)
    dset.set_storage_image(0, arr)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.bind_pipeline(pipeline)
    cmd.bind_descriptor_set(dset, pipeline, set=0)
    cmd.dispatch(2, 2, 4)  # 16/8 x 16/8 x layers
    ctx.submit(cmd)

    # Layer 0: red = 0/5 = 0, green 0.25, blue 0.5 (UNORM * 255).
    assert np.allclose(arr.read()[8, 8], [0, 64, 128, 255], atol=2), arr.read()[8, 8]


def test_empty_cubemap_compute_filled_then_sampled(ctx, fullscreen_vert):
    """The full chain: an empty cubemap is written face-by-face in compute
    (through its 2D_ARRAY storage view), auto-transitioned across all six
    layers to SHADER_READ_ONLY, and sampled as a samplerCube — each face gets
    the colour compute wrote to the matching layer."""
    comp = ctx.compile_shader(str(SHADER_DIR / "store_array.comp"), bz.ShaderStage.COMPUTE)
    fill = ctx.compute_pipeline().shader(comp).storage_image(0).build()

    cube = ctx.create_image(16, 16, bz.Format.RGBA8, cube=True)
    assert cube.is_cube and cube.array_layers == 6

    target = bz.RenderTarget(ctx, 8, 8)
    sampler_pipe = _cube_sampler(ctx, fullscreen_vert, target)

    pool = ctx.create_descriptor_pool(max_sets=2, storage_images=1, samplers=1)
    fill_set = pool.allocate_set(fill, set=0)
    fill_set.set_storage_image(0, cube)
    sample_set = pool.allocate_set(sampler_pipe, set=0)
    sample_set.set_image(0, cube)

    for i, direction in enumerate(FACE_DIRS):
        # Fill + sample in one recording so the auto-barrier hoists the
        # GENERAL->SHADER_READ_ONLY transition (all six layers) before the pass.
        cmd = ctx.create_command_buffer()
        cmd.begin()
        cmd.bind_pipeline(fill)
        cmd.bind_descriptor_set(fill_set, fill, set=0)
        cmd.dispatch(2, 2, 6)
        cmd.begin_rendering(target)
        cmd.bind_pipeline(sampler_pipe)
        cmd.bind_descriptor_set(sample_set, sampler_pipe, set=0)
        cmd.push_constants(sampler_pipe, 0, struct.pack("4f", *direction, 0.0))
        cmd.draw(3)
        cmd.end_rendering(target)
        ctx.submit(cmd)

        # Face i red = i/5 * 255; green 0.25, blue 0.5.
        expected = [round(i / 5 * 255), 64, 128]
        assert np.allclose(target.read_pixels()[4, 4, :3], expected, atol=2), \
            f"face {i}: {target.read_pixels()[4, 4, :3]}"


# ── bake once, sample later: the manual image barrier ──────────────────────


def test_compute_baked_cubemap_sampled_in_a_later_submit(ctx, fullscreen_vert):
    """cmd.barrier(image, SHADER_WRITE, SHADER_READ) transitions a compute-baked
    cubemap to sampleable, so a LATER submit can sample it — bake once, sample
    many. The cross-submit layout is what the automatic tracker can't reach."""
    comp = ctx.compile_shader(str(SHADER_DIR / "store_array.comp"), bz.ShaderStage.COMPUTE)
    fill = ctx.compute_pipeline().shader(comp).storage_image(0).build()

    cube = ctx.create_image(16, 16, bz.Format.RGBA8, cube=True)
    fill_set = ctx.create_descriptor_pool(max_sets=1, storage_images=1).allocate_set(fill, set=0)
    fill_set.set_storage_image(0, cube)

    # Bake once: fill every face, then transition to sampleable — its own submit.
    bake = ctx.create_command_buffer()
    bake.begin()
    bake.bind_pipeline(fill).bind_descriptor_set(fill_set, fill, set=0).dispatch(2, 2, 6)
    bake.barrier(cube, bz.Access.SHADER_WRITE, bz.Access.SHADER_READ)
    ctx.submit(bake)

    # Sample in separate submits, no regeneration — each face keeps its colour.
    target = bz.RenderTarget(ctx, 8, 8)
    pipeline = _cube_sampler(ctx, fullscreen_vert, target)
    sample_set = ctx.create_descriptor_pool(max_sets=1, samplers=1).allocate_set(pipeline, set=0)
    sample_set.set_image(0, cube)

    for i, direction in enumerate(FACE_DIRS):
        cmd = ctx.create_command_buffer()
        cmd.begin()
        cmd.begin_rendering(target)
        cmd.bind_pipeline(pipeline).bind_descriptor_set(sample_set, pipeline, set=0)
        cmd.push_constants(pipeline, 0, struct.pack("4f", *direction, 0.0))
        cmd.draw(3)
        cmd.end_rendering(target)
        ctx.submit(cmd)
        expected = [round(i / 5 * 255), 64, 128]
        assert np.allclose(target.read_pixels()[4, 4, :3], expected, atol=2), \
            f"face {i}: {target.read_pixels()[4, 4, :3]}"


def test_image_barrier_rejects_non_shader_access(ctx):
    """The image barrier's layout follows the access; only the two shader
    accesses name an image layout."""
    cube = ctx.create_image(8, 8, bz.Format.RGBA8, cube=True)
    cmd = ctx.create_command_buffer()
    cmd.begin()
    with pytest.raises(bz.ResourceError):
        cmd.barrier(cube, bz.Access.VERTEX_READ, bz.Access.SHADER_READ)


def test_manual_barrier_then_auto_sample_in_one_recording(ctx, fullscreen_vert):
    """A manual cmd.barrier(image) must compose with an AUTOMATIC sample of the
    same image in ONE recording. The barrier seeds the tracker, so the auto path
    sees the post-barrier layout and emits no second transition; without that
    seeding it would re-transition from a stale GENERAL and the validation-as-
    assert ctx fixture would catch the layout mismatch."""
    comp = ctx.compile_shader(str(SHADER_DIR / "store_array.comp"), bz.ShaderStage.COMPUTE)
    fill = ctx.compute_pipeline().shader(comp).storage_image(0).build()
    cube = ctx.create_image(16, 16, bz.Format.RGBA8, cube=True)

    target = bz.RenderTarget(ctx, 8, 8)
    sampler_pipe = _cube_sampler(ctx, fullscreen_vert, target)
    pool = ctx.create_descriptor_pool(max_sets=2, storage_images=1, samplers=1)
    fill_set = pool.allocate_set(fill, set=0)
    fill_set.set_storage_image(0, cube)
    sample_set = pool.allocate_set(sampler_pipe, set=0)
    sample_set.set_image(0, cube)

    for i, direction in enumerate(FACE_DIRS):
        cmd = ctx.create_command_buffer()
        cmd.begin()
        cmd.bind_pipeline(fill).bind_descriptor_set(fill_set, fill, set=0).dispatch(2, 2, 6)
        # Manual barrier mid-recording, then an automatic sample of the same image.
        cmd.barrier(cube, bz.Access.SHADER_WRITE, bz.Access.SHADER_READ)
        cmd.begin_rendering(target)
        cmd.bind_pipeline(sampler_pipe).bind_descriptor_set(sample_set, sampler_pipe, set=0)
        cmd.push_constants(sampler_pipe, 0, struct.pack("4f", *direction, 0.0))
        cmd.draw(3)
        cmd.end_rendering(target)
        ctx.submit(cmd)

        expected = [round(i / 5 * 255), 64, 128]
        assert np.allclose(target.read_pixels()[4, 4, :3], expected, atol=2), \
            f"face {i}: {target.read_pixels()[4, 4, :3]}"


# ── validation: the cube flag and layer consistency are enforced ───────────


def test_cubemap_needs_exactly_six_faces(ctx):
    with pytest.raises(bz.ResourceError):
        ctx.create_image([np.zeros((8, 8, 4), np.uint8)] * 5, cube=True)


def test_cube_faces_must_be_square(ctx):
    with pytest.raises(bz.ResourceError):
        ctx.create_image(8, 16, bz.Format.RGBA8, cube=True)


def test_layers_must_share_shape(ctx):
    with pytest.raises(bz.ResourceError):
        ctx.create_image([np.zeros((8, 8, 4), np.uint8), np.zeros((4, 4, 4), np.uint8)])


def test_single_array_with_cube_is_rejected(ctx):
    with pytest.raises(bz.ResourceError):
        ctx.create_image(np.zeros((8, 8, 4), np.uint8), cube=True)
