"""Image + Sampler + Format: the 0.5 split of the old Texture.

An Image is pixels + format; a Sampler is how to read them, cached on the
Context. Formats are one table (bytes per pixel, channels, numpy dtype), so
uploads and readbacks agree by construction.
"""

import pathlib

import numpy as np
import pytest

import bazalt as bz

SHADER_DIR = pathlib.Path(__file__).parent / "shaders"


# ── create_image(array) round trips ───────────────────────────────────────


@pytest.mark.parametrize("dtype,channels,fmt", [
    (np.uint8, 4, bz.Format.RGBA8),
    (np.uint8, 2, bz.Format.RG8),
    (np.uint8, 1, bz.Format.R8),
    (np.float16, 4, bz.Format.RGBA16F),
    (np.float32, 1, bz.Format.R32F),
    (np.float32, 4, bz.Format.RGBA32F),
])
def test_array_round_trips_through_the_gpu(ctx, dtype, channels, fmt):
    rng = np.random.default_rng(seed=7)
    shape = (5, 3) if channels == 1 else (5, 3, channels)
    if np.issubdtype(dtype, np.integer):
        arr = rng.integers(0, 256, size=shape, dtype=dtype)
    else:
        arr = rng.random(size=shape).astype(dtype)

    img = ctx.create_image(arr)
    assert (img.width, img.height) == (3, 5)
    assert img.format == fmt
    assert img.mip_levels == 1  # data images get no surprise mips

    back = img.read()
    assert back.dtype == arr.dtype
    assert back.shape == arr.shape
    np.testing.assert_array_equal(back, arr)


def test_three_channel_arrays_are_refused_with_a_padding_hint(ctx):
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(bz.ResourceError) as info:
        ctx.create_image(arr)
    assert "pad" in str(info.value)


def test_strided_arrays_are_refused(ctx):
    arr = np.zeros((8, 8, 4), dtype=np.uint8)
    with pytest.raises(bz.ResourceError) as info:
        ctx.create_image(arr[::2])
    assert "ascontiguousarray" in str(info.value)


def test_unsupported_dtype_is_refused(ctx):
    with pytest.raises(bz.ResourceError):
        ctx.create_image(np.zeros((4, 4, 4), dtype=np.int32))


# ── mipmaps: opt-in for arrays, opt-out for files ─────────────────────────


def test_array_mipmaps_opt_in(ctx):
    """create_image(array, mipmaps=True) builds the full chain; the default
    stays single-level (data images get no surprise mips) and mip 0 round-trips
    either way."""
    arr = np.full((16, 16, 4), (200, 100, 50, 255), np.uint8)

    assert ctx.create_image(arr).mip_levels == 1
    mipped = ctx.create_image(arr, mipmaps=True)
    assert mipped.mip_levels == 5  # full chain for 16x16: 16,8,4,2,1
    np.testing.assert_array_equal(mipped.read(), arr)  # mip 0 unchanged


def test_load_image_mipmaps_can_be_turned_off(ctx, tmp_path):
    """Files are mipped by default (pictures); mipmaps=False keeps one level."""
    from test_bindings import write_png

    png = tmp_path / "img.png"
    write_png(png, [[(255, 0, 0, 255)] * 16] * 16)

    assert ctx.load_image(str(png)).mip_levels == 5
    assert ctx.load_image(str(png), mipmaps=False).mip_levels == 1


# ── empty images ──────────────────────────────────────────────────────────


def test_reading_a_virgin_image_is_refused(ctx):
    """Same contract read_pixels() has had since 0.4.1: no contents, no read —
    never driver-defined garbage."""
    img = ctx.create_image(16, 16, format=bz.Format.RGBA8)
    assert not img.ready  # nothing was ever uploaded
    with pytest.raises(bz.ResourceError):
        img.read()


def test_create_image_defaults_to_rgba8(ctx):
    img = ctx.create_image(4, 4)
    assert img.format == bz.Format.RGBA8


# ── samplers ──────────────────────────────────────────────────────────────


def test_identical_sampler_descriptions_share_one_object(ctx):
    """The cache is observable: same description, same Python object."""
    a = ctx.create_sampler()
    b = ctx.create_sampler()
    assert a is b

    c = ctx.create_sampler(filter=bz.Filter.NEAREST)
    assert c is not a
    assert ctx.create_sampler(filter=bz.Filter.NEAREST) is c


def test_sampling_with_an_explicit_nearest_sampler(ctx, tmp_path):
    """set_image with a non-default sampler renders without complaint.

    2x2 texture, NEAREST, sampled at the quadrant centres — the same pixels
    the LINEAR path yields at texel centres, so the assert is exact.
    """
    from test_bindings import write_png  # stdlib-only PNG writer

    red, green = (255, 0, 0, 255), (0, 255, 0, 255)
    blue, white = (0, 0, 255, 255), (255, 255, 255, 255)
    png_path = tmp_path / "quad.png"
    write_png(png_path, [[red, green], [blue, white]])
    tex = ctx.load_image(str(png_path))

    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "textured.frag"), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, 62, 62)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .texture(0, bz.ShaderStage.FRAGMENT, set=0)
                .build(target))

    pool = ctx.create_descriptor_pool(max_sets=4, samplers=4)
    dset = pool.allocate_set(pipeline, set=0)
    dset.set_image(0, tex, sampler=ctx.create_sampler(filter=bz.Filter.NEAREST))

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(target, clear_color=[0, 0, 0, 1])
    cmd.bind_pipeline(pipeline)
    cmd.bind_descriptor_set(dset, pipeline, set=0)
    cmd.draw(3)
    cmd.end_rendering(target)
    ctx.submit(cmd)

    pixels = target.read_pixels()
    assert np.allclose(pixels[15, 15, :3], red[:3], atol=2)
    assert np.allclose(pixels[46, 46, :3], white[:3], atol=2)


# ── upload future stubs ───────────────────────────────────────────────────


def test_uploaded_images_are_ready_and_wait_returns(ctx):
    """Trivially true while uploads are synchronous; the async UploadManager
    inherits this exact surface."""
    img = ctx.create_image(np.zeros((2, 2, 4), dtype=np.uint8))
    assert img.ready
    img.wait()
