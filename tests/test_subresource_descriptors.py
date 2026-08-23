"""set_image / set_storage_image with layer= and mip=.

The sampling twin of target.layer(i, mip=), which the render side has spelled
since 0.13. A descriptor took a whole Image until 0.30, so a pass could not
read level N of a chain it was writing level N-1 of — a bloom pyramid and a
prefiltered environment map are both that shape.

The hard case is the pyramid, and it is the reason the tracker splits per
(layer, mip): ONE pass holds mip N-1 in SHADER_READ_ONLY while it writes mip N
in GENERAL, which one layout per image can never say. The referee is the
validation fixture, plus the pixels.
"""

import numpy as np
import pytest

import bazalt as bz

from conftest import SHADER_DIR

LEVELS = [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)]


def _mipped_image(ctx, size=32, levels=LEVELS):
    """An image whose every mip level holds a different flat colour, so a
    sample can say which level it read."""
    image = ctx.create_image(size, size, bz.Format.RGBA8, mip_levels=len(levels),
                             name="levels")
    for mip, colour in enumerate(levels):
        extent = max(size >> mip, 1)
        image.update(np.tile(np.array(colour, np.uint8), (extent, extent, 1)), mip=mip)
    image.wait()
    return image


def _fullscreen_sampler(ctx, target):
    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "textured.frag"), bz.ShaderStage.FRAGMENT)
    return (ctx.graphics_pipeline().vertex_shader(vert).fragment_shader(frag)
            .texture(0, bz.ShaderStage.FRAGMENT).build(target))


def _sample(ctx, image, *, layer=None, mip=None, size=16):
    """Draw the image fullscreen through a narrowed descriptor and return the
    middle pixel."""
    target = ctx.create_render_target(size, size)
    pipe = _fullscreen_sampler(ctx, target)
    dset = ctx.create_descriptor_pool().allocate_set(pipe)
    dset.set_image(0, image, layer=layer, mip=mip)
    g = ctx.graph()
    with g.add_pass(target, name="sample") as p:
        p.bind_pipeline(pipe).bind_descriptor_set(dset).draw(3)
    ctx.submit(g)
    return tuple(int(v) for v in target.color[0].read()[size // 2, size // 2])


def test_sample_one_mip(ctx):
    """mip= binds one level. Each level holds its own colour, so the pixel
    says which one the shader read — a whole-image descriptor always reads
    level 0 through the sampler's default LOD."""
    image = _mipped_image(ctx)
    assert _sample(ctx, image, mip=0) == LEVELS[0]
    assert _sample(ctx, image, mip=1) == LEVELS[1]
    assert _sample(ctx, image, mip=2) == LEVELS[2]


def test_sample_one_layer_of_an_array(ctx):
    """layer= gives a 2D view of one layer, so an array samples through a
    plain sampler2D rather than a sampler2DArray."""
    image = ctx.create_image(16, 16, bz.Format.RGBA8, layers=3, name="slices")
    for layer, colour in enumerate(LEVELS):
        image.update(np.tile(np.array(colour, np.uint8), (16, 16, 1)), layer=layer)
    image.wait()
    assert _sample(ctx, image, layer=1) == LEVELS[1]
    assert _sample(ctx, image, layer=2) == LEVELS[2]


def test_one_cube_face_samples_as_a_2d_texture(ctx):
    """Naming one face asks for that face as a texture, not for a samplerCube
    of it — the reason a narrowed layer is always VIEW_TYPE_2D."""
    faces = [np.tile(np.array((i * 40, 255 - i * 30, 60, 255), np.uint8), (8, 8, 1))
             for i in range(6)]
    cube = ctx.create_image(faces, cube=True)
    cube.wait()
    assert _sample(ctx, cube, layer=3) == (120, 165, 60, 255)


def test_write_one_layer_as_storage_then_sample_it(ctx):
    """A compute pass writes ONE layer through a narrowed storage binding and
    a render pass samples that layer, in one graph. The barrier the fold emits
    names {layer 2, 1 level} rather than the whole array; validation is the
    referee that it named the right one."""
    comp = ctx.compile_shader(str(SHADER_DIR / "store_const.comp"), bz.ShaderStage.COMPUTE)
    writer = ctx.compute_pipeline().shader(comp).storage_image(0).build()
    image = ctx.create_image(16, 16, bz.Format.RGBA8, layers=3, name="target layers")
    for layer in range(3):
        image.update(np.tile(np.array((9, 9, 9, 255), np.uint8), (16, 16, 1)), layer=layer)
    image.wait()

    target = ctx.create_render_target(16, 16)
    reader = _fullscreen_sampler(ctx, target)
    pool = ctx.create_descriptor_pool()
    wset = pool.allocate_set(writer)
    wset.set_storage_image(0, image, layer=2)
    rset = pool.allocate_set(reader)
    rset.set_image(0, image, layer=2)

    g = ctx.graph()
    with g.add_pass(name="fill layer 2") as p:
        p.bind_pipeline(writer).bind_descriptor_set(wset).dispatch(2, 2)
    with g.add_pass(target, name="read layer 2") as p:
        p.bind_pipeline(reader).bind_descriptor_set(rset).draw(3)
    ctx.submit(g)

    # 0.25 / 0.5 / 0.75 of store_const.comp, in 8-bit.
    assert tuple(int(v) for v in target.color[0].read()[8, 8]) == (64, 127, 191, 255)
    # The other layers kept the upload: a narrowed write claims only its own.
    assert tuple(int(v) for v in image.read(layer=0)[8, 8]) == (9, 9, 9, 255)


def test_a_mip_pyramid_in_one_graph(ctx):
    """The case the whole feature exists for: each pass SAMPLES level N-1 and
    WRITES level N of the same image, so the image holds two layouts at once.
    A tracker with one layout per image could only barrier the whole chain,
    which either discards the source or trips the layers."""
    comp = ctx.compile_shader(str(SHADER_DIR / "sample_texture.comp"), bz.ShaderStage.COMPUTE)
    down = (ctx.compute_pipeline().shader(comp)
            .texture(0).storage_image(1).build())
    image = ctx.create_image(64, 64, bz.Format.RGBA8, mip_levels=3, name="pyramid")
    image.update(np.tile(np.array((200, 100, 50, 255), np.uint8), (64, 64, 1)), mip=0)
    image.wait()

    pool = ctx.create_descriptor_pool()
    g = ctx.graph()
    for level in (1, 2):
        dset = pool.allocate_set(down)
        dset.set_image(0, image, mip=level - 1)
        dset.set_storage_image(1, image, mip=level)
        with g.add_pass(name=f"downsample to mip {level}") as p:
            p.bind_pipeline(down).bind_descriptor_set(dset).dispatch(8, 8)
    ctx.submit(g)

    # A flat source box-filters to itself, so every level ends the same colour.
    assert tuple(int(v) for v in image.read(mip=2)[4, 4]) == (200, 100, 50, 255)
    assert tuple(int(v) for v in image.read(mip=0)[4, 4]) == (200, 100, 50, 255)


def test_the_whole_image_is_usable_after_a_pyramid(ctx):
    """The split reconciles: a whole-image use after narrowed ones emits one
    barrier per differing subresource and collapses the state again, so a
    clear right after it is a plain whole-image barrier."""
    comp = ctx.compile_shader(str(SHADER_DIR / "sample_texture.comp"), bz.ShaderStage.COMPUTE)
    down = ctx.compute_pipeline().shader(comp).texture(0).storage_image(1).build()
    image = ctx.create_image(32, 32, bz.Format.RGBA8, mip_levels=2, name="pyramid")
    image.update(np.tile(np.array((10, 20, 30, 255), np.uint8), (32, 32, 1)), mip=0)
    image.wait()

    pool = ctx.create_descriptor_pool()
    dset = pool.allocate_set(down)
    dset.set_image(0, image, mip=0)
    dset.set_storage_image(1, image, mip=1)

    g = ctx.graph()
    with g.add_pass(name="downsample") as p:
        p.bind_pipeline(down).bind_descriptor_set(dset).dispatch(4, 4)
    g.add_pass(name="clear").clear_image(image, (1.0, 1.0, 1.0, 1.0))
    ctx.submit(g)

    assert tuple(int(v) for v in image.read(mip=0)[4, 4]) == (255, 255, 255, 255)


def test_layer_on_a_3d_image_is_refused(ctx):
    """A volume has depth, not layers, and the message says so."""
    volume = ctx.create_image(8, 8, bz.Format.RGBA8, depth=4)
    comp = ctx.compile_shader(str(SHADER_DIR / "store_const.comp"), bz.ShaderStage.COMPUTE)
    pipe = ctx.compute_pipeline().shader(comp).storage_image(0).build()
    dset = ctx.create_descriptor_pool().allocate_set(pipe)
    with pytest.raises(bz.ResourceError, match="3D volume with no layers"):
        dset.set_storage_image(0, volume, layer=1)


def test_a_subresource_outside_the_image_names_the_counts(ctx):
    image = _mipped_image(ctx)
    target = ctx.create_render_target(8, 8)
    pipe = _fullscreen_sampler(ctx, target)
    dset = ctx.create_descriptor_pool().allocate_set(pipe)
    with pytest.raises(bz.ResourceError, match="has 1 layer"):
        dset.set_image(0, image, layer=4)
    with pytest.raises(bz.ResourceError, match="has 3 mip level"):
        dset.set_image(0, image, mip=7)


def test_layer_and_mip_are_keyword_only(ctx):
    """Two adjacent ints selecting different axes is the trap target.layer()
    fixed in 0.23, and the set verbs follow the same rule."""
    image = _mipped_image(ctx)
    target = ctx.create_render_target(8, 8)
    pipe = _fullscreen_sampler(ctx, target)
    dset = ctx.create_descriptor_pool().allocate_set(pipe)
    with pytest.raises(TypeError):
        dset.set_image(0, image, None, 0, 1)
