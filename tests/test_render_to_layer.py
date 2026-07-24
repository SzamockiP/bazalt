"""Render-to-layer / render-to-mip: rasterize a scene into ONE subresource
(array layer / cube face / mip level) of a render target.

Compute already writes layers/faces/mips since 0.10 (see test_cubemaps). This
covers the GRAPHICS path: a pass whose attachment is target.layer(i) / .mip(m).
The attachments stay ordinary bz.Image objects sampled the usual way — the only
new API is the layers=/cube=/mip_levels= ctor kwargs and the .layer()/.mip()
slice. Layered auto-barriers (per-subresource, and the whole-image final mark)
are audited by the validation-as-assert `ctx` fixture, not just the numbers.
"""

import struct

import numpy as np
import pytest

import bazalt as bz

from conftest import SHADER_DIR


def _sample_depth_layer(ctx, target, layer):
    """Fullscreen-sample one layer of `target`'s depth array into a fresh colour
    target and read it back as grayscale depth. NEAREST: depth formats don't
    universally filter."""
    fullscreen = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    view_frag = ctx.compile_shader(str(SHADER_DIR / "depth_array_view.frag"), bz.ShaderStage.FRAGMENT)
    screen = bz.RenderTarget(ctx, target.width, target.height)
    pipe = (ctx.graphics_pipeline()
            .vertex_shader(fullscreen)
            .fragment_shader(view_frag)
            .texture(0, bz.ShaderStage.FRAGMENT, set=0)
            .push_constant(4, bz.ShaderStage.FRAGMENT)
            .build(screen))

    pool = ctx.create_descriptor_pool(max_sets=1, samplers=1)
    dset = pool.allocate_set(pipe, set=0)
    dset.set_image(0, target.depth, sampler=ctx.create_sampler(filter=bz.Filter.NEAREST))

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(screen, clear_color=[0, 0, 0, 1])
    cmd.bind_pipeline(pipe)
    cmd.bind_descriptor_set(dset, pipe, set=0)
    cmd.push_constants(pipe, 0, struct.pack("i", layer))
    cmd.draw(3)
    cmd.end_rendering(screen)
    ctx.submit(cmd)
    return screen.read_pixels()


def test_render_into_specific_layer(ctx, triangle_shaders, triangle_buffers):
    """A 2-layer depth target: clear-only into layer 0, triangle into layer 1.
    Sampling proves the triangle landed in layer 1 and layer 0 saw only the
    clear — i.e. .layer(i) routed geometry to the subresource it named."""
    vert, _ = triangle_shaders
    vbuf, ibuf = triangle_buffers

    target = bz.RenderTarget(ctx, 64, 64, color=None, depth=bz.Format.D32F, layers=2)
    assert target.depth.array_layers == 2

    depth_pipe = (ctx.graphics_pipeline()
                  .vertex_shader(vert)
                  .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                  .depth_test(True)
                  .build(target.layer(0)))

    cmd = ctx.create_command_buffer()
    cmd.begin()
    # Layer 0: clear only, no draw — leaves the whole layer at clear depth 1.0.
    cmd.begin_rendering(target.layer(0))
    cmd.end_rendering(target.layer(0))
    # Layer 1: the triangle.
    cmd.begin_rendering(target.layer(1))
    cmd.bind_pipeline(depth_pipe)
    cmd.bind_vertex_buffer(vbuf)
    cmd.bind_index_buffer(ibuf)
    cmd.draw_indexed(3)
    cmd.end_rendering(target.layer(1))
    ctx.submit(cmd)

    layer0 = _sample_depth_layer(ctx, target, 0)
    layer1 = _sample_depth_layer(ctx, target, 1)

    # Layer 0: untouched by geometry -> uniform clear depth (white).
    assert layer0[2, 2, 0] == 255
    assert layer0[40, 32, 0] == 255, "layer 0 must have no triangle"
    # Layer 1: clear at the corner, triangle (z=0, black) in the interior.
    assert layer1[2, 2, 0] == 255, "layer 1 corner is the clear depth"
    assert layer1[40, 32, 0] == 0, "layer 1 interior is the triangle"


def test_render_into_mip(ctx):
    """A 3-mip colour target: clear each mip to a distinct colour via a pass
    targeting .mip(m). Sampling each LOD back returns that mip's colour, proving
    (a) .mip(m) selected the right level and (b) extent() shrank per mip — a
    wrong renderArea for the half/quarter-size mip would trip validation."""
    fullscreen = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    lod_frag = ctx.compile_shader(str(SHADER_DIR / "sample_lod.frag"), bz.ShaderStage.FRAGMENT)

    target = bz.RenderTarget(ctx, 64, 64, color=bz.Format.RGBA8, mip_levels=3)
    colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]

    cmd = ctx.create_command_buffer()
    cmd.begin()
    for m, c in enumerate(colors):
        cmd.begin_rendering(target.mip(m), clear_color=c)  # clear-only, no draw
        cmd.end_rendering(target.mip(m))
    ctx.submit(cmd)

    screen = bz.RenderTarget(ctx, 8, 8)
    pipe = (ctx.graphics_pipeline()
            .vertex_shader(fullscreen)
            .fragment_shader(lod_frag)
            .texture(0, bz.ShaderStage.FRAGMENT, set=0)
            .push_constant(4, bz.ShaderStage.FRAGMENT)
            .build(screen))
    pool = ctx.create_descriptor_pool(max_sets=1, samplers=1)
    dset = pool.allocate_set(pipe, set=0)
    dset.set_image(0, target.color[0], sampler=ctx.create_sampler(filter=bz.Filter.NEAREST))

    expected = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    for m in range(3):
        cmd = ctx.create_command_buffer()
        cmd.begin()
        cmd.begin_rendering(screen, clear_color=[0, 0, 0, 1])
        cmd.bind_pipeline(pipe)
        cmd.bind_descriptor_set(dset, pipe, set=0)
        cmd.push_constants(pipe, 0, struct.pack("f", float(m)))
        cmd.draw(3)
        cmd.end_rendering(screen)
        ctx.submit(cmd)
        assert np.array_equal(screen.read_pixels()[4, 4, :3], expected[m]), \
            f"mip {m}: {screen.read_pixels()[4, 4, :3]}"


def test_render_into_layer_and_mip(ctx):
    """Combined axes: a layered AND mipped target, render into (layer 1, mip 1).
    Same SubresourceTarget machinery as the single-axis slices — this pins that
    both can be selected at once and the pass is validation-clean (the view and
    the barrier must agree on {layer 1, mip 1}, or the ctx fixture flags it)."""
    target = bz.RenderTarget(ctx, 32, 32, color=bz.Format.RGBA8, layers=2, mip_levels=2)
    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(target.layer(1, mip=1), clear_color=[1, 0, 0, 1])
    cmd.end_rendering(target.layer(1, mip=1))
    ctx.submit(cmd)


def test_combined_axis_bounds_are_checked(ctx):
    target = bz.RenderTarget(ctx, 16, 16, color=bz.Format.RGBA8, layers=2, mip_levels=2)
    with pytest.raises(bz.ResourceError):
        target.layer(0, mip=2)
    with pytest.raises(bz.ResourceError):
        target.layer(2, mip=0)


def test_layer_out_of_range_is_refused(ctx):
    target = bz.RenderTarget(ctx, 16, 16, depth=bz.Format.D32F, layers=2)
    with pytest.raises(bz.ResourceError):
        target.layer(2)


def test_mip_out_of_range_is_refused(ctx):
    target = bz.RenderTarget(ctx, 16, 16, mip_levels=2)
    with pytest.raises(bz.ResourceError):
        target.mip(2)


def test_cube_target_makes_a_sampleable_cubemap(ctx):
    """cube=True gives the colour attachment 6 layers with a CUBE view, so after
    rendering all six faces target.color[0] samples as a cubemap."""
    target = bz.RenderTarget(ctx, 16, 16, color=bz.Format.RGBA8, depth=bz.Format.D32F, cube=True)
    assert target.color[0].is_cube
    assert target.color[0].array_layers == 6
    assert target.depth.array_layers == 6


def test_msaa_with_layers_is_refused(ctx):
    """A multisampled image can't be layered (Image forbids samples>1 + layers);
    the ctor must reject it up front rather than at vkCreateImage."""
    with pytest.raises(bz.ResourceError):
        bz.RenderTarget(ctx, 16, 16, samples=4, layers=2)
