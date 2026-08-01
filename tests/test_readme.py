"""The README's code, executed.

It is the first code anyone reads, and it was wrong the moment it was written:
the pipeline declared vertex attributes but the snippet called draw() without
binding a buffer. The validation layers caught it — a documentation bug that
only a running test can find.

The README shows the two examples with a window. The window is the only part
these tests replace, with an offscreen RenderTarget: everything else (the
pipelines, the descriptor sets, the recording) is the README line for line.
"""

import struct

import numpy as np
import pytest

import bazalt as bz
from conftest import SHADER_DIR


def test_readme_triangle(ctx):
    """Kept in step with the 'A triangle' and 'The same code, with no window'
    sections of README.md."""
    target = ctx.create_render_target(800, 600, depth=bz.Format.D32F)

    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(ctx.compile_shader(str(SHADER_DIR / "triangle.vert"),
                                                  bz.ShaderStage.VERTEX))
                .fragment_shader(ctx.compile_shader(str(SHADER_DIR / "triangle.frag"),
                                                    bz.ShaderStage.FRAGMENT))
                .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                .build(target))

    vbuf = ctx.create_buffer([
        +0.0, -0.5, 0.0, 1.0, 0.0, 0.0,
        -0.5, +0.5, 0.0, 0.0, 1.0, 0.0,
        +0.5, +0.5, 0.0, 0.0, 0.0, 1.0,
    ], bz.BufferType.VERTEX, bz.MemoryUsage.STATIC, bz.DataType.FLOAT)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(target, clear_color=[0.1, 0.2, 0.3, 1.0]) as c:
        c.bind_pipeline(pipeline).bind_vertex_buffer(vbuf).draw(3)

    ctx.submit(cmd)

    pixels = target.color[0].read()
    assert pixels.shape == (600, 800, 4)
    assert pixels.dtype == np.uint8
    # Something was actually drawn, not just cleared.
    assert not np.allclose(pixels[300, 400, :3], np.array([26, 51, 77]), atol=2)


def test_readme_compute_writes_an_image(ctx):
    """Kept in step with the 'Compute writes an image' section of README.md.

    The point of the section is the barrier nobody writes: the dispatch leaves
    the image in GENERAL, the draw samples it, and the gradient below only comes
    out if bazalt put the transition between them.
    """
    W = H = 64

    generate = (ctx.compute_pipeline()
                .shader(ctx.compile_shader(str(SHADER_DIR / "pattern.comp"),
                                           bz.ShaderStage.COMPUTE))
                .storage_image(0)
                .push_constant(4)
                .build())

    target = ctx.create_render_target(W, H)
    present = (ctx.graphics_pipeline()
               .vertex_shader(ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"),
                                                 bz.ShaderStage.VERTEX))
               # present.frag in the README — same three lines.
               .fragment_shader(ctx.compile_shader(str(SHADER_DIR / "textured.frag"),
                                                   bz.ShaderStage.FRAGMENT))
               .texture(0, bz.ShaderStage.FRAGMENT, set=0)
               .build(target))

    image = ctx.create_image(W, H, bz.Format.RGBA8)

    pool = ctx.create_descriptor_pool(max_sets=2, textures=1, storage_images=1)
    write_set = pool.allocate_set(generate, set=0)
    write_set.set_storage_image(0, image)
    read_set = pool.allocate_set(present, set=0)
    read_set.set_image(0, image)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    (cmd.bind_pipeline(generate)
        .bind_descriptor_set(write_set, generate, set=0)
        .push_constants(generate, 0, struct.pack("<f", 0.0))
        .dispatch((W + 7) // 8, (H + 7) // 8))
    with cmd.rendering(target) as c:
        c.bind_pipeline(present).bind_descriptor_set(read_set, present, set=0).draw(3)

    ctx.submit(cmd)

    pixels = target.color[0].read()
    assert pixels.shape == (H, W, 4)
    # pattern.comp writes uv.x into green and uv.y into blue.
    assert int(pixels[H // 2, 0, 1]) < int(pixels[H // 2, W - 1, 1])
    assert int(pixels[0, W // 2, 2]) < int(pixels[H - 1, W // 2, 2])


def test_readme_notebook_section(extra_context):
    """Kept in step with the 'In a notebook' section of README.md.

    The claim the section makes is the ordering one: read the pixels inside the
    block, because a closed Context refuses. Both halves are asserted here so the
    advice cannot drift away from the behaviour."""
    context = extra_context()
    with context as ctx:
        target = ctx.create_render_target(64, 64)
        cmd = ctx.create_command_buffer()
        cmd.begin()
        with cmd.rendering(target, clear_color=[0.1, 0.2, 0.3, 1.0]):
            pass
        ctx.submit(cmd)
        pixels = target.color[0].read()

    assert pixels.shape == (64, 64, 4)
    assert np.allclose(pixels[8, 8, :3], np.array([26, 51, 77]), atol=2)

    with pytest.raises(bz.StateError):
        target.color[0].read()


def test_readme_projection_matrix():
    """Kept in step with the 'Vulkan clip space' section of README.md.

    The section tells the reader to use perspectiveRH_ZO rather than perspective,
    and the difference is invisible until a depth test goes wrong. glm is an
    example dependency rather than a test one, so this skips where it is absent.
    """
    glm = pytest.importorskip("glm")

    # The OpenGL matrix the section warns about: depth runs -1..1.
    gl = glm.perspective(glm.radians(60.0), 1.0, 0.1, 100.0)
    near = gl * glm.vec4(0, 0, -0.1, 1)
    far = gl * glm.vec4(0, 0, -100.0, 1)
    assert near.z / near.w == pytest.approx(-1.0, abs=1e-4)
    assert far.z / far.w == pytest.approx(1.0, abs=1e-4)

    # What the README tells you to write instead: depth runs 0..1.
    proj = glm.perspectiveRH_ZO(glm.radians(60.0), 1.0, 0.1, 100.0)
    near = proj * glm.vec4(0, 0, -0.1, 1)
    far = proj * glm.vec4(0, 0, -100.0, 1)
    assert near.z / near.w == pytest.approx(0.0, abs=1e-4)
    assert far.z / far.w == pytest.approx(1.0, abs=1e-4)

    # And the y sign is still the caller's to flip: RH_ZO does not do it.
    assert proj[1][1] > 0
    proj[1][1] *= -1
    assert proj[1][1] < 0


def test_readme_negative_viewport_flips(ctx):
    """Kept in step with the same section, which offers
    cmd.set_viewport(0, h, w, -h) as the escape hatch that makes a y_up switch
    unnecessary. If this stops working, the section is advertising a ceiling
    bazalt does not actually leave open."""
    target = ctx.create_render_target(64, 64)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(ctx.compile_shader(str(SHADER_DIR / "triangle.vert"),
                                                  bz.ShaderStage.VERTEX))
                .fragment_shader(ctx.compile_shader(str(SHADER_DIR / "triangle.frag"),
                                                    bz.ShaderStage.FRAGMENT))
                .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                .cull_mode(bz.CullMode.NONE, bz.FrontFace.COUNTER_CLOCKWISE)
                .build(target))
    # Wholly in NDC y < 0, which is the TOP half under Vulkan's default.
    vbuf = ctx.create_buffer([
        -0.8, -0.9, 0.0, 1.0, 1.0, 1.0,
        +0.8, -0.9, 0.0, 1.0, 1.0, 1.0,
        +0.0, -0.2, 0.0, 1.0, 1.0, 1.0,
    ], bz.BufferType.VERTEX, bz.MemoryUsage.STATIC, bz.DataType.FLOAT)

    def halves(flip):
        cmd = ctx.create_command_buffer()
        cmd.begin()
        with cmd.rendering(target, clear_color=[0, 0, 0, 1]) as c:
            if flip:
                c.set_viewport(0.0, 64.0, 64.0, -64.0)
            c.bind_pipeline(pipeline).bind_vertex_buffer(vbuf).draw(3)
        ctx.submit(cmd)
        lit = target.color[0].read()[:, :, 0] > 128
        return int(lit[:32].sum()), int(lit[32:].sum())

    top, bottom = halves(flip=False)
    assert top > 0 and bottom == 0

    flipped_top, flipped_bottom = halves(flip=True)
    assert flipped_bottom == top and flipped_top == 0
