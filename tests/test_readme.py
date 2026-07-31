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

import bazalt as bz
from conftest import SHADER_DIR


def test_readme_triangle(ctx):
    """Kept in step with the 'A triangle' and 'The same code, with no window'
    sections of README.md."""
    target = bz.RenderTarget(ctx, 800, 600, depth=bz.Format.D32F)

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

    target = bz.RenderTarget(ctx, W, H)
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
