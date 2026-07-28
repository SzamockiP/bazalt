"""The README's headless example, executed.

It is the first code anyone reads, and it was wrong the moment it was written:
the pipeline declared vertex attributes but the snippet called draw() without
binding a buffer. The validation layers caught it — a documentation bug that
only a running test can find.
"""

import numpy as np

import bazalt as bz
from conftest import SHADER_DIR


def test_readme_headless_quickstart(ctx):
    """Kept in step with the 'Rendering Without a Window' section of README.md."""
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


def test_readme_compute_quickstart(ctx):
    """Kept in step with the 'GPU Compute' section of README.md."""
    sim = (ctx.compute_pipeline()
           .shader(ctx.compile_shader(str(SHADER_DIR / "double.comp"),
                                      bz.ShaderStage.COMPUTE))
           .storage_buffer(0)
           .build())

    data = np.arange(128, dtype=np.float32)
    sbuf = ctx.create_buffer(data, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)

    pool = ctx.create_descriptor_pool(max_sets=4, storage_buffers=4)
    dset = pool.allocate_set(sim, set=0)
    dset.set_buffer(0, sbuf)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.bind_pipeline(sim).bind_descriptor_set(dset, sim, set=0).dispatch(128 // 64)
    ctx.submit(cmd)

    assert np.allclose(sbuf.read(np.float32), data * 2)
