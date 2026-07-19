"""Descriptor sets return to their pool when collected.

Pools used to be one-way (no FREE_DESCRIPTOR_SET_BIT, no vkFreeDescriptorSets
anywhere): allocate max_sets times and the pool was full forever. GC is the
one obvious way to give a set back — there is no explicit free() verb.
"""

import gc

import bazalt as bz


def test_dropped_sets_return_to_the_pool(ctx, triangle_shaders):
    """Allocate-and-drop far more sets than the pool holds at once."""
    vert, frag = triangle_shaders
    target = bz.RenderTarget(ctx, 16, 16)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                .uniform_buffer(0, bz.ShaderStage.VERTEX, set=0)
                .build(target))

    pool = ctx.create_descriptor_pool(max_sets=4, uniform_buffers=4)

    # 4x the capacity: fails with "pool may be full" unless frees happen.
    for _ in range(16):
        dset = pool.allocate_set(pipeline, set=0)
        del dset
        gc.collect()
        # The freed handle is reclaimed once the GPU provably passed the frame
        # that could have used it; an idle submit provides that proof point.
        cmd = ctx.create_command_buffer()
        cmd.begin()
        ctx.submit(cmd)


def test_pool_dropped_before_its_sets_is_safe(ctx, triangle_shaders):
    """A set keeps its pool alive; dropping the pool first must not dangle."""
    vert, frag = triangle_shaders
    target = bz.RenderTarget(ctx, 16, 16)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                .uniform_buffer(0, bz.ShaderStage.VERTEX, set=0)
                .build(target))

    pool = ctx.create_descriptor_pool(max_sets=4, uniform_buffers=4)
    dset = pool.allocate_set(pipeline, set=0)
    del pool
    gc.collect()

    # The set still works: it holds the pool internally.
    ubuf = ctx.create_buffer([0.0] * 4, bz.BufferType.UNIFORM, bz.MemoryUsage.DYNAMIC)
    # (binding a DYNAMIC buffer to a static set raises — just poke the set to
    # prove it is alive and validated, then drop everything)
    try:
        dset.set_buffer(0, ubuf)
    except bz.ResourceError:
        pass

    del dset
    gc.collect()
    cmd = ctx.create_command_buffer()
    cmd.begin()
    ctx.submit(cmd)
