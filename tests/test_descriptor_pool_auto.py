"""The automatic descriptor pool (0.23): create_descriptor_pool() with no
arguments grows blocks as sets are allocated, each block sized from the layout
being served.

What the tests pin: growth actually happens (more allocations than any default
block holds), frame sets follow the same path, a freed set goes back to its own
block, the fixed-size pool still exhausts as an error, and a bindless array
larger than the default block lands in a block sized for it — the MoltenVK
caveat from 0.22, as a regression test.
"""

import numpy as np
import pytest

import bazalt as bz
from conftest import SHADER_DIR


def buffer_pipeline(ctx):
    comp = ctx.compile_shader(str(SHADER_DIR / "double.comp"), bz.ShaderStage.COMPUTE)
    return ctx.compute_pipeline().shader(comp).storage_buffer(0).build()


def test_auto_pool_takes_no_arguments(ctx):
    pool = ctx.create_descriptor_pool()
    pipe = buffer_pipeline(ctx)
    dset = pool.allocate_set(pipe, set=0)
    buf = ctx.create_buffer(64, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    dset.set_buffer(0, buf)


def test_auto_pool_grows_past_any_single_block(ctx):
    """200 live sets is more than the default block holds, so this only passes
    if a second (and third) block is allocated on demand."""
    pool = ctx.create_descriptor_pool()
    pipe = buffer_pipeline(ctx)
    sets = [pool.allocate_set(pipe, set=0) for _ in range(200)]
    assert len(sets) == 200


def test_auto_pool_serves_frame_sets(ctx):
    pool = ctx.create_descriptor_pool()
    pipe = buffer_pipeline(ctx)
    sets = [pool.allocate_frame_set(pipe, set=0) for _ in range(100)]
    assert len(sets) == 100


def test_freed_sets_return_to_their_own_block(ctx):
    """Allocate, drop, allocate again, twice over: every free must land in the
    block that allocated the set (an auto pool owns several), or the validation
    layers report the mismatch and the fixture fails the test."""
    pool = ctx.create_descriptor_pool()
    pipe = buffer_pipeline(ctx)
    for _ in range(3):
        sets = [pool.allocate_set(pipe, set=0) for _ in range(100)]
        sets.clear()
        ctx.wait()
        ctx.begin_frame()


def test_a_full_fixed_pool_raises_one_type(ctx):
    """The escape hatch keeps its contract: explicit sizes mean one block, and
    a pool-full result is ResourceError — one type, whichever VkResult the
    driver picks. Before 0.23 the same mistake arrived as OutOfMemoryError from
    OUT_OF_POOL_MEMORY and as ResourceError from FRAGMENTED_POOL.

    WHETHER a driver refuses at all is its own business: the spec lets an
    implementation serve more sets than the pool was sized for, and lavapipe
    does, because its descriptors are host memory. So this probes and skips
    with what it saw — a skip that names the observation survives a driver
    changing its mind, which is the rule the 0.22 MoltenVK skips follow."""
    pool = ctx.create_descriptor_pool(max_sets=1, storage_buffers=1)
    pipe = buffer_pipeline(ctx)
    served = []
    for _ in range(64):
        try:
            served.append(pool.allocate_set(pipe, set=0))
        except bz.OutOfMemoryError:
            pytest.fail("a full pool must raise ResourceError, not OutOfMemoryError")
        except bz.ResourceError:
            return
    pytest.skip(f"this driver served {len(served)} sets from a pool sized for 1")


def test_half_specified_pool_is_a_value_error(ctx):
    with pytest.raises(ValueError, match="max_sets"):
        ctx.create_descriptor_pool(textures=4)


def test_auto_pool_fits_a_bindless_array(extra_context):
    """An array larger than the default per-type block size, so the block must
    be sized from the layout — which is also what MoltenVK requires even for
    partially written arrays (0.22).

    80 is chosen between two bounds: above the 64-descriptor default block, so
    a fixed-size block cannot serve it, and below MoltenVK's
    maxPerStageDescriptorUpdateAfterBindSamplers of 96, which a first version
    of this test walked straight through with 256."""
    ctx = extra_context(optional=[bz.Feature.BINDLESS])
    if not ctx.supports(bz.Feature.BINDLESS):
        pytest.skip("GPU reports no descriptorIndexing")
    target = bz.RenderTarget(ctx, 8, 8)
    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "bindless_push.frag"), bz.ShaderStage.FRAGMENT)
    pipe = (ctx.graphics_pipeline()
            .vertex_shader(vert)
            .fragment_shader(frag)
            .texture(0, bz.ShaderStage.FRAGMENT, set=0, count=80)
            .push_constant(4, bz.ShaderStage.FRAGMENT)
            .build(target))

    pool = ctx.create_descriptor_pool()
    dset = pool.allocate_set(pipe, set=0)
    pixels = np.zeros((4, 4, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    dset.set_image(0, ctx.create_image(pixels), index=79)
