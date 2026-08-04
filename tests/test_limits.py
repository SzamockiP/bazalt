"""What the device allows, in numbers (0.26).

There is nothing to compare these against — they are the driver's own answers —
so the tests check the relationships that must hold on any device, plus the one
thing a limit is FOR: deciding what to do before the driver refuses.
"""

import numpy as np
import pytest

import bazalt as bz

WORKGROUP_SHADER = """
#version 450
layout(local_size_x_id = 0) in;
layout(set = 0, binding = 0, std430) buffer Out { uint v[]; };
void main() {
    if (gl_GlobalInvocationID.x == 0u) { v[0] = gl_WorkGroupSize.x; }
}
"""


def test_the_limits_are_plausible(ctx):
    limits = ctx.limits
    assert limits.max_storage_buffer > 0
    assert limits.max_uniform_buffer > 0
    # Vulkan's own floors, so a device reporting less is not a Vulkan device.
    assert limits.max_push_constants >= 128
    assert limits.max_workgroup_invocations >= 128
    assert limits.max_workgroup_memory >= 16384
    # Vulkan's floor is [128, 128, 64], so the z axis is allowed to be smaller.
    assert limits.max_workgroup_size[0] >= 128 and limits.max_workgroup_size[1] >= 128
    assert limits.max_workgroup_size[2] >= 64
    assert all(axis >= 65535 for axis in limits.max_dispatch)


def test_a_buffer_may_be_larger_than_a_descriptor_can_see(ctx):
    """The relationship the whole release turns on."""
    assert ctx.limits.max_buffer >= ctx.limits.max_storage_buffer


def test_an_allocation_can_back_what_a_descriptor_can_see(ctx):
    """max_allocation is the other half of the ceiling: a buffer needs one.

    So the smaller of it and max_buffer is what a caller may really ask for,
    and a device where it fell under the descriptor range would make
    max_storage_buffer unreachable — which no real driver does.
    """
    assert ctx.limits.max_allocation >= ctx.limits.max_storage_buffer


def test_the_subgroup_range_contains_the_width(ctx):
    assert ctx.limits.min_subgroup_size <= ctx.limits.max_subgroup_size
    if ctx.subgroup_size:
        assert ctx.limits.min_subgroup_size <= ctx.subgroup_size <= ctx.limits.max_subgroup_size


def test_limits_are_read_only(ctx):
    with pytest.raises(AttributeError):
        ctx.limits.max_storage_buffer = 1


def test_repr_names_the_numbers(ctx):
    assert "max_storage_buffer" in repr(ctx.limits)


def test_a_workgroup_size_the_pipeline_picks(extra_context):
    """local_size_x_id needs maintenance4, which is what the Feature names."""
    context = extra_context(optional=[bz.Feature.WORKGROUP_SIZE])
    if not context.supports(bz.Feature.WORKGROUP_SIZE):
        pytest.skip("this GPU has no maintenance4")

    for size in (16, 32):
        pipeline = (context.compute_pipeline()
                    .shader(context.compile_shader(source=WORKGROUP_SHADER, stage=bz.ShaderStage.COMPUTE))
                    .storage_buffer(0)
                    .constant(0, size)
                    .build())
        out = context.create_buffer(
            np.zeros(4, dtype=np.uint32), bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
        bound = context.create_descriptor_pool().allocate_set(pipeline)
        bound.set_buffer(0, out)

        cmd = context.create_command_buffer()
        cmd.begin()
        cmd.bind_pipeline(pipeline).bind_descriptor_set(bound, pipeline).dispatch(1)
        context.submit(cmd)

        assert out.read(np.uint32)[0] == size


def test_a_workgroup_within_the_reported_maximum_builds(extra_context):
    """The limit is only useful if a size taken from it actually works."""
    context = extra_context(optional=[bz.Feature.WORKGROUP_SIZE])
    if not context.supports(bz.Feature.WORKGROUP_SIZE):
        pytest.skip("this GPU has no maintenance4")

    biggest = min(context.limits.max_workgroup_size[0], context.limits.max_workgroup_invocations)
    pipeline = (context.compute_pipeline()
                .shader(context.compile_shader(source=WORKGROUP_SHADER, stage=bz.ShaderStage.COMPUTE))
                .storage_buffer(0)
                .constant(0, biggest)
                .build())
    assert pipeline is not None
