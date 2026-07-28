"""0.17: per-instance vertex attributes, wider vertex formats, strips.

The vertex input used to be one binding at rate VERTEX with three float
formats, so per-instance data could only travel through an SSBO indexed by
gl_InstanceIndex. Every test is two-sided where it can be: the same recording
with one instance and with four, the same four vertices as a list and as a
strip.

Headless, so CI covers it.
"""

import pathlib
import struct

import numpy as np
import pytest

import bazalt as bz

SHADER_DIR = pathlib.Path(__file__).parent / "shaders"

# One quad in the top-left NDC quadrant, as a triangle strip. Instance offsets
# move it into the other three.
QUAD = [-1.0, -1.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0]

# (offset x, offset y, r, g, b, a) — one per quadrant, distinct colours.
INSTANCES = [
    (0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0, 1.0, 0.0, 1.0),
    (0.0, 1.0, 0.0, 0.0, 1.0, 1.0),
    (1.0, 1.0, 1.0, 1.0, 0.0, 1.0),
]

# Where to sample each quadrant in a 64x64 readback: [row, column].
CENTRES = [(16, 16), (16, 48), (48, 16), (48, 48)]


@pytest.fixture
def instanced(ctx):
    """Builds the instanced pipeline; `instance_formats` picks the layout."""
    vert = ctx.compile_shader(str(SHADER_DIR / "instanced.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "vertex_color.frag"), bz.ShaderStage.FRAGMENT)

    def make(target, instance_formats, topology=bz.Topology.TRIANGLE_STRIP):
        return (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .vertex_format([bz.VertexFormat.FLOAT2])
                .instance_format(instance_formats)
                .topology(topology)
                # The strip's winding alternates, and this test is about the
                # input rate, not about facing.
                .cull_mode(bz.CullMode.NONE, bz.FrontFace.COUNTER_CLOCKWISE)
                .build(target))

    return make


def render(ctx, target, pipeline, vbuf, ibuf, instances, vertex_count=4):
    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0])
    cmd.bind_pipeline(pipeline)
    cmd.bind_vertex_buffer(vbuf)
    cmd.bind_vertex_buffer(ibuf, binding=1)
    cmd.draw(vertex_count, instances=instances)
    cmd.end_rendering(target)
    ctx.submit(cmd)
    return target.color[0].read()


def float_instance_buffer(ctx, rows):
    return ctx.create_buffer([v for row in rows for v in row], bz.BufferType.VERTEX,
                             bz.MemoryUsage.STATIC, bz.DataType.FLOAT)


def test_one_draw_paints_every_instance(ctx, instanced):
    """Four quadrants, four colours, one draw — and one instance paints only
    the first, which is what makes this a two-sided test rather than a
    screenshot."""
    target = bz.RenderTarget(ctx, 64, 64)
    pipeline = instanced(target, [bz.VertexFormat.FLOAT2, bz.VertexFormat.FLOAT4])
    vbuf = ctx.create_buffer(QUAD, bz.BufferType.VERTEX, bz.MemoryUsage.STATIC, bz.DataType.FLOAT)
    ibuf = float_instance_buffer(ctx, INSTANCES)

    pixels = render(ctx, target, pipeline, vbuf, ibuf, instances=4)
    for (row, col), inst in zip(CENTRES, INSTANCES):
        expected = [round(c * 255) for c in inst[2:5]]
        assert np.allclose(pixels[row, col][:3], expected, atol=2), f"quadrant at {row},{col}"

    pixels = render(ctx, target, pipeline, vbuf, ibuf, instances=1)
    assert np.allclose(pixels[CENTRES[0]][:3], [255, 0, 0], atol=2)
    for row, col in CENTRES[1:]:
        assert np.all(pixels[row, col][:3] == 0), "an unused instance still painted"


def test_ubyte4_norm_instance_colour(ctx, instanced):
    """A four-byte colour reads back as 0..1, at a quarter of the size.

    The instance rows are packed by hand (2 floats + 4 bytes), which is the
    layout the format list describes.
    """
    target = bz.RenderTarget(ctx, 64, 64)
    pipeline = instanced(target, [bz.VertexFormat.FLOAT2, bz.VertexFormat.UBYTE4_NORM])
    vbuf = ctx.create_buffer(QUAD, bz.BufferType.VERTEX, bz.MemoryUsage.STATIC, bz.DataType.FLOAT)

    packed = b"".join(struct.pack("<2f4B", x, y, 128, 0, 255, 255) for x, y in
                      [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
    assert len(packed) == 4 * 12, "12 bytes per instance: 2 floats plus 4 bytes"
    ibuf = ctx.create_buffer(np.frombuffer(packed, dtype=np.uint8),
                             bz.BufferType.VERTEX, bz.MemoryUsage.STATIC)

    pixels = render(ctx, target, pipeline, vbuf, ibuf, instances=4)
    for row, col in CENTRES:
        assert np.allclose(pixels[row, col][:3], [128, 0, 255], atol=2)


def test_strip_covers_the_quad_a_list_leaves_half(ctx, instanced):
    """The same four vertices: a strip makes two triangles, a list makes one."""
    target = bz.RenderTarget(ctx, 64, 64)
    vbuf = ctx.create_buffer(QUAD, bz.BufferType.VERTEX, bz.MemoryUsage.STATIC, bz.DataType.FLOAT)
    ibuf = float_instance_buffer(ctx, INSTANCES[:1])

    strip = instanced(target, [bz.VertexFormat.FLOAT2, bz.VertexFormat.FLOAT4])
    painted_strip = int(np.count_nonzero(render(ctx, target, strip, vbuf, ibuf, 1)[..., 0]))

    triangles = instanced(target, [bz.VertexFormat.FLOAT2, bz.VertexFormat.FLOAT4],
                          topology=bz.Topology.TRIANGLE_LIST)
    painted_list = int(np.count_nonzero(render(ctx, target, triangles, vbuf, ibuf, 1)[..., 0]))

    assert painted_strip > painted_list * 1.5, (
        f"strip painted {painted_strip} pixels, list {painted_list} — the strip "
        "should cover about twice the area")


def test_draw_indexed_takes_an_instance_count(ctx, instanced):
    """draw_indexed(instances=) is what draw_indexed_instanced used to be."""
    target = bz.RenderTarget(ctx, 64, 64)
    pipeline = instanced(target, [bz.VertexFormat.FLOAT2, bz.VertexFormat.FLOAT4],
                         topology=bz.Topology.TRIANGLE_LIST)
    vbuf = ctx.create_buffer(QUAD, bz.BufferType.VERTEX, bz.MemoryUsage.STATIC, bz.DataType.FLOAT)
    ibuf = float_instance_buffer(ctx, INSTANCES)
    indices = ctx.create_buffer([0, 1, 2, 1, 3, 2], bz.BufferType.INDEX,
                                bz.MemoryUsage.STATIC, bz.DataType.UINT32)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0]) as c:
        (c.bind_pipeline(pipeline)
          .bind_vertex_buffer(vbuf)
          .bind_vertex_buffer(ibuf, binding=1)
          .bind_index_buffer(indices)
          .draw_indexed(6, instances=4))
    ctx.submit(cmd)

    pixels = target.color[0].read()
    for (row, col), inst in zip(CENTRES, INSTANCES):
        expected = [round(c * 255) for c in inst[2:5]]
        assert np.allclose(pixels[row, col][:3], expected, atol=2)


def test_draw_indexed_instanced_is_gone(ctx):
    """The old spelling is removed, not deprecated — pre-1.0 breaks are batched."""
    cmd = ctx.create_command_buffer()
    assert not hasattr(cmd, "draw_indexed_instanced")
