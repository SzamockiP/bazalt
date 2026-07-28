"""debugPrintfEXT() output has to reach the logger, and reach it as SHADER.

The routing is the part worth pinning. Printf arrives on the same debug-utils
messenger as the validation findings, so a naive implementation reports a print
as Source.VALIDATION — which is exactly what the `ctx` fixture fails a test on.
A shader that says hello would then break every test around it.
"""

import pytest

import bazalt as bz


PRINTF_COMPUTE = """
#version 450
#extension GL_EXT_debug_printf : enable
layout(local_size_x = 1) in;
void main()
{
    debugPrintfEXT("bazalt printf %d", 1234);
}
"""

PLAIN_COMPUTE = """
#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) buffer Out { uint values[]; };
void main()
{
    values[0] = 7u;
}
"""


def printf_messages(seen):
    return [m for m in seen if m.source == bz.Source.SHADER and "bazalt printf" in m.text]


def dispatch_printf(context):
    """Compile and run the printing shader, returning what the logger saw."""
    seen = []
    context.logger.on_message(seen.append)

    shader = context.compile_shader("printf.comp", bz.ShaderStage.COMPUTE, source=PRINTF_COMPUTE)
    pipeline = context.compute_pipeline().shader(shader).build()

    cmd = context.create_command_buffer()
    cmd.begin().bind_pipeline(pipeline).dispatch(1)
    context.submit(cmd)

    context.logger.flush()
    return seen


def test_printf_reaches_the_logger_as_shader_output(extra_context):
    context = extra_context(shader_printf=True)
    if not context.shader_printf:
        pytest.skip("Context reports shader_printf off")

    seen = dispatch_printf(context)
    prints = printf_messages(seen)
    if not prints:
        pytest.skip("no validation layer with DebugPrintf support on this machine")

    # The value, not just the format string: a layer that delivered the literal
    # would prove nothing about the shader having run.
    assert any("1234" in m.text for m in prints)


def test_printf_is_not_reported_as_a_validation_error(extra_context):
    """The whole reason the routing exists: a print must not fail its own test."""
    context = extra_context(shader_printf=True)
    seen = dispatch_printf(context)
    if not printf_messages(seen):
        pytest.skip("no validation layer with DebugPrintf support on this machine")

    assert not [m for m in seen if m.source == bz.Source.VALIDATION and m.severity >= bz.Severity.ERROR]


def test_printf_text_is_the_shaders_words_not_the_layers_report(extra_context):
    """The layer wraps the print in its own boilerplate; bazalt peels it off.

    Without this the useful five characters arrive behind two hundred of object
    handles and message ids, once per print, which for a print inside a loop is
    the difference between a tool and a wall of text.
    """
    context = extra_context(shader_printf=True)
    prints = printf_messages(dispatch_printf(context))
    if not prints:
        pytest.skip("no validation layer with DebugPrintf support on this machine")

    assert any(m.text == "bazalt printf 1234" for m in prints), \
        "expected the bare print, got:\n" + "\n".join(f"  {m.text!r}" for m in prints)


def test_printf_off_by_default(ctx):
    assert ctx.shader_printf is False


def test_printf_needs_the_validation_layers():
    """The contradiction is named, not silently resolved either way."""
    with pytest.raises(bz.InitializationError):
        bz.Context(validation="off", shader_printf=True)


def test_a_context_with_printf_still_runs_ordinary_shaders(extra_context):
    """Printf turns the shader optimizer off for the whole Context, so the
    ordinary path has to keep working — that switch is the one thing about this
    feature that touches shaders which never print."""
    context = extra_context(shader_printf=True)
    shader = context.compile_shader("plain.comp", bz.ShaderStage.COMPUTE, source=PLAIN_COMPUTE)
    pipeline = context.compute_pipeline().shader(shader).storage_buffer(0).build()

    buf = context.create_buffer([0, 0, 0, 0], bz.BufferType.STORAGE, bz.MemoryUsage.STATIC,
                                bz.DataType.UINT32)
    pool = context.create_descriptor_pool(max_sets=1, storage_buffers=1)
    dset = pool.allocate_set(pipeline, set=0)
    dset.set_buffer(0, buf)

    cmd = context.create_command_buffer()
    cmd.begin().bind_pipeline(pipeline).bind_descriptor_set(dset, pipeline, 0).dispatch(1)
    context.submit(cmd)

    assert buf.read("uint32")[0] == 7
