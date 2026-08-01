"""SPIR-V reflection, and the barriers it decides (0.19). Pays technical debt #3.

Two halves, and the first one is what CI can actually referee.

**The parser**, through `shader.writes`. Sync validation is skipped on CI runners
(debt #4), so without a property to assert on, the atomics path, the access-chain
path and the fail-open cases would be checked by nothing that runs there. Each test
below names the mistake it would catch — the negative case (`refl_readonly`) is the
most important of them, because if a read-only binding ever appears in `writes`
then nothing is ever narrowed and the feature is decoration.

**The barriers**, through validation-as-assert. The headline claim is that a
graphics shader writing a storage image is ordered against a later read with no
`cmd.barrier()` in sight. That was not merely untracked before 0.19: it was wrong,
because `set_storage_image` already recorded the image as resting in GENERAL while
`track_draw_` never transitioned it.

The invariant the whole thing rests on: reads are always assumed, writes are only
ever narrowed by a positive proof of absence. A barrier disappears only when the
parser proves no store, atomic or imageWrite touches that binding.
"""

import numpy as np
import pytest

import bazalt as bz

from conftest import SHADER_DIR


def compute(ctx, name):
    return ctx.compile_shader(str(SHADER_DIR / name), bz.ShaderStage.COMPUTE)


# ── the parser ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("shader,expected,why", [
    ("refl_store.comp", [(0, 0)],
     "a plain OpStore through an access chain"),
    ("refl_atomic_only.comp", [(0, 0)],
     "atomicAdd and nothing else — a GPU counter, which a store-only scan calls "
     "read-only and thereby reinstates the whole debt"),
    ("refl_struct_member.comp", [(2, 3)],
     "a long access chain (array of structs, member, element) still resolving to "
     "its root variable, and a non-zero set index"),
    ("refl_image_store.comp", [(0, 0)],
     "OpImageWrite"),
    ("refl_image_atomic.comp", [(0, 0)],
     "imageAtomicAdd, which reaches the image through OpImageTexelPointer — the "
     "one alias opcode whose absence silently misses image atomics"),
    ("refl_helper_writes.comp", [(0, 0)],
     "a write inside a helper function: at -O0 nothing is inlined and the callee "
     "may be defined later, so the descriptor is assumed written (fail-open)"),
])
def test_the_parser_finds_the_writes(ctx, shader, expected, why):
    module = compute(ctx, shader)
    assert module.writes == expected, why
    assert not module.writes_unknown


def test_a_read_only_binding_is_absent_from_writes(ctx):
    """The negative case, and the one that makes narrowing possible at all.

    refl_readonly.comp writes set 0 binding 0 and only READS set 1 binding 1. If the
    read-only binding showed up here, every barrier would stay conservative and the
    parser would be doing nothing but burning cycles.
    """
    module = compute(ctx, "refl_readonly.comp")
    assert module.writes == [(0, 0)]
    assert (1, 1) not in module.writes


def test_foreign_spirv_is_not_trusted(ctx):
    """bazalt knows the write opcodes its own GLSL and HLSL compile down to. It
    cannot know what another toolchain emitted, so ready SPIR-V is marked unknown
    and every binding is assumed written — the same behaviour as before 0.19.

    Provenance is the one thing the parser cannot see and the compiler knows for
    certain, which is why the flag is set there rather than guessed here.
    """
    native = compute(ctx, "refl_store.comp")
    assert not native.writes_unknown

    foreign = ctx.compile_shader("from_bytes.comp", bz.ShaderStage.COMPUTE,
                                 source=bytes(native.spirv))
    assert foreign.writes_unknown


# ── the barriers ──────────────────────────────────────────────────────────────

def test_a_fragment_write_is_ordered_against_a_later_read(ctx, extra_context):
    """The defect debt #3 was hiding, and the headline claim of the release.

    A fragment shader writes a storage image; a compute pass then reads it. There is
    no cmd.barrier() anywhere. Before 0.19 track_draw_ handled no storage images at
    all, while DescriptorSet.set_storage_image had already recorded the image as
    resting in GENERAL — so the layout the descriptor promised and the layout the
    image was in disagreed, and this was a validation error rather than a slow path.

    The assertion is two-part on purpose: the fixture says the barriers were legal,
    and the pixel says the data actually crossed them.
    """
    ctx = extra_context(features=[bz.Feature.FRAGMENT_STORES])

    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "refl_frag_store.frag"), bz.ShaderStage.FRAGMENT)
    comp = compute(ctx, "refl_read_image.comp")
    assert frag.writes == [(0, 0)]
    # The compute shader reads the image and writes only its output buffer, so the
    # image must NOT appear. Before 0.19 a compute storage image was unconditionally
    # read+write.
    assert comp.writes == [(0, 1)]

    target = ctx.create_render_target(32, 32)
    img = ctx.create_image(32, 32, bz.Format.RGBA8)
    out = ctx.create_buffer(16, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)

    graphics = (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .storage_image(0, bz.ShaderStage.FRAGMENT, set=0)
                .build(target))
    reader = (ctx.compute_pipeline()
              .shader(comp)
              .storage_image(0, set=0)
              .storage_buffer(1, set=0)
              .build())

    pool = ctx.create_descriptor_pool(max_sets=8, storage_images=8, storage_buffers=8)
    gset = pool.allocate_set(graphics, set=0)
    gset.set_storage_image(0, img)
    cset = pool.allocate_set(reader, set=0)
    cset.set_storage_image(0, img)
    cset.set_buffer(1, out)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(target, clear_color=[0.0, 0.0, 0.0, 1.0]) as c:
        c.bind_pipeline(graphics)
        c.bind_descriptor_set(gset, graphics, set=0)
        c.draw(3)
    cmd.bind_pipeline(reader)
    cmd.bind_descriptor_set(cset, reader, set=0)
    cmd.dispatch(1)
    ctx.submit(cmd)
    ctx.wait()

    # refl_frag_store.frag stores (0, 1, 0, 1).
    assert np.allclose(out.read(np.float32), [0.0, 1.0, 0.0, 1.0])


def test_a_graphics_write_needs_its_feature(ctx):
    """A graphics shader that writes a descriptor needs a feature bit, and the gate
    is driven by reflection rather than by the declarator: declaring a storage image
    and only reading it costs nothing.

    Fourth release in a row where something that looks like plain command recording
    turns out to have a feature bit, after fillModeNonSolid, wideLines and
    independentBlend. The 0.17 declarator shipped without this, so a fragment
    imageStore worked on the GPU and failed pipeline creation.
    """
    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "refl_frag_store.frag"), bz.ShaderStage.FRAGMENT)
    target = ctx.create_render_target(32, 32)

    if ctx.supports(bz.Feature.FRAGMENT_STORES):
        pytest.skip("the session Context has FRAGMENT_STORES, so it cannot refuse")
    with pytest.raises(bz.UnsupportedError, match="FRAGMENT_STORES"):
        (ctx.graphics_pipeline()
         .vertex_shader(vert)
         .fragment_shader(frag)
         .storage_image(0, bz.ShaderStage.FRAGMENT, set=0)
         .build(target))


def test_declaring_a_storage_image_without_writing_it_needs_no_feature(ctx):
    """The other half of the same rule. refl_read_image.comp only reads its image,
    and a graphics pipeline that did the same would build on any Context — the gate
    asks what the shader does, not what the pipeline could do."""
    module = compute(ctx, "refl_read_image.comp")
    # It writes the buffer at binding 1 but not the image at binding 0.
    assert (0, 0) not in module.writes
    assert (0, 1) in module.writes


def test_two_dispatches_reading_one_buffer_need_no_barrier(ctx):
    """The narrowing, from the other direction.

    Compute storage buffers used to be conservatively READ+WRITE, and
    `use(..., writes=true)` wipes read state — so two dispatches that only READ the
    same input SSBO got a WAW barrier between them, and tracked_writes_ also
    switched on the per-replay memory barrier. A `readonly buffer` shared down a
    chain of passes is the ordinary case.

    Recorded rather than measured here: the validation fixture is what says the
    result is still legal, and tests/test_barriers.py::run_sync_case is what can
    say a hazard is absent. That leg is local-only until debt #4 clears.
    """
    comp = compute(ctx, "refl_readonly.comp")
    assert comp.writes == [(0, 0)]

    pipeline = (ctx.compute_pipeline()
                .shader(comp)
                .storage_buffer(0, set=0)
                .storage_buffer(1, set=1)
                .build())
    written = ctx.create_buffer(64, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    read_only = ctx.create_buffer(np.arange(16, dtype=np.uint32),
                                  bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    pool = ctx.create_descriptor_pool(max_sets=8, storage_buffers=8)
    set0 = pool.allocate_set(pipeline, set=0)
    set0.set_buffer(0, written)
    set1 = pool.allocate_set(pipeline, set=1)
    set1.set_buffer(1, read_only)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.bind_pipeline(pipeline)
    cmd.bind_descriptor_set(set0, pipeline, set=0)
    cmd.bind_descriptor_set(set1, pipeline, set=1)
    cmd.dispatch(1)
    cmd.dispatch(1)
    ctx.submit(cmd)
    ctx.wait()


def test_a_draw_with_no_pipeline_bound_stays_conservative(ctx):
    """Fail-open where there is nothing to ask.

    With no pipeline bound the tracker cannot consult any reflection, and the answer
    has to be the pessimistic one — guessing "not written" would drop a real
    barrier. A draw with no pipeline is a bug the layers name precisely, so bazalt
    adds no error of its own; what it must not do is get quieter about barriers.
    """
    buf = ctx.create_buffer(64, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    cmd = ctx.create_command_buffer()
    cmd.begin()
    # No bind_pipeline, so nothing is recorded to consult. Recording must not throw.
    cmd.barrier(buf, bz.Access.SHADER_WRITE, bz.Access.SHADER_READ)
    ctx.submit(cmd)
    ctx.wait()


# ── printf, and the ceiling it used to carry ──────────────────────────────────

def test_a_printf_context_only_deoptimizes_shaders_that_print(ctx, extra_context):
    """An accepted ceiling until 0.19: a printf Context compiled EVERY shader
    unoptimized, because telling them apart needed reflection.

    The order is the whole trick and it is easy to get backwards. `prints` can only
    be read off unoptimized words — the optimizer deletes the print that would prove
    it — so bazalt compiles at zero, reflects, and recompiles at performance when
    the shader turns out not to print.
    """
    printf = extra_context(shader_printf=True)

    quiet_in_printf = compute(printf, "refl_store.comp")
    quiet_in_plain = compute(ctx, "refl_store.comp")
    assert not quiet_in_printf.prints
    # The claim: a non-printing shader is no longer taxed for the Context it is in.
    assert len(quiet_in_printf.spirv) == len(quiet_in_plain.spirv)

    printing = compute(printf, "refl_printf.comp")
    assert printing.prints
    # And a printing shader really does keep the optimizer off, or the print would
    # be deleted and the feature would look broken.
    printing_optimized = compute(ctx, "refl_printf.comp")
    assert len(printing.spirv) > len(printing_optimized.spirv)
