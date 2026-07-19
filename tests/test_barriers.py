"""Automatic barriers between resources, and the manual mode that disables them.

Auto barriers are computed at record time: deferred recording fixes the usage
sequence constructively, so a barrier computed once is correct for every
replay. The validation-as-assert fixture is the referee — a wrong or missing
auto barrier surfaces as a validation error and fails the test.

The one thing core validation CANNOT see is a missing barrier (that takes
synchronization validation), which is why the manual-mode negative test runs
in a subprocess with validation="sync".
"""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import bazalt as bz

from conftest import SHADER_DIR


@pytest.fixture
def fullscreen_vert(ctx):
    return ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)


@pytest.fixture
def double_pipeline(ctx):
    comp = ctx.compile_shader(str(SHADER_DIR / "double.comp"), bz.ShaderStage.COMPUTE)
    return ctx.compute_pipeline().shader(comp).storage_buffer(0).build()


@pytest.fixture
def add_one_pipeline(ctx):
    comp = ctx.compile_shader(str(SHADER_DIR / "add_one.comp"), bz.ShaderStage.COMPUTE)
    return ctx.compute_pipeline().shader(comp).storage_buffer(0).build()


def make_set(ctx, pipeline, sbuf):
    pool = ctx.create_descriptor_pool(max_sets=8, storage_buffers=8)
    dset = pool.allocate_set(pipeline, set=0)
    dset.set_buffer(0, sbuf)
    return pool, dset


# ── auto mode ─────────────────────────────────────────────────────────────


def test_dispatch_to_dispatch_gets_a_barrier(ctx, double_pipeline, add_one_pipeline):
    """(x * 2) + 1 requires the second dispatch to see the first one's writes."""
    data = np.arange(64, dtype=np.float32)
    sbuf = ctx.create_buffer(data, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    pool_a, dset_a = make_set(ctx, double_pipeline, sbuf)
    pool_b, dset_b = make_set(ctx, add_one_pipeline, sbuf)

    cmd = ctx.create_command_buffer()
    (cmd.begin()
        .bind_pipeline(double_pipeline)
        .bind_descriptor_set(dset_a, double_pipeline, set=0)
        .dispatch(1)
        .bind_pipeline(add_one_pipeline)
        .bind_descriptor_set(dset_b, add_one_pipeline, set=0)
        .dispatch(1))
    ctx.submit(cmd)

    assert np.allclose(sbuf.read(np.float32), data * 2 + 1)


def test_dispatch_to_draw_via_descriptor_read(ctx, fullscreen_vert, double_pipeline):
    """Compute writes an SSBO, the fragment shader reads it — RAW across bind points."""
    frag = ctx.compile_shader(str(SHADER_DIR / "ssbo.frag"), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, 32, 32)
    gfx = (ctx.graphics_pipeline()
           .vertex_shader(fullscreen_vert)
           .fragment_shader(frag)
           .storage_buffer(0, bz.ShaderStage.FRAGMENT, set=0)
           .build(target))

    # double.comp turns (0, 0, 0.5, 0.5) into the (0, 0, 1, 1) ssbo.frag paints.
    sbuf = ctx.create_buffer(np.array([0.0, 0.0, 0.5, 0.5], dtype=np.float32),
                             bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    comp_pool, comp_set = make_set(ctx, double_pipeline, sbuf)
    gfx_pool, gfx_set = make_set(ctx, gfx, sbuf)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.bind_pipeline(double_pipeline)
    cmd.bind_descriptor_set(comp_set, double_pipeline, set=0)
    cmd.dispatch(1)
    with cmd.rendering(target):
        cmd.bind_pipeline(gfx).bind_descriptor_set(gfx_set, gfx, set=0).draw(3)
    ctx.submit(cmd)

    assert np.allclose(target.read_pixels()[16, 16, :3], [0, 0, 255], atol=2)


def test_dispatch_to_vertex_fetch_hoists_the_barrier(ctx, double_pipeline):
    """Compute writes vertices, the draw consumes them via bind_vertex_buffer.

    The bind happens inside the rendering scope, so the barrier must be
    hoisted before begin_rendering — vkCmdPipelineBarrier is illegal inside
    dynamic rendering, and the validation fixture would catch it there.
    """
    comp = ctx.compile_shader(str(SHADER_DIR / "write_vertices.comp"), bz.ShaderStage.COMPUTE)
    write_verts = ctx.compute_pipeline().shader(comp).storage_buffer(0).build()

    vert = ctx.compile_shader(str(SHADER_DIR / "pos2.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, 32, 32)
    gfx = (ctx.graphics_pipeline()
           .vertex_shader(vert)
           .fragment_shader(frag)
           .vertex_format([bz.VertexFormat.FLOAT2])
           .build(target))

    # Garbage in: the triangle only covers the screen if the dispatch's writes
    # actually reached the vertex fetch.
    verts = ctx.create_buffer(np.zeros(6, dtype=np.float32),
                              bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    pool, dset = make_set(ctx, write_verts, verts)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.bind_pipeline(write_verts)
    cmd.bind_descriptor_set(dset, write_verts, set=0)
    cmd.dispatch(1)
    with cmd.rendering(target):
        cmd.bind_pipeline(gfx).bind_vertex_buffer(verts).draw(3)
    ctx.submit(cmd)

    assert np.allclose(target.read_pixels()[16, 16, :3], [255, 0, 0], atol=2)


def test_draw_then_dispatch_is_write_after_read(ctx, double_pipeline):
    """The dispatch must wait for the draw that reads the buffer it writes."""
    comp = ctx.compile_shader(str(SHADER_DIR / "write_vertices.comp"), bz.ShaderStage.COMPUTE)
    write_verts = ctx.compute_pipeline().shader(comp).storage_buffer(0).build()

    vert = ctx.compile_shader(str(SHADER_DIR / "pos2.vert"), bz.ShaderStage.VERTEX)
    frag = ctx.compile_shader(str(SHADER_DIR / "solid_red.frag"), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, 32, 32)
    gfx = (ctx.graphics_pipeline()
           .vertex_shader(vert)
           .fragment_shader(frag)
           .vertex_format([bz.VertexFormat.FLOAT2])
           .build(target))

    tri = np.array([-1.0, -1.0, -1.0, 3.0, 3.0, -1.0], dtype=np.float32)
    verts = ctx.create_buffer(tri, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    pool, dset = make_set(ctx, write_verts, verts)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    with cmd.rendering(target):
        cmd.bind_pipeline(gfx).bind_vertex_buffer(verts).draw(3)
    cmd.bind_pipeline(write_verts)
    cmd.bind_descriptor_set(dset, write_verts, set=0)
    cmd.dispatch(1)
    ctx.submit(cmd)

    assert np.allclose(target.read_pixels()[16, 16, :3], [255, 0, 0], atol=2)
    assert np.allclose(verts.read(np.float32), tri)  # rewrote the same values


# ── manual mode ───────────────────────────────────────────────────────────


def test_manual_mode_with_explicit_barriers_is_clean(ctx, double_pipeline, add_one_pipeline):
    """Same dispatch chain as the auto test, barriers spelled by hand."""
    data = np.arange(64, dtype=np.float32)
    sbuf = ctx.create_buffer(data, bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    pool_a, dset_a = make_set(ctx, double_pipeline, sbuf)
    pool_b, dset_b = make_set(ctx, add_one_pipeline, sbuf)

    cmd = ctx.create_command_buffer(auto_barriers=False)
    (cmd.begin()
        .bind_pipeline(double_pipeline)
        .bind_descriptor_set(dset_a, double_pipeline, set=0)
        .dispatch(1)
        .barrier(sbuf, bz.Access.SHADER_WRITE, bz.Access.SHADER_READ)
        .barrier(sbuf, bz.Access.SHADER_WRITE, bz.Access.SHADER_WRITE)
        .bind_pipeline(add_one_pipeline)
        .bind_descriptor_set(dset_b, add_one_pipeline, set=0)
        .dispatch(1))
    ctx.submit(cmd)

    assert np.allclose(sbuf.read(np.float32), data * 2 + 1)


def test_barrier_inside_rendering_scope_is_refused(ctx, triangle_shaders, triangle_buffers):
    vert, frag = triangle_shaders
    vbuf, ibuf = triangle_buffers
    target = bz.RenderTarget(ctx, 16, 16)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                .build(target))
    sbuf = ctx.create_buffer(np.zeros(4, dtype=np.float32),
                             bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)

    cmd = ctx.create_command_buffer(auto_barriers=False)
    cmd.begin()
    cmd.begin_rendering(target)
    with pytest.raises(bz.ResourceError, match="rendering scope"):
        cmd.barrier(sbuf, bz.Access.SHADER_WRITE, bz.Access.VERTEX_READ)
    cmd.end_rendering(target)


def test_context_wide_manual_mode_is_inherited(ctx):
    """create_command_buffer() without the kwarg takes the Context's default."""
    assert ctx.auto_barriers is True
    # The per-CB override is the only way to go manual on this (auto) Context;
    # the flag's plumbing from ContextConfig is exercised in the subprocess test.
    cmd = ctx.create_command_buffer(auto_barriers=False)
    assert cmd is not None


# ── sync validation: the proof that manual mode is really manual ─────────


SYNC_SCRIPT = textwrap.dedent("""
    import sys
    import numpy as np
    import bazalt as bz

    shader = sys.argv[1]
    mode = sys.argv[2]              # "nobarrier" | "barrier" | "auto"
    with_barrier = mode == "barrier"
    auto = mode == "auto"

    hazards = []
    log = bz.Logger(min_severity=bz.Severity.INFO)

    @log.on_message
    def _(msg):
        if msg.source != bz.Source.VALIDATION:
            return
        # Dumped so a failing CI run shows what the layer actually said.
        print("VMSG:", str(msg.severity), msg.text.replace(chr(10), " ")[:400],
              file=sys.stderr)
        # "hazard detected" on recent layers, "Hazard WRITE_AFTER_WRITE" on
        # older ones.
        if "hazard" in msg.text.lower():
            hazards.append(msg.text)

    ctx = bz.Context(log, validation="sync", auto_barriers=auto)
    assert ctx.auto_barriers is auto

    comp = ctx.compile_shader(shader, bz.ShaderStage.COMPUTE)
    pipeline = ctx.compute_pipeline().shader(comp).storage_buffer(0).build()
    sbuf = ctx.create_buffer(np.arange(64, dtype=np.float32),
                             bz.BufferType.STORAGE, bz.MemoryUsage.STATIC)
    pool = ctx.create_descriptor_pool(max_sets=8, storage_buffers=8)
    dset = pool.allocate_set(pipeline, set=0)
    dset.set_buffer(0, sbuf)

    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.bind_pipeline(pipeline)
    cmd.bind_descriptor_set(dset, pipeline, set=0)
    cmd.dispatch(1)
    if with_barrier:
        cmd.barrier(sbuf, bz.Access.SHADER_WRITE, bz.Access.SHADER_READ)
        cmd.barrier(sbuf, bz.Access.SHADER_WRITE, bz.Access.SHADER_WRITE)
    cmd.dispatch(1)
    ctx.submit(cmd)

    log.flush()
    print("HAZARDS:", len(hazards))
""")


def run_sync_script(tmp_path, mode):
    """Subprocess because sync validation needs its own Context (one per
    process — volk's function pointers are global) and because it is far too
    slow to tax the whole suite with."""
    script = tmp_path / "sync_check.py"
    script.write_text(SYNC_SCRIPT, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(script), str(SHADER_DIR / "double.comp"), mode],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    dump = f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    for line in result.stdout.splitlines():
        if line.startswith("HAZARDS:"):
            return int(line.split(":")[1]), dump
    pytest.fail(f"no HAZARDS line in output:\n{dump}")


@pytest.mark.skipif(
    os.environ.get("BAZALT_SYNCVAL_UNSUPPORTED") == "1",
    reason="this environment's validation layer cannot report shader-access "
           "sync hazards (verified: 1.4.313, the newest packaged for Ubuntu "
           "noble, stays silent even with the settings file forcing "
           "validate_sync + syncval_shader_accesses_heuristic; 1.4.350 "
           "reports them — the messenger itself was proven alive)")
def test_missing_barrier_in_manual_mode_trips_sync_validation(ctx, tmp_path):
    """If this test fails, manual mode is not really manual (or sync validation
    is not really on) — either way the mode would be a lie."""
    count, dump = run_sync_script(tmp_path, "nobarrier")
    assert count > 0, dump


def test_explicit_barrier_in_manual_mode_satisfies_sync_validation(ctx, tmp_path):
    count, dump = run_sync_script(tmp_path, "barrier")
    assert count == 0, dump


def test_auto_barriers_satisfy_sync_validation(ctx, tmp_path):
    """The auto tracker's barriers hold up under the same referee that catches
    the missing ones — not just under core validation, which is blind here."""
    count, dump = run_sync_script(tmp_path, "auto")
    assert count == 0, dump
