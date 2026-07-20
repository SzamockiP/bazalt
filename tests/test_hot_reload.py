"""Hot reload (0.8): edit a shader or an image on disk and the running Context
picks it up — shaders recompile and rebuild their pipelines in place, images
re-upload into the same VkImage. A bad edit (typo, wrong size, corrupt file) is
logged and the last good version keeps rendering; the application never dies.

The whole suite runs with hot_reload=True (see conftest), and the drain that
applies changes runs on every ctx.submit(), so a headless test edits a file,
submits, and reads back the result. Files are written through touch()/bump so
their mtime always moves — coarse filesystem timestamps (CI) would otherwise
hide the change from the mtime-polling watcher.
"""

import gc
import os
import pathlib
import time

import numpy as np
import pytest

import bazalt as bz
from test_bindings import write_png  # stdlib-only PNG writer

SHADER_DIR = pathlib.Path(__file__).parent / "shaders"
POLL = float(os.environ.get("BAZALT_HOT_RELOAD_POLL_MS", "20")) / 1000.0


# ── helpers ────────────────────────────────────────────────────────────────


def solid_frag(r, g, b):
    return (
        "#version 450\n"
        "layout(location = 0) out vec4 o;\n"
        f"void main() {{ o = vec4({r}.0, {g}.0, {b}.0, 1.0); }}\n"
    )


def bump_mtime(path):
    """Push the file's mtime forward so a != comparison always sees the change,
    even on filesystems whose timestamp resolution is coarser than the poll."""
    st = path.stat()
    os.utime(path, (st.st_atime, st.st_mtime + 2))


def touch(path, text):
    path.write_text(text)
    bump_mtime(path)


def write_png_bumped(path, rows):
    write_png(path, rows)
    bump_mtime(path)


def center(px):
    return px[px.shape[0] // 2, px.shape[1] // 2]


def is_red(px):
    c = center(px)
    return c[0] > 200 and c[1] < 60 and c[2] < 60


def is_green(px):
    c = center(px)
    return c[1] > 200 and c[0] < 60 and c[2] < 60


def poll_until(render, predicate, timeout=5.0):
    """Render (which submits, and so drains the watcher) until the pixels satisfy
    predicate or the timeout elapses. Returns the last frame either way."""
    deadline = time.monotonic() + timeout
    px = render()
    while not predicate(px) and time.monotonic() < deadline:
        time.sleep(POLL * 3)
        px = render()
    return px


def fullscreen_pipeline(ctx, frag, target):
    vert = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    return (ctx.graphics_pipeline()
            .vertex_shader(vert)
            .fragment_shader(frag)
            .build(target))


def make_renderer(ctx, target, pipeline):
    def render():
        cmd = ctx.create_command_buffer()
        cmd.begin()
        cmd.begin_rendering(target, clear_color=[0, 0, 0, 1])
        cmd.bind_pipeline(pipeline)
        cmd.draw(3)
        cmd.end_rendering(target)
        ctx.submit(cmd)  # drains pending hot reloads
        return target.read_pixels()
    return render


def drive_drain(ctx):
    """A clear-only submit whose only purpose is to run the watcher drain."""
    t = bz.RenderTarget(ctx, 8, 8)
    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(t, clear_color=[0, 0, 0, 1])
    cmd.end_rendering(t)
    ctx.submit(cmd)


# ── shaders ────────────────────────────────────────────────────────────────


def test_editing_a_fragment_shader_rebuilds_the_pipeline(ctx, tmp_path):
    frag_path = tmp_path / "solid.frag"
    frag_path.write_text(solid_frag(1, 0, 0))
    frag = ctx.compile_shader(str(frag_path), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, 32, 32)
    render = make_renderer(ctx, target, fullscreen_pipeline(ctx, frag, target))

    assert is_red(render())

    # A semantic change, not a cosmetic one: the optimizer would fold a
    # comment-only edit back to identical SPIR-V and the test would lie green.
    touch(frag_path, solid_frag(0, 1, 0))
    assert is_green(poll_until(render, is_green))


def test_a_typo_is_logged_and_the_old_pipeline_survives(ctx, tmp_path, messages):
    frag_path = tmp_path / "solid.frag"
    frag_path.write_text(solid_frag(1, 0, 0))
    frag = ctx.compile_shader(str(frag_path), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, 32, 32)
    render = make_renderer(ctx, target, fullscreen_pipeline(ctx, frag, target))
    assert is_red(render())

    touch(frag_path, "#version 450\nthis is not valid glsl\n")

    def shader_error():
        return any(m.source == bz.Source.SHADER and m.severity >= bz.Severity.ERROR
                   for m in messages())

    deadline = time.monotonic() + 5.0
    while not shader_error() and time.monotonic() < deadline:
        time.sleep(POLL * 3)
        render()

    assert shader_error(), "a shader typo must reach the logger as a SHADER error"
    assert is_red(render()), "the last good pipeline must keep rendering"

    # And recovery: fixing the file brings the reload back to life.
    touch(frag_path, solid_frag(0, 1, 0))
    assert is_green(poll_until(render, is_green))


def test_editing_an_included_file_rebuilds_dependent_pipelines(ctx, tmp_path):
    include = tmp_path / "tint.glsl"
    include.write_text("vec3 tint() { return vec3(1.0, 0.0, 0.0); }\n")
    frag_path = tmp_path / "uses_include.frag"
    frag_path.write_text(
        "#version 450\n"
        'layout(location = 0) out vec4 o;\n'
        '#include "tint.glsl"\n'
        "void main() { o = vec4(tint(), 1.0); }\n"
    )
    frag = ctx.compile_shader(str(frag_path), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, 32, 32)
    render = make_renderer(ctx, target, fullscreen_pipeline(ctx, frag, target))
    assert is_red(render())

    # The .frag is untouched; only the include changes.
    touch(include, "vec3 tint() { return vec3(0.0, 1.0, 0.0); }\n")
    assert is_green(poll_until(render, is_green))


def test_dropping_an_include_stops_watching_it(ctx, tmp_path, messages):
    include = tmp_path / "tint.glsl"
    include.write_text("vec3 tint() { return vec3(1.0, 0.0, 0.0); }\n")
    frag_path = tmp_path / "drops_include.frag"
    frag_path.write_text(
        "#version 450\n"
        'layout(location = 0) out vec4 o;\n'
        '#include "tint.glsl"\n'
        "void main() { o = vec4(tint(), 1.0); }\n"
    )
    frag = ctx.compile_shader(str(frag_path), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, 32, 32)
    render = make_renderer(ctx, target, fullscreen_pipeline(ctx, frag, target))
    assert is_red(render())

    # Rewrite the frag to inline green and drop the #include entirely.
    touch(frag_path, solid_frag(0, 1, 0))
    assert is_green(poll_until(render, is_green))

    # The include is no longer part of the module, so editing it must do nothing
    # — no colour change, no error. (The watch table was re-derived after the
    # recompile above.)
    start = len(messages())
    touch(include, "vec3 tint() { return vec3(0.0, 0.0, 1.0); }\n")
    for _ in range(10):
        time.sleep(POLL * 3)
        px = render()
    assert is_green(px), "an edit to a no-longer-included file must not change output"
    assert not [m for m in messages()[start:]
                if m.source == bz.Source.SHADER and m.severity >= bz.Severity.ERROR]


def test_a_dropped_pipeline_is_pruned_not_rebuilt(ctx, tmp_path, messages):
    frag_path = tmp_path / "orphan.frag"
    frag_path.write_text(solid_frag(1, 0, 0))
    frag = ctx.compile_shader(str(frag_path), bz.ShaderStage.FRAGMENT)
    target = bz.RenderTarget(ctx, 16, 16)
    pipeline = fullscreen_pipeline(ctx, frag, target)
    make_renderer(ctx, target, pipeline)()  # render once so it's fully live

    # Drop every reference, then change the file it was built from.
    del pipeline, frag
    gc.collect()
    start = len(messages())
    touch(frag_path, solid_frag(0, 1, 0))

    # Draining over a dead weak_ptr must not crash or log — the pipeline is just
    # pruned. validation-as-assert (the ctx fixture) is the real judge here.
    for _ in range(10):
        time.sleep(POLL * 3)
        drive_drain(ctx)
    assert not [m for m in messages()[start:]
                if m.severity >= bz.Severity.ERROR]


# ── images ─────────────────────────────────────────────────────────────────

RED = (255, 0, 0, 255)
GREEN = (0, 255, 0, 255)


def test_reloading_a_same_size_image_updates_its_contents(ctx, tmp_path):
    png = tmp_path / "tex.png"
    write_png(png, [[RED] * 4] * 4)
    img = ctx.load_image(str(png))
    img.wait()
    assert is_red(img.read())

    write_png_bumped(png, [[GREEN] * 4] * 4)
    deadline = time.monotonic() + 5.0
    px = img.read()
    while not is_green(px) and time.monotonic() < deadline:
        time.sleep(POLL * 2)
        drive_drain(ctx)     # drain enqueues the reload onto the upload worker
        px = img.read()      # blocks on the current upload serial
    assert is_green(px)
    assert img.mip_levels >= 1


def test_reloading_a_wrong_size_image_warns_and_keeps_the_old_one(ctx, tmp_path, messages):
    png = tmp_path / "tex.png"
    write_png(png, [[RED] * 4] * 4)
    img = ctx.load_image(str(png))
    img.wait()
    assert is_red(img.read())

    start = len(messages())
    write_png_bumped(png, [[GREEN] * 8] * 8)   # 8x8, not 4x4

    def upload_warning():
        return any(m.source == bz.Source.UPLOAD and m.severity >= bz.Severity.WARNING
                   for m in messages()[start:])

    deadline = time.monotonic() + 5.0
    while not upload_warning() and time.monotonic() < deadline:
        time.sleep(POLL * 2)
        drive_drain(ctx)

    assert upload_warning(), "a resize must warn on Source.UPLOAD"
    assert img.width == 4 and img.height == 4
    assert is_red(img.read()), "the existing image must be untouched"


def test_reloading_a_corrupt_image_warns_and_survives(ctx, tmp_path, messages):
    from test_uploads import write_corrupt_png

    png = tmp_path / "tex.png"
    write_png(png, [[RED] * 4] * 4)
    img = ctx.load_image(str(png))
    img.wait()
    assert is_red(img.read())

    start = len(messages())
    write_corrupt_png(png)
    bump_mtime(png)

    def upload_warning():
        return any(m.source == bz.Source.UPLOAD and m.severity >= bz.Severity.WARNING
                   for m in messages()[start:])

    deadline = time.monotonic() + 5.0
    while not upload_warning() and time.monotonic() < deadline:
        time.sleep(POLL * 2)
        drive_drain(ctx)

    assert upload_warning(), "a corrupt file must warn, not raise"
    assert img.ready, "the image stays usable"
    assert is_red(img.read()), "the previous contents remain"
