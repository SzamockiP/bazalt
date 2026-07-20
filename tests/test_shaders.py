"""The 0.7 Shader Toolbox: in-memory sources, #include, .spv loading, HLSL,
and compare samplers.

One function carries all of it: compile_shader(path, stage, source=...). The
extension of `path` picks the language (.hlsl) or format (.spv); with source=
the path is a virtual name that still supplies the language, ShaderError.path
and the #include base directory.
"""

import pathlib

import numpy as np
import pytest

import bazalt as bz

SHADER_DIR = pathlib.Path(__file__).parent / "shaders"

FRAG_OK = """#version 450
layout(location = 0) out vec4 o;
void main() { o = vec4(1.0); }
"""


def render_fullscreen(ctx, vert, frag, size=64):
    """One fullscreen pass with the given shaders; returns the pixels."""
    target = bz.RenderTarget(ctx, size, size)
    pipeline = (ctx.graphics_pipeline()
                .vertex_shader(vert)
                .fragment_shader(frag)
                .build(target))
    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(target, clear_color=[0, 0, 0, 1])
    cmd.bind_pipeline(pipeline)
    cmd.draw(3)
    cmd.end_rendering(target)
    ctx.submit(cmd)
    return target.read_pixels()


# ── in-memory sources ─────────────────────────────────────────────────────


def test_compile_from_source_without_touching_disk(ctx):
    """The path is a virtual name: it sits in a directory that does not exist,
    so a successful compile proves the file system was never consulted."""
    shader = ctx.compile_shader("no_such_dir/inline.frag", bz.ShaderStage.FRAGMENT,
                                source=FRAG_OK)
    assert shader.path == "no_such_dir/inline.frag"
    assert shader.includes == []
    assert len(shader.spirv) > 0 and len(shader.spirv) % 4 == 0


def test_source_error_reports_virtual_path_and_line(ctx):
    bad = "#version 450\nlayout(location = 0) out vec4 o;\nvoid main() { o = undefined_symbol; }\n"
    with pytest.raises(bz.ShaderError) as info:
        ctx.compile_shader("inline.frag", bz.ShaderStage.FRAGMENT, source=bad)
    assert info.value.path == "inline.frag"
    assert info.value.line == 3


# ── #include ──────────────────────────────────────────────────────────────


def write_include_tree(root, k_value="0.25"):
    """main.frag includes inc/common.glsl, which includes deeper.glsl RELATIVE
    TO ITSELF — resolution follows the including file, not the top-level one."""
    (root / "inc").mkdir(exist_ok=True)
    (root / "inc" / "deeper.glsl").write_text("const float D = 0.5;\n")
    (root / "inc" / "common.glsl").write_text(
        '#include "deeper.glsl"\n'
        f"const float K = {k_value};\n")
    main = root / "main.frag"
    main.write_text(
        "#version 450\n"
        '#include "inc/common.glsl"\n'
        "layout(location = 0) out vec4 o;\n"
        "void main() { o = vec4(K, D, 0.0, 1.0); }\n")
    return main


def test_include_resolves_relative_to_including_file(ctx, tmp_path):
    main = write_include_tree(tmp_path)
    shader = ctx.compile_shader(str(main), bz.ShaderStage.FRAGMENT)
    assert len(shader.spirv) > 0


def test_shader_records_its_includes(ctx, tmp_path):
    """shader.includes is the hot-reload contract for 0.8: the watcher watches
    path plus these. Absolute, both nesting levels, top-level file not among
    them (it is `path`)."""
    main = write_include_tree(tmp_path)
    shader = ctx.compile_shader(str(main), bz.ShaderStage.FRAGMENT)

    recorded = {pathlib.Path(p).resolve() for p in shader.includes}
    expected = {(tmp_path / "inc" / "common.glsl").resolve(),
                (tmp_path / "inc" / "deeper.glsl").resolve()}
    assert recorded == expected
    assert main.resolve() not in recorded
    assert all(pathlib.Path(p).is_absolute() for p in shader.includes)


def test_changing_included_file_changes_compile_result(ctx, tmp_path):
    """The whole point of the includer: without it, shaderc cannot see #include
    at all, and an edit to a shared .glsl would change nothing. The edit is
    semantically different (a different constant), so the optimizer cannot fold
    both versions to identical SPIR-V and green-wash the test."""
    main = write_include_tree(tmp_path, k_value="0.25")
    before = ctx.compile_shader(str(main), bz.ShaderStage.FRAGMENT).spirv

    write_include_tree(tmp_path, k_value="0.75")
    after = ctx.compile_shader(str(main), bz.ShaderStage.FRAGMENT).spirv
    assert before != after


def test_error_in_included_file_reports_that_file_and_line(ctx, tmp_path):
    """ShaderError.path names the file the error is actually in — the include,
    not the top-level shader. That is the file the user (and the 0.8 watcher)
    must open to fix it."""
    main = write_include_tree(tmp_path)
    (tmp_path / "inc" / "common.glsl").write_text(
        '#include "deeper.glsl"\n'
        "const float K = undefined_symbol;\n")

    with pytest.raises(bz.ShaderError) as info:
        ctx.compile_shader(str(main), bz.ShaderStage.FRAGMENT)
    assert pathlib.Path(info.value.path).resolve() == (tmp_path / "inc" / "common.glsl").resolve()
    assert info.value.line == 2


def test_missing_include_is_a_shader_error_naming_the_file(ctx, tmp_path):
    """A missing top-level file is a ResourceError, but a missing include is a
    ShaderError: the compiler discovered it, and hot reload needs the
    recoverable path."""
    main = tmp_path / "main.frag"
    main.write_text('#version 450\n#include "nope.glsl"\nvoid main() {}\n')
    with pytest.raises(bz.ShaderError) as info:
        ctx.compile_shader(str(main), bz.ShaderStage.FRAGMENT)
    assert "nope.glsl" in str(info.value)


# ── .spv loading ──────────────────────────────────────────────────────────


def test_spv_round_trip_produces_identical_image(ctx, tmp_path, triangle_shaders,
                                                 triangle_buffers):
    """compile → save shader.spirv → reload via compile_shader(*.spv) → the
    same triangle, byte for byte."""
    vert, frag = triangle_shaders
    vbuf, ibuf = triangle_buffers

    def render(vs, fs):
        target = bz.RenderTarget(ctx, 64, 64)
        pipeline = (ctx.graphics_pipeline()
                    .vertex_shader(vs)
                    .fragment_shader(fs)
                    .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                    .build(target))
        cmd = ctx.create_command_buffer()
        cmd.begin()
        cmd.begin_rendering(target, clear_color=[0.1, 0.2, 0.3, 1])
        cmd.bind_pipeline(pipeline)
        cmd.bind_vertex_buffer(vbuf)
        cmd.bind_index_buffer(ibuf)
        cmd.draw_indexed(3)
        cmd.end_rendering(target)
        ctx.submit(cmd)
        return target.read_pixels()

    reference = render(vert, frag)

    vs_path = tmp_path / "triangle_vert.spv"
    fs_path = tmp_path / "triangle_frag.spv"
    vs_path.write_bytes(vert.spirv)
    fs_path.write_bytes(frag.spirv)

    vert2 = ctx.compile_shader(str(vs_path), bz.ShaderStage.VERTEX)
    frag2 = ctx.compile_shader(str(fs_path), bz.ShaderStage.FRAGMENT)
    assert vert2.spirv == vert.spirv
    assert frag2.spirv == frag.spirv

    assert np.array_equal(render(vert2, frag2), reference)


def test_spv_stage_mismatch_is_a_shader_error(ctx, tmp_path):
    """The binary knows its stage (OpEntryPoint); binding a fragment .spv as
    VERTEX must be one readable error, not a validation storm."""
    frag = ctx.compile_shader("mem.frag", bz.ShaderStage.FRAGMENT, source=FRAG_OK)
    p = tmp_path / "frag.spv"
    p.write_bytes(frag.spirv)
    with pytest.raises(bz.ShaderError) as info:
        ctx.compile_shader(str(p), bz.ShaderStage.VERTEX)
    assert "VERTEX" in str(info.value)


def test_spv_garbage_is_a_shader_error(ctx, tmp_path):
    p = tmp_path / "junk.spv"
    p.write_bytes(b"\x00" * 16)
    with pytest.raises(bz.ShaderError):
        ctx.compile_shader(str(p), bz.ShaderStage.VERTEX)


def test_spv_missing_file_is_a_resource_error(ctx, tmp_path):
    with pytest.raises(bz.ResourceError):
        ctx.compile_shader(str(tmp_path / "absent.spv"), bz.ShaderStage.VERTEX)


def test_source_plus_spv_path_is_an_error(ctx):
    with pytest.raises(bz.ShaderError):
        ctx.compile_shader("x.spv", bz.ShaderStage.VERTEX, source="void main() {}")


# ── HLSL ──────────────────────────────────────────────────────────────────


def test_hlsl_triangle_matches_glsl_triangle(ctx):
    """The same fullscreen gradient from the HLSL pair and the GLSL pair —
    identical images, which also pins the winding and Y orientation."""
    vs_hlsl = ctx.compile_shader(str(SHADER_DIR / "fullscreen_vs.hlsl"), bz.ShaderStage.VERTEX)
    ps_hlsl = ctx.compile_shader(str(SHADER_DIR / "uv_gradient_ps.hlsl"), bz.ShaderStage.FRAGMENT)
    vs_glsl = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)
    ps_glsl = ctx.compile_shader(str(SHADER_DIR / "uv_gradient.frag"), bz.ShaderStage.FRAGMENT)

    hlsl_pixels = render_fullscreen(ctx, vs_hlsl, ps_hlsl)
    glsl_pixels = render_fullscreen(ctx, vs_glsl, ps_glsl)
    assert not np.array_equal(hlsl_pixels[..., :2], np.zeros_like(hlsl_pixels[..., :2])), \
        "the gradient should have drawn something"
    assert np.array_equal(hlsl_pixels, glsl_pixels)


def test_hlsl_error_is_a_shader_error(ctx):
    with pytest.raises(bz.ShaderError) as info:
        ctx.compile_shader("bad.hlsl", bz.ShaderStage.FRAGMENT,
                           source="float4 main() : SV_Target0 { return undefined; }")
    assert info.value.path == "bad.hlsl"


# ── compare samplers ──────────────────────────────────────────────────────


def test_compare_sampler_distinct_cache_entry(ctx):
    """The cache key includes the compare state: a compare sampler never
    aliases the plain one, and identical requests share one object."""
    plain = ctx.create_sampler()
    shadow = ctx.create_sampler(compare=bz.CompareOp.LESS)
    assert shadow is not plain
    assert ctx.create_sampler(compare=bz.CompareOp.LESS) is shadow
    assert ctx.create_sampler(compare=bz.CompareOp.GREATER) is not shadow


MANUAL_COMPARE_FRAG = """#version 450
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 o;
layout(set = 0, binding = 0) uniform sampler2D depthMap;
void main() {
    float lit = 0.5 < texture(depthMap, uv).r ? 1.0 : 0.0;
    o = vec4(vec3(lit), 1.0);
}
"""

SHADOW_COMPARE_FRAG = """#version 450
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 o;
layout(set = 0, binding = 0) uniform sampler2DShadow depthMap;
void main() {
    float lit = texture(depthMap, vec3(uv, 0.5));
    o = vec4(vec3(lit), 1.0);
}
"""


def test_shadow_compare_matches_manual_compare(ctx, triangle_shaders, triangle_buffers):
    """Headless twin of example 09: the hardware compare (CompareOp.LESS is
    "reference < texel", i.e. lit) must produce the same image as the manual
    in-shader comparison. NEAREST on both samplers: with linear filtering the
    two are legitimately different (PCF averages comparison RESULTS, the manual
    path compares an averaged DEPTH), so only NEAREST is bit-comparable."""
    vert, _ = triangle_shaders
    vbuf, ibuf = triangle_buffers

    shadow = bz.RenderTarget(ctx, 64, 64, color=None, depth=bz.Format.D32F)
    depth_pipe = (ctx.graphics_pipeline()
                  .vertex_shader(vert)
                  .vertex_format([bz.VertexFormat.FLOAT3, bz.VertexFormat.FLOAT3])
                  .depth_test(True)
                  .build(shadow))
    cmd = ctx.create_command_buffer()
    cmd.begin()
    cmd.begin_rendering(shadow)
    cmd.bind_pipeline(depth_pipe)
    cmd.bind_vertex_buffer(vbuf)
    cmd.bind_index_buffer(ibuf)
    cmd.draw_indexed(3)
    cmd.end_rendering(shadow)
    ctx.submit(cmd)

    fullscreen = ctx.compile_shader(str(SHADER_DIR / "fullscreen.vert"), bz.ShaderStage.VERTEX)

    def compare_pass(frag_source, name, sampler):
        frag = ctx.compile_shader(name, bz.ShaderStage.FRAGMENT, source=frag_source)
        screen = bz.RenderTarget(ctx, 64, 64)
        pipeline = (ctx.graphics_pipeline()
                    .vertex_shader(fullscreen)
                    .fragment_shader(frag)
                    .texture(0, bz.ShaderStage.FRAGMENT, set=0)
                    .build(screen))
        pool = ctx.create_descriptor_pool(max_sets=4, samplers=4)
        dset = pool.allocate_set(pipeline, set=0)
        dset.set_image(0, shadow.depth, sampler=sampler)

        cmd = ctx.create_command_buffer()
        cmd.begin()
        cmd.begin_rendering(screen, clear_color=[0, 0, 0, 1])
        cmd.bind_pipeline(pipeline)
        cmd.bind_descriptor_set(dset, pipeline, set=0)
        cmd.draw(3)
        cmd.end_rendering(screen)
        ctx.submit(cmd)
        return screen.read_pixels()

    manual = compare_pass(MANUAL_COMPARE_FRAG, "manual_cmp.frag",
                          ctx.create_sampler(filter=bz.Filter.NEAREST))
    hardware = compare_pass(SHADOW_COMPARE_FRAG, "shadow_cmp.frag",
                            ctx.create_sampler(filter=bz.Filter.NEAREST,
                                               compare=bz.CompareOp.LESS))

    assert not np.array_equal(manual, np.zeros_like(manual)), "scene should not be all-shadow"
    assert np.array_equal(manual, hardware)
