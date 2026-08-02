"""The 0.7 Shader Toolbox: in-memory sources, #include, .spv loading, HLSL,
and compare samplers.

compile_shader is two overloads since 0.24, one per place the code comes from.
compile_shader(path, stage) reads a file, and the extension picks the language
(.hlsl) or the format (.spv) unless language= overrides it.
compile_shader(source=, stage=) takes the code itself, with language= and an
optional name= for diagnostics — no filename to invent.
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
    target = ctx.create_render_target(size, size)
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
    return target.color[0].read()


# ── in-memory sources ─────────────────────────────────────────────────────


def test_compile_from_source_needs_no_path(ctx):
    """0.24: the source form takes no path at all. Before it, in-memory code had
    to invent a filename, because the extension of that filename was what chose
    the parser."""
    shader = ctx.compile_shader(source=FRAG_OK, stage=bz.ShaderStage.FRAGMENT)
    assert shader.includes == []
    assert len(shader.spirv) > 0 and len(shader.spirv) % 4 == 0


def test_unnamed_source_says_so_in_errors(ctx):
    """Something has to appear where a path would. `<source>` reads as "this
    came from memory" rather than as a file somebody could go and open."""
    bad = "#version 450\nlayout(location = 0) out vec4 o;\nvoid main() { o = undefined_symbol; }\n"
    with pytest.raises(bz.ShaderError) as info:
        ctx.compile_shader(source=bad, stage=bz.ShaderStage.FRAGMENT)
    assert info.value.path == "<source>"
    assert info.value.line == 3


def test_source_error_reports_the_name_and_line(ctx):
    bad = "#version 450\nlayout(location = 0) out vec4 o;\nvoid main() { o = undefined_symbol; }\n"
    with pytest.raises(bz.ShaderError) as info:
        ctx.compile_shader(source=bad, stage=bz.ShaderStage.FRAGMENT, name="inline.frag")
    assert info.value.path == "inline.frag"
    assert info.value.line == 3


def test_name_is_only_a_label(ctx):
    """The point of the split: a name that looks like a path changes nothing.
    `.hlsl` does not select HLSL here and `.spv` does not mean "already
    compiled" — both of those are jobs the path form owns."""
    assert ctx.compile_shader(source=FRAG_OK, stage=bz.ShaderStage.FRAGMENT,
                              name="looks_like.hlsl").path == "looks_like.hlsl"
    assert ctx.compile_shader(source=FRAG_OK, stage=bz.ShaderStage.FRAGMENT,
                              name="looks_like.spv").path == "looks_like.spv"


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


def test_include_dirs_are_a_fallback(ctx, tmp_path):
    """0.16: a search path, tried only when the name is not beside the including
    file. A shader that already compiled must keep resolving the same way, so
    the primary rule stays first."""
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "shared.glsl").write_text("const float S = 0.5;\n")

    main = tmp_path / "uses_lib.frag"
    main.write_text(
        "#version 450\n"
        '#include "shared.glsl"\n'
        "layout(location = 0) out vec4 o;\n"
        "void main() { o = vec4(S, 0.0, 0.0, 1.0); }\n")

    # Without the search path there is no shared.glsl anywhere near the shader.
    with pytest.raises(bz.ShaderError):
        ctx.compile_shader(str(main), bz.ShaderStage.FRAGMENT)

    shader = ctx.compile_shader(str(main), bz.ShaderStage.FRAGMENT, include_dirs=[str(lib)])
    assert len(shader.spirv) > 0
    recorded = {pathlib.Path(p).resolve() for p in shader.includes}
    assert recorded == {(lib / "shared.glsl").resolve()}


def test_a_neighbouring_file_wins_over_the_search_path(ctx, tmp_path):
    """The order is what keeps include_dirs safe to add: a directory added later
    must not shadow a file the shader has been including all along."""
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "shared.glsl").write_text("const float S = 0.25;\n")
    (tmp_path / "shared.glsl").write_text("const float S = 0.75;\n")

    main = tmp_path / "prefers_neighbour.frag"
    main.write_text(
        "#version 450\n"
        '#include "shared.glsl"\n'
        "layout(location = 0) out vec4 o;\n"
        "void main() { o = vec4(S, 0.0, 0.0, 1.0); }\n")

    shader = ctx.compile_shader(str(main), bz.ShaderStage.FRAGMENT, include_dirs=[str(lib)])
    recorded = {pathlib.Path(p).resolve() for p in shader.includes}
    assert recorded == {(tmp_path / "shared.glsl").resolve()}


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
        target = ctx.create_render_target(64, 64)
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
        return target.color[0].read()

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
    frag = ctx.compile_shader(source=FRAG_OK, stage=bz.ShaderStage.FRAGMENT)
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


def test_language_on_a_spv_path_is_refused(ctx, tmp_path):
    """language= says how to PARSE, and a .spv file is already parsed. Refused
    rather than ignored: an argument that quietly does nothing is the failure
    mode the 0.24 split exists to remove."""
    p = tmp_path / "prebuilt.spv"
    p.write_bytes(b"\x00" * 16)
    with pytest.raises(ValueError, match="already-compiled"):
        ctx.compile_shader(str(p), bz.ShaderStage.VERTEX, language=bz.ShaderLanguage.GLSL)


def test_language_with_spirv_bytes_is_refused(ctx):
    frag = ctx.compile_shader(source=FRAG_OK, stage=bz.ShaderStage.FRAGMENT)
    with pytest.raises(ValueError, match="ready SPIR-V"):
        ctx.compile_shader(source=frag.spirv, stage=bz.ShaderStage.FRAGMENT,
                           language=bz.ShaderLanguage.GLSL)


def test_the_two_forms_do_not_blend(ctx):
    """`path` and `source=` are separate overloads since 0.24, so the fused call
    that used to be the only spelling no longer resolves."""
    with pytest.raises(TypeError):
        ctx.compile_shader("x.frag", bz.ShaderStage.FRAGMENT, source=FRAG_OK)


# ── SPIR-V straight from memory (0.16) ────────────────────────────────────


def test_source_bytes_are_taken_as_spirv(ctx, triangle_shaders):
    """0.16: source= is text when it is str and SPIR-V when it is bytes.

    The type is the whole signal, and it has to be, since pybind converts both
    to std::string — an implementation that ignored it would try to compile a
    binary as GLSL.
    """
    vert, _ = triangle_shaders
    reloaded = ctx.compile_shader(source=vert.spirv, stage=bz.ShaderStage.VERTEX)
    assert reloaded.spirv == vert.spirv


def test_spirv_bytes_get_the_same_checks_as_a_file(ctx):
    """Bytes from memory deserve the diagnostics a .spv file gets, because the
    two paths share one validator."""
    frag = ctx.compile_shader(source=FRAG_OK, stage=bz.ShaderStage.FRAGMENT)

    with pytest.raises(bz.ShaderError, match="VERTEX"):
        ctx.compile_shader(source=frag.spirv, stage=bz.ShaderStage.VERTEX)

    with pytest.raises(bz.ShaderError, match="4-byte marker"):
        ctx.compile_shader(source=b"\x00" * 16, stage=bz.ShaderStage.FRAGMENT)

    # Not a whole number of 32-bit words: the wrong argument, not a truncated
    # binary, so the message says so instead of reporting the missing marker.
    with pytest.raises(bz.ShaderError, match="multiple of 4"):
        ctx.compile_shader(source=b"\x00" * 15, stage=bz.ShaderStage.FRAGMENT)


def test_spirv_bytes_render(ctx, triangle_shaders, triangle_buffers):
    """The round trip that makes the feature worth having: compile once, keep
    the words, build a pipeline from them later with no file involved."""
    vert, frag = triangle_shaders
    vbuf, ibuf = triangle_buffers

    def render(vs, fs):
        target = ctx.create_render_target(64, 64)
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
        return target.color[0].read()

    reference = render(vert, frag)
    from_bytes = render(
        ctx.compile_shader(source=vert.spirv, stage=bz.ShaderStage.VERTEX),
        ctx.compile_shader(source=frag.spirv, stage=bz.ShaderStage.FRAGMENT))
    assert np.array_equal(from_bytes, reference)


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
        ctx.compile_shader(source="float4 main() : SV_Target0 { return undefined; }",
                           stage=bz.ShaderStage.FRAGMENT,
                           language=bz.ShaderLanguage.HLSL,
                           name="bad.hlsl")
    assert info.value.path == "bad.hlsl"


def test_hlsl_from_memory_needs_no_hlsl_filename(ctx):
    """Until 0.24 this call had to pass a made-up path ending in .hlsl, because
    the extension was the only way to reach the HLSL parser. language= is that
    way now, and the shader has no name at all."""
    ps = ctx.compile_shader(source="float4 main() : SV_Target0 { return float4(1, 0, 0, 1); }",
                            stage=bz.ShaderStage.FRAGMENT,
                            language=bz.ShaderLanguage.HLSL)
    assert len(ps.spirv) > 0


def test_language_overrides_the_extension(ctx, tmp_path):
    """A file holding HLSL under a GLSL name. It had no spelling before 0.24:
    the only fix was renaming the file, which is not always the caller's to do
    (a shader pack, a generated file, a vendored tree).
    """
    misnamed = tmp_path / "gradient.frag"
    misnamed.write_text((SHADER_DIR / "uv_gradient_ps.hlsl").read_text())

    # Read as GLSL — which is what the extension says — it does not compile.
    with pytest.raises(bz.ShaderError):
        ctx.compile_shader(str(misnamed), bz.ShaderStage.FRAGMENT)

    ps = ctx.compile_shader(str(misnamed), bz.ShaderStage.FRAGMENT,
                            language=bz.ShaderLanguage.HLSL)
    reference = render_fullscreen(
        ctx,
        ctx.compile_shader(str(SHADER_DIR / "fullscreen_vs.hlsl"), bz.ShaderStage.VERTEX),
        ctx.compile_shader(str(SHADER_DIR / "uv_gradient_ps.hlsl"), bz.ShaderStage.FRAGMENT))
    assert np.array_equal(
        render_fullscreen(ctx, ctx.compile_shader(str(SHADER_DIR / "fullscreen_vs.hlsl"),
                                                  bz.ShaderStage.VERTEX), ps),
        reference)


def test_language_reaches_the_entry_point_gate(ctx, tmp_path):
    """language= must drive every decision the language drives, not only which
    parser shaderc runs. entry_point= is refused for GLSL, so a .frag declared
    HLSL has to be allowed to name one — otherwise the override would be half
    applied and the failure would read as a bug in the gate."""
    misnamed = tmp_path / "pack.frag"
    misnamed.write_text((SHADER_DIR / "two_entry_points.hlsl").read_text())

    ps = ctx.compile_shader(str(misnamed), bz.ShaderStage.FRAGMENT,
                            language=bz.ShaderLanguage.HLSL, entry_point="PSMain")
    assert len(ps.spirv) > 0

    # And the reflection follows too: a misspelled entry point is still caught.
    with pytest.raises(bz.ShaderError, match="matches no function"):
        ctx.compile_shader(str(misnamed), bz.ShaderStage.FRAGMENT,
                           language=bz.ShaderLanguage.HLSL, entry_point="NoSuchFunction")


def test_entry_point_picks_a_function_out_of_one_hlsl_file(ctx):
    """0.16: one .hlsl holding VSMain and PSMain, which is how a shader pack
    written for D3D normally looks.

    Both entry points come out of the SAME file, and the image matches the
    one-function-per-file HLSL pair — so choosing the entry point changed which
    function was compiled, not merely that something compiled.
    """
    both = str(SHADER_DIR / "two_entry_points.hlsl")
    vs = ctx.compile_shader(both, bz.ShaderStage.VERTEX, entry_point="VSMain")
    ps = ctx.compile_shader(both, bz.ShaderStage.FRAGMENT, entry_point="PSMain")

    reference = render_fullscreen(
        ctx,
        ctx.compile_shader(str(SHADER_DIR / "fullscreen_vs.hlsl"), bz.ShaderStage.VERTEX),
        ctx.compile_shader(str(SHADER_DIR / "uv_gradient_ps.hlsl"), bz.ShaderStage.FRAGMENT))
    assert np.array_equal(render_fullscreen(ctx, vs, ps), reference)


def test_an_entry_point_that_matches_nothing_is_refused(ctx):
    """Was an accepted ceiling until 0.19; now a real diagnostic.

    glslang does not treat an unknown HLSL entry point as an error: it synthesizes
    an empty one under the requested name, so the compile succeeds and the shader
    draws nothing. This test used to pin that behaviour by SPIR-V size, because
    catching it needed reflection (debt #3). Reflection exists now, so the typo is
    named at the compile that caused it.

    Gated on HLSL with an explicit entry_point, so a GLSL main() that deliberately
    does nothing is never accused — which the next test checks.
    """
    both = str(SHADER_DIR / "two_entry_points.hlsl")
    # The real function still compiles.
    real = ctx.compile_shader(both, bz.ShaderStage.VERTEX, entry_point="VSMain")
    assert len(real.spirv) > 0

    with pytest.raises(bz.ShaderError, match="matches no function"):
        ctx.compile_shader(both, bz.ShaderStage.VERTEX, entry_point="NoSuchFunction")


def test_a_deliberately_empty_glsl_shader_is_not_accused(ctx):
    """The empty-entry-point check must not fire on GLSL. A vertex shader whose
    main() writes nothing is legal and occasionally useful, and it has no
    entry_point= to have misspelled."""
    empty = "#version 450\nvoid main() {}\n"
    module = ctx.compile_shader(source=empty, stage=bz.ShaderStage.VERTEX)
    assert len(module.spirv) > 0


def test_entry_point_on_glsl_says_so(ctx):
    """The one compile knob that exists for a single language, so the wrong
    language gets a sentence rather than a shaderc error to decode. Since 0.24
    the sentence names language= as the fix, because renaming the file is no
    longer the only one."""
    with pytest.raises(bz.ShaderError, match="entry_point"):
        ctx.compile_shader(source=FRAG_OK, stage=bz.ShaderStage.FRAGMENT,
                           entry_point="PSMain")


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

    shadow = ctx.create_render_target(64, 64, color=None, depth=bz.Format.D32F)
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
        frag = ctx.compile_shader(source=frag_source, stage=bz.ShaderStage.FRAGMENT, name=name)
        screen = ctx.create_render_target(64, 64)
        pipeline = (ctx.graphics_pipeline()
                    .vertex_shader(fullscreen)
                    .fragment_shader(frag)
                    .texture(0, bz.ShaderStage.FRAGMENT, set=0)
                    .build(screen))
        pool = ctx.create_descriptor_pool(max_sets=4, textures=4)
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
        return screen.color[0].read()

    manual = compare_pass(MANUAL_COMPARE_FRAG, "manual_cmp.frag",
                          ctx.create_sampler(filter=bz.Filter.NEAREST))
    hardware = compare_pass(SHADOW_COMPARE_FRAG, "shadow_cmp.frag",
                            ctx.create_sampler(filter=bz.Filter.NEAREST,
                                               compare=bz.CompareOp.LESS))

    assert not np.array_equal(manual, np.zeros_like(manual)), "scene should not be all-shadow"
    assert np.array_equal(manual, hardware)
