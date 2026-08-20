#pragma once
#include <volk.h>
#include <string>
#include <vector>
#include <fstream>
#include <charconv>
#include <memory>
#include <expected>
#include <optional>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <variant>

#include "Context.hpp"
#include "Error.hpp"
#include "SpirvReflect.hpp"

// Appended rather than inserted in pipeline order (which would put the two
// tessellation stages between VERTEX and FRAGMENT): pybind enum values are API,
// nothing iterates this enum in pipeline order, and renumbering FRAGMENT costs a
// break for a cosmetic gain. TESS_* rather than TESSELLATION_*, matching both
// shaderc's own spelling and the .tesc/.tese extensions every GLSL toolchain uses.
enum class ShaderStage
{
    VERTEX,
    FRAGMENT,
    COMPUTE,
    TESS_CONTROL,    // VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT
    TESS_EVALUATION, // VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT
    GEOMETRY
};

// Which parser shaderc runs over the text. A plain value rather than an
// optional, because by the time a compile starts the question is answered: the
// API boundary resolves "not stated" into one of these, from the file extension
// where there is a file and from the caller where there is not.
enum class ShaderLanguage
{
    GLSL,
    HLSL
};

// The content of a shader when it does not come from a file: GLSL or HLSL text,
// or ready SPIR-V words. One parameter with two types, because from the
// caller's side both answer the same question — "here is the content instead of
// the file" — and two mutually exclusive parameters in one signature is the
// shape 0.15 already rejected for the cross-Context transfer.
//
// With SPIR-V words nothing is compiled, so neither the language nor the
// include dirs nor the entry point have any work to do.
using ShaderSource = std::variant<std::string, std::vector<std::uint32_t>>;

// Extra directories to resolve #include against, tried in order AFTER the
// directory of the including file. A shader carries its own list, so a hot
// reload recompiles it the same way the first compile did.
using IncludeDirs = std::vector<std::string>;

// A real switch, deliberately with no default case: adding a new stage makes
// every conversion site a compiler error instead of silently aliasing the new
// stage onto FRAGMENT, which is what the old `stage == VERTEX ? ... : ...`
// ternaries scattered across Pipeline.hpp and CommandBuffer.hpp would have done.
inline constexpr VkShaderStageFlagBits to_vk(ShaderStage stage)
{
    switch (stage)
    {
        case ShaderStage::VERTEX:
            return VK_SHADER_STAGE_VERTEX_BIT;
        case ShaderStage::FRAGMENT:
            return VK_SHADER_STAGE_FRAGMENT_BIT;
        case ShaderStage::COMPUTE:
            return VK_SHADER_STAGE_COMPUTE_BIT;
        case ShaderStage::TESS_CONTROL:
            return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
        case ShaderStage::TESS_EVALUATION:
            return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
        case ShaderStage::GEOMETRY:
            return VK_SHADER_STAGE_GEOMETRY_BIT;
    }
    // Not std::unreachable(): pybind enums accept arbitrary ints, so a forged
    // ShaderStage from Python must degrade gracefully, not invoke UB.
    return VK_SHADER_STAGE_VERTEX_BIT;
}

class ShaderModule
{
public:
    ShaderModule(
        std::shared_ptr<Context> context,
        VkShaderModule module,
        const std::string& path,
        ShaderStage stage,
        std::vector<std::string> includes,
        std::vector<uint32_t> spirv,
        IncludeDirs include_dirs = {},
        std::string entry_point = {},
        ShaderReflection reflection = {},
        ShaderLanguage language = ShaderLanguage::GLSL);

    ~ShaderModule();

    ShaderModule(const ShaderModule&) = delete;
    ShaderModule& operator=(const ShaderModule&) = delete;

    ShaderModule(ShaderModule&& other) noexcept;

    ShaderModule& operator=(ShaderModule&& other) noexcept;

    VkShaderModule get() const
    {
        return module_;
    }
    const std::string& path() const
    {
        return path_;
    }
    // The stage the module was compiled for — not derivable from the file
    // extension, so it must be remembered for the hot-reload recompile.
    ShaderStage stage() const
    {
        return stage_;
    }
    // Files pulled in via #include, absolute and normalized. The 0.8 hot-reload
    // watcher watches path() plus these; empty for .spv and include-free sources.
    const std::vector<std::string>& includes() const
    {
        return includes_;
    }
    const std::vector<uint32_t>& spirv() const
    {
        return spirv_;
    }
    // The three compile settings a recompile cannot re-derive from the file.
    // Empty entry_point means the language default ("main"), and the language is
    // already resolved — a reload must not re-infer it, or a file compiled with
    // an explicit language= would come back parsed as the other one.
    const IncludeDirs& include_dirs() const
    {
        return include_dirs_;
    }
    const std::string& entry_point() const
    {
        return entry_point_;
    }
    ShaderLanguage language() const
    {
        return language_;
    }
    // What the SPIR-V says this shader does — see SpirvReflect.hpp. Read by the
    // barrier tracker through Pipeline::shaders() at record time.
    const ShaderReflection& reflection() const
    {
        return reflection_;
    }

    // Swap in a freshly compiled body (hot reload). MAIN THREAD ONLY: the
    // compile that produced these parts ran an unlocked RecordingIncluder. The
    // old handle retires through the deletion queue rather than being destroyed
    // inline — a pipeline the watcher is about to rebuild() still names it until
    // that rebuild picks up the new handle, and an in-flight frame may still
    // hold the old VkPipeline built from it. Pipelines pick the new module up on
    // their next rebuild(); the ShaderModule object identity never changes, so
    // the watcher's weak_ptr and every builder's shared_ptr stay valid.
    void replace(
        VkShaderModule module,
        std::vector<std::string> includes,
        std::vector<uint32_t> spirv,
        ShaderReflection reflection);

private:
    void destroy();

    std::shared_ptr<Context> context_;
    VkShaderModule module_;
    std::string path_;
    ShaderStage stage_;
    std::vector<std::string> includes_;
    std::vector<uint32_t> spirv_;
    // Carried so a hot reload recompiles the way the first compile did. A
    // watcher only has the module, so anything the compile depended on has to
    // live here or the reloaded shader is quietly a different shader.
    IncludeDirs include_dirs_;
    std::string entry_point_;
    ShaderReflection reflection_;
    ShaderLanguage language_ = ShaderLanguage::GLSL;
};

class ShaderCompiler
{
public:
    // The compiled body of a shader without the ShaderModule wrapper: a fresh
    // VkShaderModule plus the include list and SPIR-V that go with it. compile()
    // wraps these into a new ShaderModule; the hot-reload watcher feeds them to
    // ShaderModule::replace() so the module object identity survives a reload.
    struct CompiledParts
    {
        VkShaderModule module;
        std::vector<std::string> includes;
        std::vector<uint32_t> spirv;
        // Derived from `spirv`, and travelling WITH it on purpose. Every path that
        // produces words fills this in the same call, so the two can never
        // disagree — which matters most for hot reload: computing reflection in
        // compile() instead would leave every reloaded module carrying its
        // original's answers, silently.
        ShaderReflection reflection;
    };

    // The rule, in one place: only `.hlsl` is HLSL. Everything else is GLSL,
    // which covers .vert/.frag/.comp and every other convention a project
    // invents. The API boundary calls this when the caller named no language,
    // and never after — see ShaderModule::language().
    static ShaderLanguage infer_language(const std::string& path)
    {
        return lowercase_extension(path) == ".hlsl" ? ShaderLanguage::HLSL : ShaderLanguage::GLSL;
    }

    // Whether this path names an already-compiled binary rather than something
    // to parse. A format, not a language, which is why it is a separate question
    // from infer_language and why ShaderLanguage has no SPIRV member.
    static bool is_spirv_path(const std::string& path)
    {
        return lowercase_extension(path) == ".spv";
    }

    // One entry point for every shader form. `path` names a file when `source`
    // is absent and is a diagnostic label when it is present — either way it
    // tags ShaderError.path and anchors relative #include resolution. The
    // language arrives resolved, so nothing below re-reads the extension.
    static std::expected<std::shared_ptr<ShaderModule>, Error> compile(
        Context& context,
        const std::string& path,
        ShaderStage stage,
        std::optional<ShaderSource> source = std::nullopt,
        ShaderLanguage language = ShaderLanguage::GLSL,
        IncludeDirs include_dirs = {},
        const std::string& entry_point = {});

    static constexpr const char* stage_name(ShaderStage stage)
    {
        switch (stage)
        {
            case ShaderStage::VERTEX:
                return "VERTEX";
            case ShaderStage::FRAGMENT:
                return "FRAGMENT";
            case ShaderStage::COMPUTE:
                return "COMPUTE";
            case ShaderStage::TESS_CONTROL:
                return "TESS_CONTROL";
            case ShaderStage::TESS_EVALUATION:
                return "TESS_EVALUATION";
            case ShaderStage::GEOMETRY:
                return "GEOMETRY";
        }
        return "unknown";
    }

    // Which optional device feature a stage needs, or nullopt for the three that
    // every conformant device has. A table rather than an if-chain at each call
    // site, so the compile gate and the pipeline gate cannot disagree about the
    // answer or about the wording.
    static constexpr std::optional<Feature> feature_for_stage(ShaderStage stage)
    {
        switch (stage)
        {
            case ShaderStage::TESS_CONTROL:
            case ShaderStage::TESS_EVALUATION:
                return Feature::TESSELLATION;
            case ShaderStage::GEOMETRY:
                return Feature::GEOMETRY_SHADER;
            case ShaderStage::VERTEX:
            case ShaderStage::FRAGMENT:
            case ShaderStage::COMPUTE:
                return std::nullopt;
        }
        // Not std::unreachable(): pybind enums accept arbitrary ints, and a forged
        // stage needing no feature is the harmless answer — it fails later, at the
        // switch that has to name a real VkShaderStageFlagBits.
        return std::nullopt;
    }

    // Public because the pipeline builder calls it too: see the comment on the
    // second gate in GraphicsPipelineBuilder::build.
    static std::expected<void, Error> check_stage_supported(const Context& context, ShaderStage stage);

    // The compile without the wrapper. Reads the file fresh from `path` (unless
    // `source` overrides it), so a watcher recompiles simply by calling this
    // again with the module's stored path and stage. MAIN THREAD ONLY when it
    // compiles text (RecordingIncluder is unlocked); .spv loading is thread-safe
    // but shares the entry point for one obvious way.
    static std::expected<CompiledParts, Error> compile_parts(
        Context& context,
        const std::string& path,
        ShaderStage stage,
        std::optional<ShaderSource> source = std::nullopt,
        ShaderLanguage language = ShaderLanguage::GLSL,
        const IncludeDirs& include_dirs = {},
        const std::string& entry_point = {});

private:
    static std::string lowercase_extension(const std::string& path);

    static std::expected<CompiledParts, Error> compile_text(
        Context& context,
        const std::string& path,
        ShaderStage stage,
        std::optional<std::string> source,
        ShaderLanguage language,
        const IncludeDirs& include_dirs,
        const std::string& entry_point);

    static std::expected<CompiledParts, Error> load_spv(Context& context, const std::string& path, ShaderStage stage);

    // Validate ready SPIR-V words and wrap them. Shared by the .spv file path
    // and by source= bytes, so the two cannot check different things — bytes
    // from memory deserve exactly the diagnostics a file gets.
    static std::expected<CompiledParts, Error> parts_from_spirv(
        Context& context,
        std::vector<std::uint32_t> spirv,
        ShaderStage stage,
        const std::string& tag);

    // ShaderStage to the SPIR-V ExecutionModel it compiles to. The fifth switch on
    // the enum, and like the other four it has no default: a new stage must break
    // the build here rather than silently claim to be a vertex shader.
    static constexpr std::uint32_t execution_model_of(ShaderStage stage)
    {
        switch (stage)
        {
            case ShaderStage::VERTEX:
                return 0;
            case ShaderStage::TESS_CONTROL:
                return 1;
            case ShaderStage::TESS_EVALUATION:
                return 2;
            case ShaderStage::GEOMETRY:
                return 3;
            case ShaderStage::FRAGMENT:
                return 4;
            case ShaderStage::COMPUTE:
                return 5;
        }
        // A sentinel matching no execution model, so a forged pybind int degrades to
        // "declares no entry point" instead of matching a real stage.
        return 0xFFFFFFFFu;
    }

    static std::expected<VkShaderModule, Error> make_vk_module(Context& context, const std::vector<uint32_t>& spirv);

    struct ErrorLocation
    {
        std::string path; // empty when the log didn't match — caller falls back
        int line = -1;    // -1 when unknown; honest beats a wrong guess
    };

    // shaderc formats diagnostics as "<name>:<line>: error: ...", so the first
    // ":<digits>:" is the line, and the name is everything from the start of
    // that LOG LINE (after the previous '\n', not the start of the whole log —
    // earlier diagnostics would otherwise be swallowed into the name). With the
    // includer active the name may be an INCLUDED file: exactly what
    // ShaderError.path should say, so the user — and the 0.8 watcher — opens
    // the file the error is actually in. Windows drive colons ("C:/x.frag:12:")
    // are safe: a ':' followed by a non-digit just keeps the scan moving.
    static ErrorLocation parse_error_location(const std::string& log);
};
