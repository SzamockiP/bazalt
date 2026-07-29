#pragma once
#include <volk.h>
#include <shaderc/shaderc.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
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

// The content of a shader when it does not come from its path: GLSL or HLSL
// text, or ready SPIR-V words. One parameter with two types, because from the
// caller's side both answer the same question — "here is the content instead of
// the file" — and two mutually exclusive parameters in one signature is the
// shape 0.15 already rejected for the cross-Context transfer.
//
// `path` keeps its other jobs whichever arrives: the language, the diagnostic
// tag, ShaderError.path and the base directory for #include. With SPIR-V words
// the extension stops mattering, because nothing is compiled.
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

inline constexpr shaderc_shader_kind to_shaderc_kind(ShaderStage stage)
{
    switch (stage)
    {
        case ShaderStage::VERTEX:
            return shaderc_glsl_vertex_shader;
        case ShaderStage::FRAGMENT:
            return shaderc_glsl_fragment_shader;
        case ShaderStage::COMPUTE:
            return shaderc_glsl_compute_shader;
        case ShaderStage::TESS_CONTROL:
            return shaderc_glsl_tess_control_shader;
        case ShaderStage::TESS_EVALUATION:
            return shaderc_glsl_tess_evaluation_shader;
        case ShaderStage::GEOMETRY:
            return shaderc_glsl_geometry_shader;
    }
    return shaderc_glsl_vertex_shader;
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
        ShaderReflection reflection = {})
        : context_(context),
          module_(module),
          path_(path),
          stage_(stage),
          includes_(std::move(includes)),
          spirv_(std::move(spirv)),
          include_dirs_(std::move(include_dirs)),
          entry_point_(std::move(entry_point)),
          reflection_(std::move(reflection))
    {
    }

    ~ShaderModule()
    {
        destroy();
    }

    ShaderModule(const ShaderModule&) = delete;
    ShaderModule& operator=(const ShaderModule&) = delete;

    ShaderModule(ShaderModule&& other) noexcept
        : context_(std::move(other.context_)),
          module_(other.module_),
          path_(std::move(other.path_)),
          stage_(other.stage_),
          includes_(std::move(other.includes_)),
          spirv_(std::move(other.spirv_)),
          include_dirs_(std::move(other.include_dirs_)),
          entry_point_(std::move(other.entry_point_)),
          reflection_(std::move(other.reflection_))
    {
        other.module_ = VK_NULL_HANDLE;
    }

    ShaderModule& operator=(ShaderModule&& other) noexcept
    {
        if (this != &other)
        {
            destroy();
            context_ = std::move(other.context_);
            module_ = other.module_;
            path_ = std::move(other.path_);
            stage_ = other.stage_;
            includes_ = std::move(other.includes_);
            spirv_ = std::move(other.spirv_);
            include_dirs_ = std::move(other.include_dirs_);
            entry_point_ = std::move(other.entry_point_);
            reflection_ = std::move(other.reflection_);
            other.module_ = VK_NULL_HANDLE;
        }
        return *this;
    }

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
    // The two compile settings a recompile cannot re-derive from the file. Empty
    // entry_point means the language default ("main").
    const IncludeDirs& include_dirs() const
    {
        return include_dirs_;
    }
    const std::string& entry_point() const
    {
        return entry_point_;
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
        ShaderReflection reflection)
    {
        if (module_ != VK_NULL_HANDLE && context_)
        {
            context_->defer_destroy([vk = &context_->vk(), device = context_->device(), old = module_]
                                    { vk->vkDestroyShaderModule(device, old, nullptr); });
        }
        module_ = module;
        includes_ = std::move(includes);
        spirv_ = std::move(spirv);
        // Swapped with the words it describes. A reload that kept the old
        // reflection would compute barriers for the shader that used to be there.
        reflection_ = std::move(reflection);
    }

private:
    void destroy()
    {
        if (module_ != VK_NULL_HANDLE && context_)
        {
            context_->vk().vkDestroyShaderModule(context_->device(), module_, nullptr);
        }
    }

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
};

// Resolves #include relative to the directory of the INCLUDING file (both "..."
// and <...> forms — one rule, applied recursively: an include inside an include
// resolves against the inner file's directory) and records every file it hands
// out, absolute and normalized, so the 0.8 hot-reload watcher can watch them.
// One instance serves exactly one compile() call on the caller's thread, hence
// no locking — the 0.8 watcher must keep recompilation on the main thread.
class RecordingIncluder final : public shaderc::CompileOptions::IncluderInterface
{
public:
    explicit RecordingIncluder(IncludeDirs search_dirs = {})
        : search_dirs_(std::move(search_dirs))
    {
    }

    shaderc_include_result* GetInclude(
        const char* requested_source,
        shaderc_include_type /*type*/,
        const char* requesting_source,
        size_t /*include_depth*/) override
    {
        namespace fs = std::filesystem;
        fs::path resolved = resolve_(fs::path(requesting_source).parent_path(), requested_source);

        // Each result gets its own heap Holder: shaderc may hold several results
        // at once, and every one must stay valid until its ReleaseInclude.
        auto* holder = new Holder{};

        std::ifstream file(resolved, std::ios::ate | std::ios::binary);
        if (!file.is_open())
        {
            // shaderc convention: empty source_name marks failure, content
            // carries the error message.
            holder->content = "Cannot open include file: " + resolved.generic_string();
            holder->result = {"", 0, holder->content.c_str(), holder->content.size(), holder};
            return &holder->result;
        }

        size_t size = static_cast<size_t>(file.tellg());
        holder->content.resize(size);
        file.seekg(0);
        file.read(holder->content.data(), size);
        holder->name = resolved.generic_string();
        holder->result = {
            holder->name.c_str(), holder->name.size(), holder->content.c_str(), holder->content.size(), holder};

        if (!std::ranges::contains(included_, holder->name))
        {
            included_.push_back(holder->name);
        }
        return &holder->result;
    }

    void ReleaseInclude(shaderc_include_result* result) override
    {
        delete static_cast<Holder*>(result->user_data);
    }

    const std::vector<std::string>& included() const
    {
        return included_;
    }

private:
    // The including file's directory first, so every shader that already
    // compiled keeps resolving exactly as it did. The search dirs are a
    // FALLBACK, tried in order, and only for a name that is not there: a
    // search path that could shadow a neighbouring file would change the
    // meaning of existing shaders the moment a directory was added.
    //
    // The last candidate is returned even when nothing exists, so the "cannot
    // open" message names the primary location rather than the last directory
    // tried.
    std::filesystem::path resolve_(const std::filesystem::path& including_dir, const char* requested) const
    {
        namespace fs = std::filesystem;
        const fs::path primary = normalize_(including_dir / requested);
        std::error_code ec;
        if (fs::exists(primary, ec))
        {
            return primary;
        }
        for (const std::string& dir : search_dirs_)
        {
            const fs::path candidate = normalize_(fs::path(dir) / requested);
            if (fs::exists(candidate, ec))
            {
                return candidate;
            }
        }
        return primary;
    }

    static std::filesystem::path normalize_(const std::filesystem::path& raw)
    {
        std::error_code ec;
        std::filesystem::path resolved = std::filesystem::weakly_canonical(raw, ec);
        return ec ? raw : resolved;
    }

    struct Holder
    {
        std::string name;
        std::string content;
        shaderc_include_result result{};
    };

    IncludeDirs search_dirs_;
    std::vector<std::string> included_;
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

    // One entry point for every shader form. The extension of `path` decides how
    // it is handled (GLSL by default); when `source` is given the file is never
    // opened and `path` is a virtual name — it still supplies the language, the
    // diagnostic tag, ShaderError.path, and the base directory for #include.
    static std::expected<std::shared_ptr<ShaderModule>, Error> compile(
        Context& context,
        const std::string& path,
        ShaderStage stage,
        std::optional<ShaderSource> source = std::nullopt,
        IncludeDirs include_dirs = {},
        const std::string& entry_point = {})
    {
        auto parts = compile_parts(context, path, stage, std::move(source), include_dirs, entry_point);
        if (!parts)
        {
            return std::unexpected(parts.error());
        }
        return std::make_shared<ShaderModule>(
            context.shared_from_this(),
            parts->module,
            path,
            stage,
            std::move(parts->includes),
            std::move(parts->spirv),
            std::move(include_dirs),
            entry_point,
            std::move(parts->reflection));
    }

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
    static std::expected<void, Error> check_stage_supported(const Context& context, ShaderStage stage)
    {
        const auto feature = feature_for_stage(stage);
        if (!feature || context.supports(*feature))
        {
            return {};
        }
        return std::unexpected(err_shader(
            std::format(
                "a {} shader requires the {} feature. Create the Context with "
                "features=[bz.Feature.{}] (or optional=[...])",
                stage_name(stage),
                feature_name(*feature),
                feature_name(*feature))));
    }

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
        const IncludeDirs& include_dirs = {},
        const std::string& entry_point = {})
    {
        // Before anything is compiled or loaded: a stage whose feature is off
        // cannot even have its module CREATED. Every path below ends at
        // vkCreateShaderModule, and SPIR-V for these stages declares
        // OpCapability Tessellation / Geometry, which the layers reject when the
        // matching feature is not enabled
        // (VUID-VkShaderModuleCreateInfo-pCode-08740).
        //
        // So the gate belongs here, at the line the user actually wrote. The
        // pipeline builder checks the same thing, and that check is NOT redundant
        // — it catches a module compiled on a Context that has the feature and
        // then handed to a pipeline on one that does not — but by then the invalid
        // module already exists, which is too late to be the primary diagnostic.
        if (auto e = check_stage_supported(context, stage); !e)
        {
            return std::unexpected(e.error());
        }

        // Ready SPIR-V short-circuits everything: nothing is compiled, so the
        // extension, the include dirs and the entry point have no work to do.
        if (source && std::holds_alternative<std::vector<std::uint32_t>>(*source))
        {
            return parts_from_spirv(context, std::get<std::vector<std::uint32_t>>(std::move(*source)), stage, path);
        }

        if (lowercase_extension(path) == ".spv")
        {
            if (source)
            {
                return std::unexpected(err_shader(
                    "source= gave text for compilation, and .spv is a binary format. Pass a file path, or pass "
                    "the SPIR-V itself as bytes.",
                    path));
            }
            return load_spv(context, path, stage);
        }

        std::optional<std::string> text;
        if (source)
        {
            text = std::get<std::string>(std::move(*source));
        }
        return compile_text(context, path, stage, std::move(text), include_dirs, entry_point);
    }

private:
    static std::string lowercase_extension(const std::string& path)
    {
        std::string ext = std::filesystem::path(path).extension().string();
        for (char& c : ext)
        {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        return ext;
    }

    static std::expected<CompiledParts, Error> compile_text(
        Context& context,
        const std::string& path,
        ShaderStage stage,
        std::optional<std::string> source,
        const IncludeDirs& include_dirs,
        const std::string& entry_point)
    {
        std::string text;
        if (source)
        {
            text = std::move(*source);
        }
        else
        {
            std::ifstream file(path, std::ios::ate | std::ios::binary);
            if (!file.is_open())
            {
                return std::unexpected(err_resource("Failed to open shader file: " + path));
            }

            size_t fileSize = static_cast<size_t>(file.tellg());
            text.resize(fileSize);
            file.seekg(0);
            file.read(text.data(), fileSize);
        }

        shaderc::Compiler compiler;
        shaderc::CompileOptions options;
        // Follow the negotiated device version rather than assuming 1.3: SPIR-V
        // targeted at 1.3 can be rejected by a 1.2 driver.
        shaderc_env_version env_version = VK_API_VERSION_MINOR(context.api_version()) >= 3
                                              ? shaderc_env_version_vulkan_1_3
                                              : shaderc_env_version_vulkan_1_2;
        options.SetTargetEnvironment(shaderc_target_env_vulkan, env_version);
        // debugPrintfEXT() compiles to OpExtInst against a NON-SEMANTIC instruction
        // set, and non-semantic means exactly "an optimizer may delete this": at
        // -O the print has no observable result, so dead-code elimination removes
        // it and the shader silently stops printing. So printf costs the
        // optimizer, for every shader in the Context, and that is one of the two
        // reasons shader_printf is opt-in rather than always on.
        // Since 0.19 a printf Context compiles unoptimized only where it has to.
        // The order matters and is easy to get backwards: `prints` can only be read
        // off UNOPTIMIZED words, because -O deletes the print that would prove it.
        // So compile at zero first, reflect, and recompile at performance when the
        // shader turns out not to print. A printing shader is one compile; a
        // non-printing one in a printf Context is two, in a debugging mode where
        // every shader used to pay the optimizer.
        options.SetOptimizationLevel(
            context.shader_printf() ? shaderc_optimization_level_zero : shaderc_optimization_level_performance);

        // Language is an attribute of the file name, not a second API path.
        const bool hlsl = lowercase_extension(path) == ".hlsl";
        if (hlsl)
        {
            options.SetSourceLanguage(shaderc_source_language_hlsl);
        }

        // entry_point= exists for HLSL, where one file legitimately holds VSMain
        // and PSMain. A GLSL entry point must be main, so a name here is a
        // mistake worth naming rather than a shaderc error to decode.
        if (!entry_point.empty() && !hlsl)
        {
            return std::unexpected(err_shader(
                "entry_point= applies to HLSL. A GLSL entry point must be main, so remove the argument or "
                "rename the file to .hlsl.",
                path));
        }

        // Keep a raw pointer before the unique_ptr moves into options; the
        // recorded includes are read back only while `options` is alive.
        auto includer = std::make_unique<RecordingIncluder>(include_dirs);
        RecordingIncluder* recorder = includer.get();
        options.SetIncluder(std::move(includer));

        shaderc_shader_kind kind = to_shaderc_kind(stage);
        const std::string entry = entry_point.empty() ? "main" : entry_point;

        shaderc::SpvCompilationResult module =
            compiler.CompileGlslToSpv(text, kind, path.c_str(), entry.c_str(), options);

        if (module.GetCompilationStatus() != shaderc_compilation_status_success)
        {
            std::string log = module.GetErrorMessage();
            auto loc = parse_error_location(log);
            return std::unexpected(
                err_shader("Shader compilation failed: " + log, loc.path.empty() ? path : loc.path, loc.line));
        }

        std::vector<uint32_t> spirv(module.cbegin(), module.cend());
        ShaderReflection reflection = reflect_spirv(spirv, hlsl ? entry_point : std::string{});

        // The second half of the printf decision described above. Recompiling
        // discards the unoptimized words entirely, so make_vk_module runs once,
        // after this — a module built from the wrong words would be the whole cost
        // of the optimization plus a wasted VkShaderModule.
        if (context.shader_printf() && !reflection.prints)
        {
            options.SetOptimizationLevel(shaderc_optimization_level_performance);
            shaderc::SpvCompilationResult optimized =
                compiler.CompileGlslToSpv(text, kind, path.c_str(), entry.c_str(), options);
            // A failure here would be surprising — the same source just compiled —
            // so fall back to the words already in hand rather than failing a
            // compile that succeeded.
            if (optimized.GetCompilationStatus() == shaderc_compilation_status_success)
            {
                spirv.assign(optimized.cbegin(), optimized.cend());
                reflection = reflect_spirv(spirv, hlsl ? entry_point : std::string{});
            }
        }

        // An HLSL entry point matching no function is not an error to glslang: it
        // synthesizes one under the requested name, so the compile succeeds and the
        // shader draws nothing. That was an accepted ceiling until there was a way
        // to see it. Gated on HLSL with an explicit name, because a GLSL main() that
        // deliberately does nothing is legitimate.
        if (hlsl && !entry_point.empty() && reflection.empty_entry_point)
        {
            return std::unexpected(err_shader(
                std::format(
                    "entry_point=\"{}\" matches no function in {}. glslang does not treat that as an "
                    "error — it synthesizes an empty entry point under that name, so the shader would "
                    "compile and then draw nothing. Check the spelling against the HLSL source.",
                    entry_point,
                    path),
                path));
        }

        auto vk_module = make_vk_module(context, spirv);
        if (!vk_module)
        {
            return std::unexpected(vk_module.error());
        }
        return CompiledParts{*vk_module, recorder->included(), std::move(spirv), std::move(reflection)};
    }

    static std::expected<CompiledParts, Error> load_spv(Context& context, const std::string& path, ShaderStage stage)
    {
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (!file.is_open())
        {
            return std::unexpected(err_resource("Failed to open shader file: " + path));
        }

        size_t fileSize = static_cast<size_t>(file.tellg());
        if (fileSize % sizeof(uint32_t) != 0)
        {
            return std::unexpected(err_shader(
                path + " is not SPIR-V. SPIR-V is a stream of 32-bit words, so its length "
                       "is always a multiple of 4, and this file's is not.",
                path));
        }

        std::vector<uint32_t> spirv(fileSize / sizeof(uint32_t));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(spirv.data()), fileSize);

        return parts_from_spirv(context, std::move(spirv), stage, path);
    }

    // Validate ready SPIR-V words and wrap them. Shared by the .spv file path
    // and by source= bytes, so the two cannot check different things — bytes
    // from memory deserve exactly the diagnostics a file gets.
    static std::expected<CompiledParts, Error> parts_from_spirv(
        Context& context,
        std::vector<std::uint32_t> spirv,
        ShaderStage stage,
        const std::string& tag)
    {
        constexpr uint32_t spirv_magic = 0x07230203u;
        if (spirv.empty() || spirv[0] != spirv_magic)
        {
            // Same user error as the size check above, so it gets the same quality of
            // explanation. "bad magic number" named a binary-format concept and left
            // the caller to guess which of their files was wrong.
            return std::unexpected(err_shader(
                tag + " is not SPIR-V. Every SPIR-V module starts with the same 4-byte "
                      "marker, and this one does not, so it is a source file or another "
                      "kind of binary. Compile it first, or pass the path to the .spv.",
                tag));
        }

        // One walk answers the stage question and everything else. This replaces a
        // second, near-identical loop that existed only to find OpEntryPoint.
        ShaderReflection reflection = reflect_spirv(spirv);
        if (!reflection.declares_execution_model(execution_model_of(stage)))
        {
            return std::unexpected(err_shader(
                std::format(
                    "{} declares no {} entry point — the binary was built for a different stage",
                    tag,
                    stage_name(stage)),
                tag));
        }

        // Foreign SPIR-V, so the write scan is not trusted to be complete. bazalt
        // knows the write opcodes its own GLSL/HLSL compiles down to; it cannot know
        // what another toolchain emitted, and a module using an opcode the parser
        // does not list would be reported as writing nothing. Provenance is the one
        // thing the parser cannot see and this function knows for certain, so the
        // flag is set here rather than guessed there.
        reflection.writes_unknown = true;

        auto vk_module = make_vk_module(context, spirv);
        if (!vk_module)
        {
            return std::unexpected(vk_module.error());
        }
        return CompiledParts{*vk_module, {}, std::move(spirv), std::move(reflection)};
    }

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

    static std::expected<VkShaderModule, Error> make_vk_module(Context& context, const std::vector<uint32_t>& spirv)
    {
        VkShaderModuleCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .codeSize = spirv.size() * sizeof(uint32_t),
            .pCode = spirv.data()};

        VkShaderModule vk_module;
        if (auto e = check(
                context.vk().vkCreateShaderModule(context.device(), &createInfo, nullptr, &vk_module),
                "create shader module",
                ErrorCode::Shader))
        {
            return std::unexpected(*e);
        }

        return vk_module;
    }

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
    static ErrorLocation parse_error_location(const std::string& log)
    {
        for (std::size_t i = 0; i + 1 < log.size(); ++i)
        {
            if (log[i] != ':')
            {
                continue;
            }

            std::size_t j = i + 1;
            while (j < log.size() && std::isdigit(static_cast<unsigned char>(log[j])))
            {
                ++j;
            }

            if (j > i + 1 && j < log.size() && log[j] == ':')
            {
                ErrorLocation loc;
                try
                {
                    loc.line = std::stoi(log.substr(i + 1, j - i - 1));
                }
                catch (const std::exception&)
                {
                    return {};
                }
                std::size_t start = log.rfind('\n', i);
                start = (start == std::string::npos) ? 0 : start + 1;
                loc.path = log.substr(start, i - start);
                return loc;
            }
        }
        return {};
    }
};
