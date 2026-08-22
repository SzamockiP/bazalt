#include "ShaderCompiler.hpp"

// The heavy half of this subject. shaderc (and the glslang/SPIRV-Tools headers
// behind it) is parsed by this translation unit alone since 0.27 — the eight
// binding TUs include ShaderCompiler.hpp and none of them names a shaderc type.
#include <shaderc/shaderc.hpp>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace
{

    constexpr shaderc_shader_kind to_shaderc_kind(ShaderStage stage)
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
            //
            // Owned here and released to shaderc on the way out, which hands it
            // back to ReleaseInclude below. A bare `new` leaked the Holder if
            // anything between the two threw — resizing the content for a large
            // include is the realistic one.
            auto holder = std::make_unique<Holder>();

            std::ifstream file(resolved, std::ios::ate | std::ios::binary);
            if (!file.is_open())
            {
                // shaderc convention: empty source_name marks failure, content
                // carries the error message.
                holder->content = "Cannot open include file: " + resolved.generic_string();
                holder->result = {
                    .source_name = "",
                    .source_name_length = 0,
                    .content = holder->content.c_str(),
                    .content_length = holder->content.size(),
                    .user_data = holder.get()};
                return &holder.release()->result;
            }

            const auto size = static_cast<std::size_t>(file.tellg());
            holder->content.resize(size);
            file.seekg(0);
            file.read(holder->content.data(), static_cast<std::streamsize>(size));
            holder->name = resolved.generic_string();
            holder->result = {
                .source_name = holder->name.c_str(),
                .source_name_length = holder->name.size(),
                .content = holder->content.c_str(),
                .content_length = holder->content.size(),
                .user_data = holder.get()};

            if (!std::ranges::contains(included_, holder->name))
            {
                included_.push_back(holder->name);
            }
            return &holder.release()->result;
        }

        // The other half of the release above: shaderc hands the Holder back, and
        // taking ownership into a unique_ptr frees it at the end of this scope.
        void ReleaseInclude(shaderc_include_result* result) override
        {
            const std::unique_ptr<Holder> owned(static_cast<Holder*>(result->user_data));
        }

        [[nodiscard]] const std::vector<std::string>& included() const
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
            // Not const: it is returned by value on two paths, and const would
            // turn both into copies of a path that is about to die anyway.
            fs::path primary = normalize_(including_dir / requested);
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

} // namespace

ShaderModule::ShaderModule(
    std::shared_ptr<Context> context,
    VkShaderModule module,
    std::string path,
    ShaderStage stage,
    std::vector<std::string> includes,
    std::vector<uint32_t> spirv,
    IncludeDirs include_dirs,
    std::string entry_point,
    ShaderReflection reflection,
    ShaderLanguage language)
    : context_(std::move(context)),
      module_(module),
      path_(std::move(path)),
      stage_(stage),
      includes_(std::move(includes)),
      spirv_(std::move(spirv)),
      include_dirs_(std::move(include_dirs)),
      entry_point_(std::move(entry_point)),
      reflection_(std::move(reflection)),
      language_(language)
{
}

ShaderModule::~ShaderModule()
{
    destroy();
}

ShaderModule::ShaderModule(ShaderModule&& other) noexcept
    : context_(std::move(other.context_)),
      module_(other.module_),
      path_(std::move(other.path_)),
      stage_(other.stage_),
      includes_(std::move(other.includes_)),
      spirv_(std::move(other.spirv_)),
      include_dirs_(std::move(other.include_dirs_)),
      entry_point_(std::move(other.entry_point_)),
      reflection_(std::move(other.reflection_)),
      language_(other.language_)
{
    other.module_ = VK_NULL_HANDLE;
}

ShaderModule& ShaderModule::operator=(ShaderModule&& other) noexcept
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
        language_ = other.language_;
        other.module_ = VK_NULL_HANDLE;
    }
    return *this;
}

void ShaderModule::replace(
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

void ShaderModule::destroy()
{
    if (module_ != VK_NULL_HANDLE && context_)
    {
        context_->vk().vkDestroyShaderModule(context_->device(), module_, nullptr);
    }
}

std::expected<std::shared_ptr<ShaderModule>, Error> ShaderCompiler::compile(
    Context& context,
    const std::string& path,
    ShaderStage stage,
    std::optional<ShaderSource> source,
    ShaderLanguage language,
    IncludeDirs include_dirs,
    const std::string& entry_point)
{
    auto parts = compile_parts(context, path, stage, std::move(source), language, include_dirs, entry_point);
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
        std::move(parts->reflection),
        language);
}

std::expected<void, Error> ShaderCompiler::check_stage_supported(const Context& context, ShaderStage stage)
{
    const auto feature = feature_for_stage(stage);
    if (!feature || context.supports(*feature))
    {
        return {};
    }
    // A missing stage feature is a fact about this Context, not about the file,
    // so it is Unsupported rather than Shader — the same failure surfaces at
    // build() through the same function, and the two must agree on the type.
    return std::unexpected(err_unsupported(
        std::format(
            "a {} shader requires the {} feature. Create the Context with "
            "features=[bz.Feature.{}] (or optional=[...])",
            stage_name(stage),
            feature_name(*feature),
            feature_name(*feature))));
}

std::expected<ShaderCompiler::CompiledParts, Error> ShaderCompiler::compile_parts(
    Context& context,
    const std::string& path,
    ShaderStage stage,
    std::optional<ShaderSource> source,
    ShaderLanguage language,
    const IncludeDirs& include_dirs,
    const std::string& entry_point)
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

    // Only a real file can be a prebuilt binary. With `source` the path is a
    // label, and a label ending in .spv must not change what happens — that
    // is the whole reason the two forms were split apart.
    if (!source && is_spirv_path(path))
    {
        return load_spv(context, path, stage);
    }

    std::optional<std::string> text;
    if (source)
    {
        text = std::get<std::string>(std::move(*source));
    }
    return compile_text(context, path, stage, std::move(text), language, include_dirs, entry_point);
}

std::string ShaderCompiler::lowercase_extension(const std::string& path)
{
    std::string ext = std::filesystem::path(path).extension().string();
    for (char& c : ext)
    {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return ext;
}

std::expected<ShaderCompiler::CompiledParts, Error> ShaderCompiler::compile_text(
    Context& context,
    const std::string& path,
    ShaderStage stage,
    std::optional<std::string> source,
    ShaderLanguage language,
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

        const auto fileSize = static_cast<std::size_t>(file.tellg());
        text.resize(fileSize);
        file.seekg(0);
        file.read(text.data(), static_cast<std::streamsize>(fileSize));
    }

    shaderc::Compiler compiler;
    shaderc::CompileOptions options;
    // Follow the negotiated device version rather than assuming 1.3: SPIR-V
    // targeted at 1.3 can be rejected by a 1.2 driver.
    shaderc_env_version env_version = VK_API_VERSION_MINOR(context.api_version()) >= 3 ? shaderc_env_version_vulkan_1_3
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

    // The language was decided at the API boundary — either the caller named
    // it or infer_language read the extension. Everything below keys off this
    // one value, so an explicit language= reaches the entry-point gate and
    // the reflection call as well, not only shaderc.
    const bool hlsl = language == ShaderLanguage::HLSL;
    if (hlsl)
    {
        options.SetSourceLanguage(shaderc_source_language_hlsl);
        // HLSL separates the texture from the sampler (Texture2D + SamplerState),
        // which glslang emits as SAMPLED_IMAGE plus SAMPLER — two descriptor types
        // no bazalt declarator can express, so idiomatic HLSL built a pipeline that
        // no descriptor set could feed. This folds the pair into one
        // COMBINED_IMAGE_SAMPLER at the texture's binding, which is exactly what
        // .texture() declares. The shaderc method is named for what it does to the
        // textures; the C entry point behind it is
        // shaderc_compile_options_set_auto_combined_image_sampler.
        options.SetAutoSampledTextures(true);
    }

    // entry_point= exists for HLSL, where one file legitimately holds VSMain
    // and PSMain. A GLSL entry point must be main, so a name here is a
    // mistake worth naming rather than a shaderc error to decode.
    if (!entry_point.empty() && !hlsl)
    {
        return std::unexpected(err_shader(
            "entry_point= applies to HLSL. A GLSL entry point must be main, so remove the argument or "
            "pass language=bz.ShaderLanguage.HLSL.",
            path));
    }

    // Keep a raw pointer before the unique_ptr moves into options; the
    // recorded includes are read back only while `options` is alive.
    auto includer = std::make_unique<RecordingIncluder>(include_dirs);
    RecordingIncluder* recorder = includer.get();
    options.SetIncluder(std::move(includer));

    shaderc_shader_kind kind = to_shaderc_kind(stage);
    const std::string entry = entry_point.empty() ? "main" : entry_point;

    shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(text, kind, path.c_str(), entry.c_str(), options);

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
    return CompiledParts{
        .module = *vk_module,
        .includes = recorder->included(),
        .spirv = std::move(spirv),
        .reflection = std::move(reflection)};
}

std::expected<ShaderCompiler::CompiledParts, Error> ShaderCompiler::load_spv(
    Context& context,
    const std::string& path,
    ShaderStage stage)
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
    file.read(reinterpret_cast<char*>(spirv.data()), static_cast<std::streamsize>(fileSize));

    return parts_from_spirv(context, std::move(spirv), stage, path);
}

std::expected<ShaderCompiler::CompiledParts, Error> ShaderCompiler::parts_from_spirv(
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
                "{} declares no {} entry point — the binary was built for a different stage", tag, stage_name(stage)),
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
    return CompiledParts{
        .module = *vk_module, .includes = {}, .spirv = std::move(spirv), .reflection = std::move(reflection)};
}

std::expected<VkShaderModule, Error> ShaderCompiler::make_vk_module(
    Context& context,
    const std::vector<uint32_t>& spirv)
{
    VkShaderModuleCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = spirv.size() * sizeof(uint32_t),
        .pCode = spirv.data()};

    VkShaderModule vk_module = nullptr;
    if (auto e = check(
            context.vk().vkCreateShaderModule(context.device(), &createInfo, nullptr, &vk_module),
            "create shader module",
            ErrorCode::Shader))
    {
        return std::unexpected(*e);
    }

    return vk_module;
}

ShaderCompiler::ErrorLocation ShaderCompiler::parse_error_location(const std::string& log)
{
    for (std::size_t i = 0; i + 1 < log.size(); ++i)
    {
        if (log[i] != ':' || !std::isdigit(static_cast<unsigned char>(log[i + 1])))
        {
            continue;
        }

        int line = 0;
        const char* const last = log.data() + log.size();
        const auto [end, ec] = std::from_chars(log.data() + i + 1, last, line);
        if (end == last || *end != ':')
        {
            continue;
        }
        if (ec != std::errc{})
        {
            return {}; // a line number that overflows int: give up, as before
        }
        ErrorLocation loc;
        loc.line = line;
        std::size_t start = log.rfind('\n', i);
        start = (start == std::string::npos) ? 0 : start + 1;
        loc.path = log.substr(start, i - start);
        return loc;
    }
    return {};
}
