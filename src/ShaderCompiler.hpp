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

#include "Context.hpp"
#include "Error.hpp"

enum class ShaderStage {
    VERTEX,
    FRAGMENT,
    COMPUTE
};

// A real switch, deliberately with no default case: adding a new stage makes
// every conversion site a compiler error instead of silently aliasing the new
// stage onto FRAGMENT, which is what the old `stage == VERTEX ? ... : ...`
// ternaries scattered across Pipeline.hpp and CommandBuffer.hpp would have done.
inline constexpr VkShaderStageFlagBits to_vk(ShaderStage stage) {
    switch (stage) {
        case ShaderStage::VERTEX:   return VK_SHADER_STAGE_VERTEX_BIT;
        case ShaderStage::FRAGMENT: return VK_SHADER_STAGE_FRAGMENT_BIT;
        case ShaderStage::COMPUTE:  return VK_SHADER_STAGE_COMPUTE_BIT;
    }
    // Not std::unreachable(): pybind enums accept arbitrary ints, so a forged
    // ShaderStage from Python must degrade gracefully, not invoke UB.
    return VK_SHADER_STAGE_VERTEX_BIT;
}

inline constexpr shaderc_shader_kind to_shaderc_kind(ShaderStage stage) {
    switch (stage) {
        case ShaderStage::VERTEX:   return shaderc_glsl_vertex_shader;
        case ShaderStage::FRAGMENT: return shaderc_glsl_fragment_shader;
        case ShaderStage::COMPUTE:  return shaderc_glsl_compute_shader;
    }
    return shaderc_glsl_vertex_shader;
}

class ShaderModule {
public:
    ShaderModule(std::shared_ptr<Context> context, VkShaderModule module, const std::string& path,
                 ShaderStage stage, std::vector<std::string> includes, std::vector<uint32_t> spirv)
        : context_(context), module_(module), path_(path), stage_(stage),
          includes_(std::move(includes)), spirv_(std::move(spirv)) {}

    ~ShaderModule() {
        destroy();
    }

    ShaderModule(const ShaderModule&) = delete;
    ShaderModule& operator=(const ShaderModule&) = delete;

    ShaderModule(ShaderModule&& other) noexcept
        : context_(std::move(other.context_)), module_(other.module_), path_(std::move(other.path_)),
          stage_(other.stage_),
          includes_(std::move(other.includes_)), spirv_(std::move(other.spirv_)) {
        other.module_ = VK_NULL_HANDLE;
    }

    ShaderModule& operator=(ShaderModule&& other) noexcept {
        if (this != &other) {
            destroy();
            context_ = std::move(other.context_);
            module_ = other.module_;
            path_ = std::move(other.path_);
            stage_ = other.stage_;
            includes_ = std::move(other.includes_);
            spirv_ = std::move(other.spirv_);
            other.module_ = VK_NULL_HANDLE;
        }
        return *this;
    }

    VkShaderModule get() const { return module_; }
    const std::string& path() const { return path_; }
    // The stage the module was compiled for — not derivable from the file
    // extension, so it must be remembered for the hot-reload recompile.
    ShaderStage stage() const { return stage_; }
    // Files pulled in via #include, absolute and normalized. The 0.8 hot-reload
    // watcher watches path() plus these; empty for .spv and include-free sources.
    const std::vector<std::string>& includes() const { return includes_; }
    const std::vector<uint32_t>& spirv() const { return spirv_; }

    // Swap in a freshly compiled body (hot reload). MAIN THREAD ONLY: the
    // compile that produced these parts ran an unlocked RecordingIncluder. The
    // old handle retires through the deletion queue rather than being destroyed
    // inline — a pipeline the watcher is about to rebuild() still names it until
    // that rebuild picks up the new handle, and an in-flight frame may still
    // hold the old VkPipeline built from it. Pipelines pick the new module up on
    // their next rebuild(); the ShaderModule object identity never changes, so
    // the watcher's weak_ptr and every builder's shared_ptr stay valid.
    void replace(VkShaderModule module, std::vector<std::string> includes,
                 std::vector<uint32_t> spirv) {
        if (module_ != VK_NULL_HANDLE && context_) {
            context_->defer_destroy([device = context_->device(), old = module_] {
                vkDestroyShaderModule(device, old, nullptr);
            });
        }
        module_ = module;
        includes_ = std::move(includes);
        spirv_ = std::move(spirv);
    }

private:
    void destroy() {
        if (module_ != VK_NULL_HANDLE && context_) {
            vkDestroyShaderModule(context_->device(), module_, nullptr);
        }
    }

    std::shared_ptr<Context> context_;
    VkShaderModule module_;
    std::string path_;
    ShaderStage stage_;
    std::vector<std::string> includes_;
    std::vector<uint32_t> spirv_;
};

// Resolves #include relative to the directory of the INCLUDING file (both "..."
// and <...> forms — one rule, applied recursively: an include inside an include
// resolves against the inner file's directory) and records every file it hands
// out, absolute and normalized, so the 0.8 hot-reload watcher can watch them.
// One instance serves exactly one compile() call on the caller's thread, hence
// no locking — the 0.8 watcher must keep recompilation on the main thread.
class RecordingIncluder final : public shaderc::CompileOptions::IncluderInterface {
public:
    shaderc_include_result* GetInclude(const char* requested_source,
                                       shaderc_include_type /*type*/,
                                       const char* requesting_source,
                                       size_t /*include_depth*/) override {
        namespace fs = std::filesystem;
        fs::path raw = fs::path(requesting_source).parent_path() / requested_source;
        std::error_code ec;
        fs::path resolved = fs::weakly_canonical(raw, ec);
        if (ec) {
            resolved = raw;
        }

        // Each result gets its own heap Holder: shaderc may hold several results
        // at once, and every one must stay valid until its ReleaseInclude.
        auto* holder = new Holder{};

        std::ifstream file(resolved, std::ios::ate | std::ios::binary);
        if (!file.is_open()) {
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
        holder->result = {holder->name.c_str(), holder->name.size(),
                          holder->content.c_str(), holder->content.size(), holder};

        if (!std::ranges::contains(included_, holder->name)) {
            included_.push_back(holder->name);
        }
        return &holder->result;
    }

    void ReleaseInclude(shaderc_include_result* result) override {
        delete static_cast<Holder*>(result->user_data);
    }

    const std::vector<std::string>& included() const { return included_; }

private:
    struct Holder {
        std::string name;
        std::string content;
        shaderc_include_result result{};
    };

    std::vector<std::string> included_;
};

class ShaderCompiler {
public:
    // The compiled body of a shader without the ShaderModule wrapper: a fresh
    // VkShaderModule plus the include list and SPIR-V that go with it. compile()
    // wraps these into a new ShaderModule; the hot-reload watcher feeds them to
    // ShaderModule::replace() so the module object identity survives a reload.
    struct CompiledParts {
        VkShaderModule module;
        std::vector<std::string> includes;
        std::vector<uint32_t> spirv;
    };

    // One entry point for every shader form. The extension of `path` decides how
    // it is handled (GLSL by default); when `source` is given the file is never
    // opened and `path` is a virtual name — it still supplies the language, the
    // diagnostic tag, ShaderError.path, and the base directory for #include.
    static std::expected<std::shared_ptr<ShaderModule>, Error> compile(
            Context& context, const std::string& path, ShaderStage stage,
            std::optional<std::string> source = std::nullopt) {
        auto parts = compile_parts(context, path, stage, std::move(source));
        if (!parts) {
            return std::unexpected(parts.error());
        }
        return std::make_shared<ShaderModule>(context.shared_from_this(), parts->module, path,
                                              stage, std::move(parts->includes), std::move(parts->spirv));
    }

    // The compile without the wrapper. Reads the file fresh from `path` (unless
    // `source` overrides it), so a watcher recompiles simply by calling this
    // again with the module's stored path and stage. MAIN THREAD ONLY when it
    // compiles text (RecordingIncluder is unlocked); .spv loading is thread-safe
    // but shares the entry point for one obvious way.
    static std::expected<CompiledParts, Error> compile_parts(
            Context& context, const std::string& path, ShaderStage stage,
            std::optional<std::string> source = std::nullopt) {
        if (lowercase_extension(path) == ".spv") {
            if (source) {
                return std::unexpected(err_shader(
                    "source= provides text for compilation; .spv is a binary format — pass a file path",
                    path));
            }
            return load_spv(context, path, stage);
        }
        return compile_text(context, path, stage, std::move(source));
    }

private:
    static std::string lowercase_extension(const std::string& path) {
        std::string ext = std::filesystem::path(path).extension().string();
        for (char& c : ext) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        return ext;
    }

    static std::expected<CompiledParts, Error> compile_text(
            Context& context, const std::string& path, ShaderStage stage,
            std::optional<std::string> source) {
        std::string text;
        if (source) {
            text = std::move(*source);
        } else {
            std::ifstream file(path, std::ios::ate | std::ios::binary);
            if (!file.is_open()) {
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
        shaderc_env_version env_version =
            VK_API_VERSION_MINOR(context.api_version()) >= 3
                ? shaderc_env_version_vulkan_1_3
                : shaderc_env_version_vulkan_1_2;
        options.SetTargetEnvironment(shaderc_target_env_vulkan, env_version);
        options.SetOptimizationLevel(shaderc_optimization_level_performance);

        // Language is an attribute of the file name, not a second API path.
        // Entry point stays "main" (shaderc's HLSL default) — one file per stage.
        if (lowercase_extension(path) == ".hlsl") {
            options.SetSourceLanguage(shaderc_source_language_hlsl);
        }

        // Keep a raw pointer before the unique_ptr moves into options; the
        // recorded includes are read back only while `options` is alive.
        auto includer = std::make_unique<RecordingIncluder>();
        RecordingIncluder* recorder = includer.get();
        options.SetIncluder(std::move(includer));

        shaderc_shader_kind kind = to_shaderc_kind(stage);

        shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(text, kind, path.c_str(), options);

        if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
            std::string log = module.GetErrorMessage();
            auto loc = parse_error_location(log);
            return std::unexpected(err_shader("Shader compilation failed: " + log,
                                              loc.path.empty() ? path : loc.path, loc.line));
        }

        std::vector<uint32_t> spirv(module.cbegin(), module.cend());
        auto vk_module = make_vk_module(context, spirv);
        if (!vk_module) {
            return std::unexpected(vk_module.error());
        }
        return CompiledParts{*vk_module, recorder->included(), std::move(spirv)};
    }

    static std::expected<CompiledParts, Error> load_spv(
            Context& context, const std::string& path, ShaderStage stage) {
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (!file.is_open()) {
            return std::unexpected(err_resource("Failed to open shader file: " + path));
        }

        size_t fileSize = static_cast<size_t>(file.tellg());
        if (fileSize % sizeof(uint32_t) != 0) {
            return std::unexpected(err_shader(path + " is not a SPIR-V binary (size is not a multiple of 4)", path));
        }

        std::vector<uint32_t> spirv(fileSize / sizeof(uint32_t));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(spirv.data()), fileSize);

        constexpr uint32_t spirv_magic = 0x07230203u;
        if (spirv.empty() || spirv[0] != spirv_magic) {
            return std::unexpected(err_shader(path + " is not a SPIR-V binary (bad magic number)", path));
        }

        if (!spv_declares_stage(spirv, stage)) {
            return std::unexpected(err_shader(
                std::format("{} declares no {} entry point — the binary was built for a different stage",
                            path, stage_name(stage)),
                path));
        }

        auto vk_module = make_vk_module(context, spirv);
        if (!vk_module) {
            return std::unexpected(vk_module.error());
        }
        return CompiledParts{*vk_module, {}, std::move(spirv)};
    }

    static constexpr const char* stage_name(ShaderStage stage) {
        switch (stage) {
            case ShaderStage::VERTEX:   return "VERTEX";
            case ShaderStage::FRAGMENT: return "FRAGMENT";
            case ShaderStage::COMPUTE:  return "COMPUTE";
        }
        return "unknown";
    }

    // True when ANY OpEntryPoint in the binary matches `stage` — multi-entry-
    // point modules are legal SPIR-V. Walks instructions from word 5; a zero
    // word count means a malformed binary, and bailing out (-> "no entry point
    // found") beats looping forever on garbage.
    static bool spv_declares_stage(const std::vector<uint32_t>& words, ShaderStage stage) {
        // Sentinel matches no execution model: a forged pybind int degrades to
        // "no entry point found" instead of UB. No default case — a new stage
        // must be a compiler diagnostic here, per project convention.
        uint32_t wanted = 0xFFFFFFFFu;
        switch (stage) {
            case ShaderStage::VERTEX:   wanted = 0; break;   // ExecutionModel Vertex
            case ShaderStage::FRAGMENT: wanted = 4; break;   // ExecutionModel Fragment
            case ShaderStage::COMPUTE:  wanted = 5; break;   // ExecutionModel GLCompute
        }

        constexpr uint32_t op_entry_point = 15;
        std::size_t i = 5;   // header is 5 words
        while (i < words.size()) {
            uint32_t opcode = words[i] & 0xFFFFu;
            uint32_t count = words[i] >> 16;
            if (count == 0) {
                return false;
            }
            if (opcode == op_entry_point && i + 1 < words.size() && words[i + 1] == wanted) {
                return true;
            }
            i += count;
        }
        return false;
    }

    static std::expected<VkShaderModule, Error> make_vk_module(
            Context& context, const std::vector<uint32_t>& spirv) {
        VkShaderModuleCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .codeSize = spirv.size() * sizeof(uint32_t),
            .pCode = spirv.data()
        };

        VkShaderModule vk_module;
        if (auto e = check(vkCreateShaderModule(context.device(), &createInfo, nullptr, &vk_module),
                           "create shader module", ErrorCode::Shader)) {
            return std::unexpected(*e);
        }

        return vk_module;
    }

    struct ErrorLocation {
        std::string path;   // empty when the log didn't match — caller falls back
        int line = -1;      // -1 when unknown; honest beats a wrong guess
    };

    // shaderc formats diagnostics as "<name>:<line>: error: ...", so the first
    // ":<digits>:" is the line, and the name is everything from the start of
    // that LOG LINE (after the previous '\n', not the start of the whole log —
    // earlier diagnostics would otherwise be swallowed into the name). With the
    // includer active the name may be an INCLUDED file: exactly what
    // ShaderError.path should say, so the user — and the 0.8 watcher — opens
    // the file the error is actually in. Windows drive colons ("C:/x.frag:12:")
    // are safe: a ':' followed by a non-digit just keeps the scan moving.
    static ErrorLocation parse_error_location(const std::string& log) {
        for (std::size_t i = 0; i + 1 < log.size(); ++i) {
            if (log[i] != ':') {
                continue;
            }

            std::size_t j = i + 1;
            while (j < log.size() && std::isdigit(static_cast<unsigned char>(log[j]))) {
                ++j;
            }

            if (j > i + 1 && j < log.size() && log[j] == ':') {
                ErrorLocation loc;
                try {
                    loc.line = std::stoi(log.substr(i + 1, j - i - 1));
                } catch (const std::exception&) {
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
