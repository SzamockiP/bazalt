#pragma once
#include <volk.h>
#include <VkBootstrap.h>
#include <vk_mem_alloc.h>

#include <atomic>
#include <deque>
#include <expected>
#include <format>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Error.hpp"
#include "Features.hpp"
#include "Logger.hpp"
#include "Sampler.hpp"

// How hard to try to turn on the validation layers.
//
// Auto is the default and never fails: end-user machines generally have no
// layers installed, and a missing layer is not a reason to refuse to render.
//
// Sync is On plus synchronization validation. Core validation is blind to
// missing barriers; this feature is what makes "the manual-barrier mode is
// really manual" testable at all. Costly — a debugging mode, not a default.
enum class ValidationMode
{
    Off,
    Auto,
    On,
    Sync,
};

// The async upload machinery, seen from the Context's side. A tiny virtual
// interface rather than the real class: UploadManager.hpp needs Context.hpp
// (queues, timeline, deletion queue), so Context can only know it abstractly.
// main.cpp creates the concrete UploadManager lazily on the first load_image.
class UploadManagerBase
{
public:
    virtual ~UploadManagerBase() = default;
    virtual bool uploads_done() = 0;
    virtual double upload_progress() = 0;
    virtual void wait_all() = 0;
};

// These live in headers that include Context.hpp, so Context can only name them.
class ShaderModule;
class Pipeline;
class Image;

// The hot-reload file watcher, seen from the Context's side — same abstract
// interface trick as UploadManagerBase (the concrete HotReloadWatcher lives in
// HotReload.hpp, which needs the full Pipeline/ShaderModule/Image/UploadManager
// definitions). Created in the Context binding when hot_reload=True.
//
// watch_* register a resource for watching (called from the Python bindings on
// the main thread). drain() applies whatever changed since the last call and is
// MAIN THREAD ONLY — it recompiles shaders (the includer is unlocked) and calls
// vkCreate*. The frame path and the headless submit path both drain it.
class HotReloadBase
{
public:
    virtual ~HotReloadBase() = default;
    virtual void watch_shader(std::shared_ptr<ShaderModule> module) = 0;
    virtual void watch_pipeline(std::shared_ptr<Pipeline> pipeline) = 0;
    virtual void watch_image(std::shared_ptr<Image> image, std::string path) = 0;
    virtual void drain() = 0;
};

// Everything the caller can ask of a Context, in capability terms.
struct ContextConfig
{
    ValidationMode validation = ValidationMode::Auto;
    std::vector<Feature> required;
    std::vector<Feature> optional;

    // How many frames may be recorded ahead of the GPU. 2 is the classic
    // latency/throughput trade-off; 1 is legal and useful for debugging.
    std::uint32_t frames_in_flight = 2;

    // Barriers between resources (SSBO -> vertex read, dispatch -> dispatch)
    // are computed automatically at record time. False means every one of them
    // is the caller's job via cmd.barrier(). Attachment layout transitions in
    // begin/end_rendering are NOT covered by this switch — they are the
    // RenderTarget contract and stay automatic always.
    bool auto_barriers = true;

    // frame.gpu_time_ms: a timestamp pair recorded around every windowed submit.
    // Off by default because it is a profiling diagnostic
    bool gpu_timing = false;

    // Escape hatch, documented as "you shouldn't need this". Present so that the
    // capability abstraction never becomes a ceiling.
    std::vector<std::string> raw_extensions;
};

class Context : public std::enable_shared_from_this<Context>
{
public:
    static std::expected<std::shared_ptr<Context>, Error> create(
        std::shared_ptr<Logger> logger,
        const ContextConfig& config = {})
    {
        // volk installs its function pointers as globals, and volkLoadDevice binds
        // them to one specific VkDevice. A second live Context therefore silently
        // redirects the first one's GPU calls at its own device — an access
        // violation with no diagnostic whatsoever. Sequential create/destroy is
        // fine; only overlap is not.
        //
        // The real fix is volkLoadDeviceTable + per-Context dispatch tables, which
        // touches every vk* call site. Until that's worth doing, fail loudly.
        if (live_contexts_.load() > 0)
        {
            return std::unexpected(err_init(
                "Only one bazalt Context can exist at a time. volk binds its global "
                "function pointers to a single device, so a second Context would "
                "silently corrupt the first. Destroy the existing Context before "
                "creating another (creating them one after another is fine)."));
        }

        if (config.frames_in_flight < 1 || config.frames_in_flight > 4)
        {
            return std::unexpected(
                err_init(std::format("frames_in_flight must be between 1 and 4, got {}", config.frames_in_flight)));
        }

        auto context = std::shared_ptr<Context>(new Context(logger));
        context->frames_in_flight_ = config.frames_in_flight;
        context->auto_barriers_ = config.auto_barriers;
        context->gpu_timing_ = config.gpu_timing;

        auto target_api = create_instance_(*context, config, logger);
        if (!target_api)
        {
            return std::unexpected(target_api.error());
        }
        if (auto r = select_physical_device_(*context, config, *target_api); !r)
        {
            return std::unexpected(r.error());
        }
        if (auto r = configure_features_(*context, config, logger, *target_api); !r)
        {
            return std::unexpected(r.error());
        }
        if (auto r = create_device_(*context); !r)
        {
            return std::unexpected(r.error());
        }
        if (auto r = create_allocator_and_pool_(*context); !r)
        {
            return std::unexpected(r.error());
        }

        if (logger)
        {
            // Spell out which path was negotiated: a bug report from an unfamiliar
            // machine is unreadable without knowing whether it took the 1.3-core or
            // the 1.2+KHR route, and whether it went headless.
            logger->log(
                Severity::Info,
                Source::General,
                std::format(
                    "Vulkan: Initialized ({}, API {}, dynamic rendering: {}{})",
                    context->device_name(),
                    api_version_string(context->api_version()),
                    context->dynamic_rendering_khr_ ? "KHR extension" : "core",
                    context->headless_ ? ", headless" : ""));
        }

        // Last thing before handing the Context over: only a fully built Context
        // counts as live, so a failed create() doesn't leave the slot occupied.
        context->registered_ = true;
        live_contexts_.fetch_add(1);

        return context;
    }

    ~Context()
    {
        if (registered_)
        {
            live_contexts_.fetch_sub(1);
        }

        // Before anything it observes goes away: stop the watcher thread. It
        // touches no Vulkan and no Python (errors go through the Logger's own
        // queue), so joining it is unconditional and needs no atexit dance — the
        // jthread destructor requests the stop and joins.
        hot_reload_.reset();

        // First: stop the upload worker. Its destructor abandons undecoded jobs
        // (the process is going away; decoding more would only delay exit),
        // finishes at most one in-flight submit, joins, and destroys its command
        // pool — all of which needs the device still alive.
        upload_manager_.reset();

        if (vkb_device_.device)
        {
            vkDeviceWaitIdle(vkb_device_.device);
        }

        // Everything is complete now; run whatever is still queued before the
        // pool and allocator below disappear out from under the lambdas.
        for (auto& [serial, fn] : deletion_queue_)
        {
            fn();
        }
        deletion_queue_.clear();

        for (auto& [key, sampler] : sampler_cache_)
        {
            vkDestroySampler(vkb_device_.device, sampler->get(), nullptr);
        }
        sampler_cache_.clear();

        if (submit_timeline_)
        {
            vkDestroySemaphore(vkb_device_.device, submit_timeline_, nullptr);
        }

        if (command_pool_)
        {
            vkDestroyCommandPool(vkb_device_.device, command_pool_, nullptr);
        }

        if (allocator_)
        {
            vmaDestroyAllocator(allocator_);
        }

        vkb::destroy_device(vkb_device_);
        vkb::destroy_instance(vkb_instance_);
    }

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    VkInstance instance() const
    {
        return vkb_instance_.instance;
    }
    VkDevice device() const
    {
        return vkb_device_.device;
    }
    VkPhysicalDevice physical_device() const
    {
        return vkb_physical_device_.physical_device;
    }
    VkQueue graphics_queue() const
    {
        return graphics_queue_;
    }
    std::uint32_t graphics_queue_family() const
    {
        return graphics_queue_family_;
    }
    VmaAllocator allocator() const
    {
        return allocator_;
    }
    VkCommandPool command_pool() const
    {
        return command_pool_;
    }
    std::shared_ptr<Logger> logger() const
    {
        return logger_;
    }

    // Internal — for use by renderers and other subsystems
    const vkb::Instance& vkb_instance() const
    {
        return vkb_instance_;
    }
    const vkb::Device& vkb_device() const
    {
        return vkb_device_;
    }

    // ── Capabilities ──────────────────────────────────────────────────────────

    bool supports(Feature feature) const
    {
        return enabled_features_.contains(feature);
    }

    std::string device_name() const
    {
        return vkb_physical_device_.properties.deviceName;
    }

    // The *negotiated* version — what the device was actually created against —
    // not the raw properties.apiVersion. A 1.3-capable GPU behind a 1.2 loader
    // runs the 1.2+KHR path, and shaders compiled for 1.3 would be invalid
    // there. Context stays ignorant of shaderc: ShaderCompiler maps this onto
    // a shaderc_env_version itself.
    std::uint32_t api_version() const
    {
        return negotiated_api_version_;
    }

    // True when no windowing extensions were available, so no SwapchainRenderer
    // can be created against this Context.
    bool headless() const
    {
        return headless_;
    }
    bool swapchain_supported() const
    {
        return swapchain_supported_;
    }

    // ── Frame ring ────────────────────────────────────────────────────────────
    //
    // The Context owns and advances the frame counter; renderers and the
    // headless submit path both call begin_frame_internal() when a new frame
    // starts. A monotonic serial rather than a wrapping index, because the
    // deletion queue and upload bookkeeping need "how far has the GPU
    // progressed", which a modulo index cannot answer.
    std::uint32_t frames_in_flight() const
    {
        return frames_in_flight_;
    }
    bool auto_barriers() const
    {
        return auto_barriers_;
    }
    // Whether the swapchain renderer records frame.gpu_time_ms timestamps.
    bool gpu_timing() const
    {
        return gpu_timing_;
    }
    std::uint64_t frame_serial() const
    {
        return frame_serial_;
    }
    std::uint32_t frame_index() const
    {
        return static_cast<std::uint32_t>(frame_serial_ % frames_in_flight_);
    }

    // Not exposed to Python: headless users have exactly one verb (ctx.submit),
    // and a second frame-advancing verb would be a footgun until multiple
    // submits per frame exist.
    //
    // Call sites pick the boundary that keeps `buffer.update()` and the submit
    // that consumes it on the SAME ring slot: SwapchainRenderer::begin_frame
    // advances on entry (updates happen between begin and submit), the headless
    // ctx.submit advances after submitting (updates happen before the call).
    std::uint64_t advance_frame()
    {
        return ++frame_serial_;
    }

    // ── Submission timeline ───────────────────────────────────────────────────
    //
    // One timeline semaphore counts EVERY submission on the graphics queue —
    // frame submits, headless submits, one-shot submits, async uploads. Its
    // counter answers the only synchronization question the CPU side ever asks:
    // "has the GPU passed point X?" — uniformly for windowed and headless, and
    // it is what makes async uploads awaitable.
    VkSemaphore submit_timeline() const
    {
        return submit_timeline_;
    }

    // Reserve the serial the next submit will signal. Call while holding
    // queue_mutex(), immediately before the vkQueueSubmit that signals it.
    std::uint64_t advance_submit_serial()
    {
        return ++submit_serial_;
    }
    std::uint64_t submit_serial() const
    {
        return submit_serial_.load();
    }

    std::uint64_t completed_submit_serial() const
    {
        std::uint64_t value = 0;
        vkGetSemaphoreCounterValue(vkb_device_.device, submit_timeline_, &value);
        return value;
    }

    // ── Deferred destruction ──────────────────────────────────────────────────
    //
    // A handle dropped on the CPU may still be referenced by work the GPU is
    // chewing through, so resource destructors enqueue their vkDestroy calls
    // here instead of running them inline. An entry is keyed by the submit
    // serial at drop time — no later submit can reference the handle (any
    // recording that did held a shared_ptr, which is gone by the time a
    // destructor runs) — and runs once the timeline passes that serial.
    //
    // The lambdas capture raw handles plus the VkDevice/VmaAllocator values —
    // never a shared_ptr<Context>, which would keep the Context alive from its
    // own member and leak everything.
    //
    // Thread-safe: the upload worker parks its staging buffers here too.
    void defer_destroy(std::function<void()> fn)
    {
        std::lock_guard lock(deletion_mutex_);
        deletion_queue_.emplace_back(submit_serial_.load(), std::move(fn));
    }

    void flush_deletion_queue()
    {
        const std::uint64_t completed = completed_submit_serial();

        // Run the ready entries outside the lock: a destructor lambda must be
        // free to enqueue (it doesn't today, but that trap is invisible).
        std::vector<std::function<void()>> ready;
        {
            std::lock_guard lock(deletion_mutex_);
            // Two producers interleave, so keys are not strictly ordered —
            // scan rather than pop-from-front. The queue stays tiny.
            for (auto it = deletion_queue_.begin(); it != deletion_queue_.end();)
            {
                if (it->first <= completed)
                {
                    ready.push_back(std::move(it->second));
                    it = deletion_queue_.erase(it);
                }
                else
                {
                    ++it;
                }
            }
        }
        for (auto& fn : ready)
        {
            fn();
        }
    }

    // ── Async uploads ─────────────────────────────────────────────────────────

    UploadManagerBase* upload_manager() const
    {
        return upload_manager_.get();
    }
    void set_upload_manager(std::unique_ptr<UploadManagerBase> manager)
    {
        upload_manager_ = std::move(manager);
    }

    // ── Hot reload ────────────────────────────────────────────────────────────
    //
    // Null unless the Context was created with hot_reload=True. The frame path
    // (SwapchainRenderer::begin_frame) and the headless submit both drain it on
    // the main thread.
    HotReloadBase* hot_reload() const
    {
        return hot_reload_.get();
    }
    void set_hot_reload(std::unique_ptr<HotReloadBase> watcher)
    {
        hot_reload_ = std::move(watcher);
    }

    // ── Sampler cache ─────────────────────────────────────────────────────────
    //
    // Identical descriptions share one VkSampler; Texture used to create a
    // fresh sampler per texture. Cached handles live until ~Context — the
    // descriptor space is a handful of combinations, never worth evicting.
    std::expected<std::shared_ptr<Sampler>, Error> get_sampler(const SamplerDesc& desc)
    {
        const std::uint32_t key = sampler_cache_key(desc);
        if (auto it = sampler_cache_.find(key); it != sampler_cache_.end())
        {
            return it->second;
        }

        const bool anisotropy = desc.anisotropy && supports(Feature::ANISOTROPIC_FILTERING);
        const VkFilter filter = desc.filter == Filter::NEAREST ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
        VkSamplerAddressMode address = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        switch (desc.address_mode)
        {
            case AddressMode::REPEAT:
                address = VK_SAMPLER_ADDRESS_MODE_REPEAT;
                break;
            case AddressMode::CLAMP:
                address = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                break;
            case AddressMode::MIRROR:
                address = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
                break;
        }

        VkSamplerCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .magFilter = filter,
            .minFilter = filter,
            .mipmapMode = desc.filter == Filter::NEAREST ? VK_SAMPLER_MIPMAP_MODE_NEAREST
                                                         : VK_SAMPLER_MIPMAP_MODE_LINEAR,
            .addressModeU = address,
            .addressModeV = address,
            .addressModeW = address,
            .mipLodBias = 0.0f,
            .anisotropyEnable = anisotropy ? VK_TRUE : VK_FALSE,
            .maxAnisotropy = anisotropy ? 16.0f : 1.0f,
            .compareEnable = desc.compare ? VK_TRUE : VK_FALSE,
            .compareOp = desc.compare ? to_vk(*desc.compare) : VK_COMPARE_OP_ALWAYS,
            .minLod = 0.0f,
            // The whole mip chain. The old per-texture sampler had maxLod = 0,
            // which would have clamped every mip away the moment mips existed.
            .maxLod = VK_LOD_CLAMP_NONE,
            .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
            .unnormalizedCoordinates = VK_FALSE};

        VkSampler handle = VK_NULL_HANDLE;
        if (auto e = check(
                vkCreateSampler(vkb_device_.device, &info, nullptr, &handle), "create sampler", ErrorCode::Resource))
        {
            return std::unexpected(*e);
        }

        auto sampler = std::make_shared<Sampler>(handle, desc);
        sampler_cache_.emplace(key, sampler);
        return sampler;
    }

    // ── Debug object names ────────────────────────────────────────────────────
    //
    // Attach `name` to a Vulkan handle so validation messages name the culprit
    // (the Filar A philosophy: diagnostics should say who). A silent no-op when
    // the name is empty or VK_EXT_debug_utils is not enabled — vk-bootstrap only
    // requests it when a debug callback is set, i.e. when validation is on, and
    // volk then leaves vkSetDebugUtilsObjectNameEXT null. So names cost nothing
    // in a release run and simply do not appear.
    void set_debug_name(VkObjectType type, std::uint64_t handle, const std::string& name)
    {
        if (name.empty() || handle == 0 || vkSetDebugUtilsObjectNameEXT == nullptr)
        {
            return;
        }
        VkDebugUtilsObjectNameInfoEXT info{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
            .pNext = nullptr,
            .objectType = type,
            .objectHandle = handle,
            .pObjectName = name.c_str()};
        vkSetDebugUtilsObjectNameEXT(vkb_device_.device, &info);
    }

    // Guards against two SwapchainRenderers fighting over the frame ring.
    bool has_swapchain_renderer() const
    {
        return has_swapchain_renderer_;
    }
    void set_has_swapchain_renderer(bool value)
    {
        has_swapchain_renderer_ = value;
    }

    // VkQueue is externally synchronized. Today every submit happens on the main
    // thread, so this mutex is uncontended — it exists because 0.5's upload
    // worker submits from its own thread, and every vkQueueSubmit/Present/
    // WaitIdle must hold it from then on.
    std::mutex& queue_mutex()
    {
        return queue_mutex_;
    }

private:
    Context(std::shared_ptr<Logger> logger)
        : logger_(logger)
    {
    }

    // ── create() steps ────────────────────────────────────────────────────────

    // volk + instance (with validation and the headless fallback). Returns the
    // negotiated instance API version.
    static std::expected<std::uint32_t, Error> create_instance_(
        Context& ctx,
        const ContextConfig& config,
        const std::shared_ptr<Logger>& logger)
    {
        if (auto e = check(volkInitialize(), "initialize volk"))
        {
            return std::unexpected(*e);
        }

        auto system_info = vkb::SystemInfo::get_system_info();
        if (!system_info)
        {
            return std::unexpected(err_init("Vulkan: " + system_info.error().message()));
        }

        // Take 1.3 where it exists, otherwise 1.2. Requiring 1.3 outright — as this
        // used to — rejects older Intel iGPUs, MoltenVK and any driver still on 1.2,
        // which is a large slice of real machines. Everything bazalt needs from 1.3
        // is available on 1.2 through VK_KHR_dynamic_rendering.
        const bool has_1_3 = system_info->is_instance_version_available(1, 3);
        const std::uint32_t target_api = has_1_3 ? VK_API_VERSION_1_3 : VK_API_VERSION_1_2;

        // Instance + Debug Messenger
        auto inst_builder = vkb::InstanceBuilder{}
                                .set_app_name("Bazalt Engine")
                                .set_app_version(1, 0, 0)
                                .require_api_version(target_api);

        // Validation is independent of whether a logger was supplied. It used to
        // be gated on `logger != nullptr`, which meant the default path rendered
        // with the layers off and stayed silent about its own bugs.
        if (config.validation != ValidationMode::Off)
        {
            if (config.validation == ValidationMode::On || config.validation == ValidationMode::Sync)
            {
                inst_builder.enable_validation_layers();
            }
            else
            {
                // Enables the layers only if they are actually present.
                inst_builder.request_validation_layers();
            }

            if (config.validation == ValidationMode::Sync)
            {
                inst_builder.add_validation_feature_enable(VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT);
                // Verified empirically on SDK 1.4.350: without this setting the
                // layer does not track shader descriptor accesses at all, so a
                // missing barrier between two dispatches goes UNREPORTED — which
                // would make the whole Sync mode a placebo.
                //
                // Gated on the extension: add_layer_setting() hard-requires
                // VK_EXT_layer_settings, and older layers (e.g. Ubuntu's apt
                // package) neither expose it nor need it — there the shader
                // accesses are tracked by default and the setting doesn't exist.
                if (system_info->is_extension_available(VK_EXT_LAYER_SETTINGS_EXTENSION_NAME))
                {
                    static const VkBool32 syncval_shader_accesses = VK_TRUE;
                    inst_builder.add_layer_setting(
                        VkLayerSettingEXT{
                            .pLayerName = "VK_LAYER_KHRONOS_validation",
                            .pSettingName = "syncval_shader_accesses_heuristic",
                            .type = VK_LAYER_SETTING_TYPE_BOOL32_EXT,
                            .valueCount = 1,
                            .pValues = &syncval_shader_accesses});
                }
            }

            inst_builder.set_debug_callback(debug_callback)
                .set_debug_callback_user_data_pointer(logger.get())
                .set_debug_messenger_severity(
                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
                .set_debug_messenger_type(
                    VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                    VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT);
        }

        // Surface extensions are NOT enabled by hand. vk-bootstrap already adds the
        // right ones per platform (win32/xcb/xlib/wayland/metal), and the previous
        // hand-rolled enable_extension("VK_KHR_surface") was both redundant and a
        // hard failure path: enable_extension refuses to build when absent.
        // It also silently ignored the extension list GLFW reports.
        for (const auto& extension : config.raw_extensions)
        {
            inst_builder.enable_extension(extension.c_str());
        }

        auto inst_ret = inst_builder.build();

        // A machine with no display has no windowing extensions, and vk-bootstrap
        // treats that as fatal. Fall back to a headless instance instead: whether a
        // window is involved is decided later by creating a SwapchainRenderer (or
        // not), so Context has no business demanding one — and no business making
        // the user pass headless=True to say so.
        if (!inst_ret && inst_ret.error() == vkb::InstanceError::windowing_extensions_not_present)
        {
            inst_builder.set_headless(true);
            inst_ret = inst_builder.build();
            if (inst_ret)
            {
                ctx.headless_ = true;
                if (logger)
                {
                    logger->log(
                        Severity::Info,
                        Source::General,
                        "Vulkan: no windowing extensions present, continuing headless");
                }
            }
        }

        if (!inst_ret)
        {
            Error error = err_init("Vulkan: " + inst_ret.error().message());
            error.result = inst_ret.vk_result();
            return std::unexpected(error);
        }
        ctx.vkb_instance_ = inst_ret.value();
        volkLoadInstance(ctx.vkb_instance_.instance);

        return target_api;
    }

    static std::expected<void, Error> select_physical_device_(
        Context& ctx,
        const ContextConfig& config,
        std::uint32_t target_api)
    {
        // Nothing is required here beyond the API version: swapchain support used to
        // be a required extension, which rejected headless-only GPUs outright and
        // made the require_present(false) on the next line pointless. Optional bits
        // are enabled per-device in configure_features_, once we know what this
        // device actually has.
        auto selector = vkb::PhysicalDeviceSelector{ctx.vkb_instance_}
                            .set_minimum_version(VK_API_VERSION_MAJOR(target_api), VK_API_VERSION_MINOR(target_api))
                            .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
                            .require_present(false);

        // Required features must gate *selection*, so a machine with two GPUs picks
        // the one that can do the job rather than failing on the preferred one.
        VkPhysicalDeviceFeatures required_features{};
        for (Feature feature : config.required)
        {
            enable_feature(required_features, feature);
        }
        selector.set_required_features(required_features);

        auto phys_ret = selector.select();
        if (!phys_ret)
        {
            std::string detail;
            for (Feature feature : config.required)
            {
                detail += (detail.empty() ? "" : ", ");
                detail += feature_name(feature);
            }
            Error error = err_init(
                "Vulkan: no suitable GPU found" +
                (detail.empty() ? std::string() : " for required features [" + detail + "]") + ": " +
                phys_ret.error().message());
            error.result = phys_ret.vk_result();
            return std::unexpected(error);
        }
        ctx.vkb_physical_device_ = phys_ret.value();
        return {};
    }

    // EVERY enable_* call in here must happen BEFORE the DeviceBuilder exists
    // (i.e. before create_device_): its constructor takes the PhysicalDevice
    // *by value*, so anything enabled afterwards is written to a copy and
    // silently dropped. That mistake cost a swapchain that was never enabled on
    // the device — an access violation with no diagnostic.
    static std::expected<void, Error> configure_features_(
        Context& ctx,
        const ContextConfig& config,
        const std::shared_ptr<Logger>& logger,
        std::uint32_t target_api)
    {
        // vkb::PhysicalDevice::features is documented as the *selected* features,
        // not the available ones, so ask the driver directly.
        VkPhysicalDeviceFeatures available{};
        vkGetPhysicalDeviceFeatures(ctx.vkb_physical_device_.physical_device, &available);

        // Dynamic rendering: core in 1.3, an extension on 1.2. Same capability,
        // two spellings — resolve it here so nothing else has to care.
        ctx.dynamic_rendering_khr_ = false;
        const bool has_1_3 = target_api >= VK_API_VERSION_1_3;
        const bool device_has_1_3 = ctx.vkb_physical_device_.properties.apiVersion >= VK_API_VERSION_1_3;

        if (has_1_3 && device_has_1_3)
        {
            // core path — VkPhysicalDeviceVulkan13Features in create_device_
            ctx.negotiated_api_version_ = VK_API_VERSION_1_3;
        }
        else if (ctx.vkb_physical_device_.enable_extension_if_present(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME))
        {
            ctx.dynamic_rendering_khr_ = true;
            ctx.negotiated_api_version_ = VK_API_VERSION_1_2;
        }
        else
        {
            return std::unexpected(err_init(
                "Vulkan: this GPU/driver supports neither Vulkan 1.3 nor "
                "VK_KHR_dynamic_rendering, which bazalt requires. Updating the "
                "graphics driver usually fixes this."));
        }

        // Presentation is optional: a headless or compute-only Context is legitimate.
        // SwapchainRenderer verifies present support at its own creation time.
        ctx.swapchain_supported_ =
            ctx.vkb_physical_device_.enable_extension_if_present(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

        // Optional features: enable what this device happens to have, and record
        // what stuck so supports() can answer honestly.
        VkPhysicalDeviceFeatures enabled_features{};
        for (Feature feature : config.required)
        {
            enable_feature(enabled_features, feature);
            ctx.enabled_features_.insert(feature);
        }
        for (Feature feature : config.optional)
        {
            if (feature_available(available, feature))
            {
                enable_feature(enabled_features, feature);
                ctx.enabled_features_.insert(feature);
            }
            else if (logger)
            {
                logger->log(
                    Severity::Info,
                    Source::General,
                    std::format("Vulkan: optional feature {} is not supported by this GPU", feature_name(feature)));
            }
        }

        // Anisotropic filtering is on by default when present, because Texture uses
        // it. It used to be *required*, which turned a nicety into a reach blocker.
        if (feature_available(available, Feature::ANISOTROPIC_FILTERING))
        {
            enable_feature(enabled_features, Feature::ANISOTROPIC_FILTERING);
            ctx.enabled_features_.insert(Feature::ANISOTROPIC_FILTERING);
        }

        // Timeline semaphores pace the deletion queue and async uploads. They are
        // core in 1.2 (our floor), so the entry points are always loaded — but the
        // feature bit still has to be enabled at device creation, and checking it
        // here turns a cryptic device-creation error code into a sentence.
        VkPhysicalDeviceVulkan12Features available12{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = nullptr};
        VkPhysicalDeviceFeatures2 available2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &available12};
        vkGetPhysicalDeviceFeatures2(ctx.vkb_physical_device_.physical_device, &available2);
        if (!available12.timelineSemaphore)
        {
            return std::unexpected(err_init(
                "Vulkan: this GPU/driver does not support timeline semaphores, which "
                "bazalt requires (they are mandatory in conformant Vulkan 1.2 drivers). "
                "Updating the graphics driver usually fixes this."));
        }

        ctx.vkb_physical_device_.enable_features_if_present(enabled_features);
        return {};
    }

    // Only entered once the PhysicalDevice is final and safe to copy.
    static std::expected<void, Error> create_device_(Context& ctx)
    {
        // shaderDemoteToHelperInvocation / shaderTerminateInvocation are mandatory
        // in Vulkan 1.3, and glslang compiles `discard` to one of those opcodes
        // when targeting SPIR-V 1.6 — so a fragment shader with `discard` breaks
        // unless they are enabled alongside the 1.3 target.
        VkPhysicalDeviceVulkan13Features features13{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            .pNext = nullptr,
            .shaderDemoteToHelperInvocation = VK_TRUE,
            .shaderTerminateInvocation = VK_TRUE,
            .dynamicRendering = VK_TRUE};
        VkPhysicalDeviceDynamicRenderingFeaturesKHR dynamic_rendering_khr{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR,
            .pNext = nullptr,
            .dynamicRendering = VK_TRUE};

        // Core in 1.2, so this rides along on both paths. No KHR aliasing needed,
        // unlike dynamic rendering: the floor is 1.2, so the core entry points
        // (vkWaitSemaphores etc.) are always loaded.
        VkPhysicalDeviceVulkan12Features features12{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            .pNext = nullptr,
            .timelineSemaphore = VK_TRUE};

        auto dev_builder = vkb::DeviceBuilder{ctx.vkb_physical_device_};
        dev_builder.add_pNext(&features12);
        if (ctx.dynamic_rendering_khr_)
        {
            dev_builder.add_pNext(&dynamic_rendering_khr);
        }
        else
        {
            dev_builder.add_pNext(&features13);
        }

        auto dev_ret = dev_builder.build();

        if (!dev_ret)
        {
            Error error = err_init("Vulkan: " + dev_ret.error().message());
            error.result = dev_ret.vk_result();
            return std::unexpected(error);
        }
        ctx.vkb_device_ = dev_ret.value();
        volkLoadDevice(ctx.vkb_device_.device);
        ctx.alias_dynamic_rendering_entry_points();

        auto gq = ctx.vkb_device_.get_queue(vkb::QueueType::graphics);
        if (!gq)
        {
            return std::unexpected(err_init("Vulkan: Failed to get graphics queue"));
        }
        ctx.graphics_queue_ = gq.value();
        ctx.graphics_queue_family_ = ctx.vkb_device_.get_queue_index(vkb::QueueType::graphics).value();

        return {};
    }

    static std::expected<void, Error> create_allocator_and_pool_(Context& ctx)
    {
        VmaVulkanFunctions vulkanFunctions = {};
        vulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
        vulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

        VmaAllocatorCreateInfo allocatorInfo = {};
        allocatorInfo.physicalDevice = ctx.vkb_physical_device_.physical_device;
        allocatorInfo.device = ctx.vkb_device_.device;
        allocatorInfo.instance = ctx.vkb_instance_.instance;
        allocatorInfo.pVulkanFunctions = &vulkanFunctions;
        // The negotiated version, not a hardcoded one: telling VMA 1.3 on the
        // 1.2+KHR path would let it call entry points the device never promised.
        allocatorInfo.vulkanApiVersion = ctx.negotiated_api_version_;

        if (auto e = check(vmaCreateAllocator(&allocatorInfo, &ctx.allocator_), "create VMA allocator"))
        {
            return std::unexpected(*e);
        }

        VkCommandPoolCreateInfo poolInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = ctx.graphics_queue_family_};

        if (auto e = check(
                vkCreateCommandPool(ctx.vkb_device_.device, &poolInfo, nullptr, &ctx.command_pool_),
                "create command pool"))
        {
            return std::unexpected(*e);
        }

        // The submission timeline (see submit_timeline() above). Core 1.2; the
        // feature bit was enabled in create_device_.
        VkSemaphoreTypeCreateInfo timelineType{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
            .pNext = nullptr,
            .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
            .initialValue = 0};
        VkSemaphoreCreateInfo timelineInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &timelineType, .flags = 0};
        if (auto e = check(
                vkCreateSemaphore(ctx.vkb_device_.device, &timelineInfo, nullptr, &ctx.submit_timeline_),
                "create submission timeline semaphore"))
        {
            return std::unexpected(*e);
        }

        return {};
    }

    // On a 1.2 device the dynamic-rendering entry points are loaded under their
    // KHR names and the core symbols stay null. Call sites use the core names
    // (vkCmdBeginRendering), so point them at the KHR implementations here. The
    // structs are already aliases, so nothing else has to know which path we took.
    void alias_dynamic_rendering_entry_points()
    {
#if defined(VK_VERSION_1_3) && defined(VK_KHR_dynamic_rendering)
        if (!vkCmdBeginRendering && vkCmdBeginRenderingKHR)
        {
            vkCmdBeginRendering = vkCmdBeginRenderingKHR;
        }
        if (!vkCmdEndRendering && vkCmdEndRenderingKHR)
        {
            vkCmdEndRendering = vkCmdEndRenderingKHR;
        }
#endif
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
        VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
        VkDebugUtilsMessageTypeFlagsEXT message_type,
        const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
        void* user_data)
    {
        Logger* logger = static_cast<Logger*>(user_data);
        if (logger)
        {
            // The severity travels as data. It used to be glued onto the front of
            // the text as "ERROR: ", which is why the callback named on_error was
            // receiving warnings and info messages indistinguishably.
            Severity severity = Severity::Info;
            if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
            {
                severity = Severity::Error;
            }
            else if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
            {
                severity = Severity::Warning;
            }

            logger->log(severity, Source::Validation, callback_data->pMessage);
        }
        return VK_FALSE;
    }

    std::shared_ptr<Logger> logger_;

    vkb::Instance vkb_instance_;
    vkb::PhysicalDevice vkb_physical_device_;
    vkb::Device vkb_device_;

    VkQueue graphics_queue_ = VK_NULL_HANDLE;
    std::uint32_t graphics_queue_family_ = 0;
    std::mutex queue_mutex_;

    VmaAllocator allocator_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;

    static inline std::atomic<int> live_contexts_{0};

    std::set<Feature> enabled_features_;
    bool registered_ = false;
    bool has_swapchain_renderer_ = false;
    bool headless_ = false;
    bool swapchain_supported_ = false;
    bool dynamic_rendering_khr_ = false;

    // Set by configure_features_: 1.3 on the core path, 1.2 on the KHR path.
    std::uint32_t negotiated_api_version_ = VK_API_VERSION_1_2;

    std::uint32_t frames_in_flight_ = 2;
    bool auto_barriers_ = true;
    bool gpu_timing_ = false;
    std::uint64_t frame_serial_ = 0;

    VkSemaphore submit_timeline_ = VK_NULL_HANDLE;
    std::atomic<std::uint64_t> submit_serial_{0};

    std::mutex deletion_mutex_;
    std::deque<std::pair<std::uint64_t, std::function<void()>>> deletion_queue_;

    std::unordered_map<std::uint32_t, std::shared_ptr<Sampler>> sampler_cache_;
    std::unique_ptr<UploadManagerBase> upload_manager_;
    std::unique_ptr<HotReloadBase> hot_reload_;
};
