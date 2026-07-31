#pragma once
#include <volk.h>
#include <VkBootstrap.h>
#include <vk_mem_alloc.h>

#include <atomic>
#include <cstdlib>
#include <deque>
#include <expected>
#include <format>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Device.hpp"
#include "Error.hpp"
#include "Features.hpp"
// For Format and its device-resolved spelling (vk_format below).
#include "Format.hpp"
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
// bindings/ContextBind.cpp creates the concrete UploadManager right after
// Context::create, so
// this is never null and there is one place that counts uploads.
class UploadManagerBase
{
public:
    virtual ~UploadManagerBase() = default;
    virtual double upload_progress() = 0;
    virtual void wait_all() = 0;
    // An upload that skipped the worker (create_buffer, create_image(array) —
    // already-decoded bytes submitted on the calling thread) joins the same
    // batch, so one counter answers "are the uploads done" for both kinds.
    virtual void note_direct_upload(std::uint64_t serial) = 0;
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

    // debugPrintfEXT() from a shader, delivered through the Logger.
    //
    // A separate switch rather than a fifth ValidationMode: the modes are
    // exclusive states of "how hard do the layers check", and printf composes
    // with any of them — you want it *together with* validation="sync", not
    // instead of it. It is off by default because the layer instruments every
    // shader to implement it, and because it forces unoptimized SPIR-V (see
    // ShaderCompiler: spirv-opt eliminates the non-semantic instructions the
    // printf is made of).
    bool shader_printf = false;

    // Which GPU to run on, as the UUID of a Device from list_devices(). Empty
    // means the automatic choice (prefer discrete, must satisfy `required`) —
    // the only behaviour that existed before 0.14 and still the default, because
    // picking for the user is right until the user knows better.
    std::optional<DeviceUUID> device;

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
        if (auto r = select_physical_device_(*context, config); !r)
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

        return context;
    }

    ~Context()
    {
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
            vk_.vkDeviceWaitIdle(vkb_device_.device);
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
            vk_.vkDestroySampler(vkb_device_.device, sampler->get(), nullptr);
        }
        sampler_cache_.clear();

        if (submit_timeline_)
        {
            vk_.vkDestroySemaphore(vkb_device_.device, submit_timeline_, nullptr);
        }

        if (command_pool_)
        {
            vk_.vkDestroyCommandPool(vkb_device_.device, command_pool_, nullptr);
        }

        if (pipeline_cache_)
        {
            vk_.vkDestroyPipelineCache(vkb_device_.device, pipeline_cache_, nullptr);
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

    // ── Device dispatch ───────────────────────────────────────────────────────
    //
    // Every device-level vk* call in bazalt goes through this table. volk's
    // globals are process-wide and volkLoadDevice binds them to ONE VkDevice, so
    // a second live Context used to silently redirect the first one's GPU calls
    // at its own device. Per-Context tables are what removed that limit — and
    // they are load-bearing for the upload worker and the hot-reload thread,
    // which call into their own device from their own threads.
    //
    // The device-level globals are deliberately never loaded (create_instance_
    // calls volkLoadInstanceOnly). They stay null, so a call site that forgot to
    // go through here dies immediately instead of quietly landing on whichever
    // device happened to be created last.
    //
    // Instance-level calls (vkGetPhysicalDevice*, the WSI queries,
    // vkDestroySurfaceKHR, vkSetDebugUtilsObjectNameEXT) stay on the globals ON
    // PURPOSE: those are loader trampolines that dispatch on the handle they are
    // given, so one pointer is correct for every instance in the process.
    const VolkDeviceTable& vk() const
    {
        return vk_;
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
    // Shared by every pipeline built on this Context, so a second pipeline that
    // repeats work the first one did (the common case under hot reload, where a
    // rebuild differs from its predecessor by one shader) reuses the driver's
    // compilation instead of redoing it. Nothing is written to disk: the cache
    // blob's format is tied to the driver, and persisting it belongs to 1.0
    // together with a frozen API. VK_NULL_HANDLE if creation failed, which
    // vkCreate*Pipelines accepts as "no cache".
    VkPipelineCache pipeline_cache() const
    {
        return pipeline_cache_;
    }
    std::shared_ptr<Logger> logger() const
    {
        return logger_;
    }

    // The VkFormat behind a bazalt Format. Every format maps 1:1 except
    // DEPTH_STENCIL, whose spelling is a per-device choice (the spec guarantees
    // only that one of the two combined formats works), so every call site asks
    // the Context instead of reading format_info().vk directly.
    VkFormat vk_format(Format format) const
    {
        const VkFormat vk = format_info(format).vk;
        return vk != VK_FORMAT_UNDEFINED ? vk : depth_stencil_format_;
    }

    VkFormat depth_stencil_format() const
    {
        return depth_stencil_format_;
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

    // The feature structs this device was created with. Read by the descriptor
    // layout code, which has to ask which descriptor-indexing bits actually stuck
    // rather than assume the ones descriptorIndexing guarantees.
    const DeviceFeatures& negotiated_features() const
    {
        return negotiated_features_;
    }

    // Every pipeline stage that can run a shader ON THIS DEVICE — the stage mask
    // "a shader read/wrote this" resolves to in a barrier.
    //
    // It is per-Context and not a constant, which is the whole point. Vulkan
    // forbids the tessellation and geometry stage bits in a barrier mask unless
    // the matching feature is enabled (VUID-vkCmdPipelineBarrier-srcStageMask-04090
    // and -04091), so a constant wide enough for a tessellating Context is a
    // validation error on every other one. And a constant narrow enough to be
    // always legal silently drops the read a tessellation shader just made. The
    // enabled feature set is the only thing that answers both.
    VkPipelineStageFlags all_shader_stages() const
    {
        return all_shader_stages_;
    }

    // The highest MSAA sample count this GPU can back with *both* a colour and a
    // depth attachment — the intersection is what a RenderTarget actually needs,
    // since a single count has to serve every attachment in one pass. Returned as
    // a plain int (1/2/4/8/…) so `RenderTarget(..., samples=ctx.max_samples())` is
    // the one obvious way to pick a valid count without touching Vulkan flag bits.
    std::uint32_t max_samples() const
    {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(vkb_physical_device_.physical_device, &props);
        VkSampleCountFlags counts = props.limits.framebufferColorSampleCounts &
                                    props.limits.framebufferDepthSampleCounts;
        for (VkSampleCountFlagBits bit :
             {VK_SAMPLE_COUNT_64_BIT,
              VK_SAMPLE_COUNT_32_BIT,
              VK_SAMPLE_COUNT_16_BIT,
              VK_SAMPLE_COUNT_8_BIT,
              VK_SAMPLE_COUNT_4_BIT,
              VK_SAMPLE_COUNT_2_BIT})
        {
            if (counts & bit)
            {
                return static_cast<std::uint32_t>(bit);
            }
        }
        return 1;
    }

    // The largest patch_control_points this GPU accepts. Read straight off the
    // cached properties rather than re-querying like max_samples() does, because
    // vk-bootstrap already filled them (device_name() below reads the same
    // struct). NOT bound for Python on purpose: the guaranteed minimum is 32 and
    // a patch is 3 or 4 vertices in practice, so nobody needs to ask — this exists
    // so the pipeline builder can name the limit in its error instead of letting
    // the validation layers do it.
    std::uint32_t max_patch_control_points() const
    {
        return vkb_physical_device_.properties.limits.maxTessellationPatchSize;
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
    // headless submit path both advance it when a new frame
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
    // Whether debugPrintfEXT() output is delivered. Read by ShaderCompiler, which
    // must not optimize the SPIR-V when it is on.
    bool shader_printf() const
    {
        return shader_printf_;
    }
    std::uint64_t frame_serial() const
    {
        return frame_serial_;
    }
    std::uint32_t frame_index() const
    {
        return static_cast<std::uint32_t>(frame_serial_ % frames_in_flight_);
    }

    // Call sites pick the boundary that keeps `buffer.update()` and the submit
    // that consumes it on the SAME ring slot: begin_frame() advances on entry
    // (updates happen between begin and present), the headless ctx.submit
    // advances after submitting (updates happen before the call).
    std::uint64_t advance_frame()
    {
        return ++frame_serial_;
    }

    // Open a new logical frame. THE frame verb of a windowed loop, and the
    // reason 0.14 could grow a second window: everything below is per-Context,
    // not per-swapchain. The ring slot indexes CommandBuffer's command buffers,
    // DynamicBuffer's per-frame copies and the per-frame descriptor sets — all
    // allocated from pools this Context owns. A renderer only keeps its own
    // fences and semaphores on that slot, so it has no business advancing it:
    // with N windows it would advance N times per logical frame and every
    // consumer above would read a slot nobody wrote.
    //
    // Hot reload and the deletion queue hang off the same boundary — once per
    // frame, not once per window.
    void begin_frame()
    {
        advance_frame();

        // Apply any pending hot reloads before this frame records: pipeline
        // rebuilds are handle swaps and old handles retire through the deletion
        // queue keyed by the current submit serial, so in-flight frames are safe.
        if (hot_reload_)
        {
            hot_reload_->drain();
        }

        // A frame boundary is the natural point to reclaim deferred handles;
        // the submission timeline says how far the GPU actually got.
        flush_deletion_queue();
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
        vk_.vkGetSemaphoreCounterValue(vkb_device_.device, submit_timeline_, &value);
        return value;
    }

    // ── Asynchronous headless submits ─────────────────────────────────────────
    //
    // A headless ctx.submit() blocks on the serial it just signalled, which is
    // right when the next line reads the result and wrong when it does not: a
    // compute prototype
    // that submits in a loop leaves the GPU idle between iterations, and the
    // whole loop runs at the speed of the round trip rather than of the work.
    //
    // submit(wait=False) skips the wait. What replaces it is per-slot pacing:
    // the ring has frames_in_flight slots, and reusing one while its previous
    // submit is still running would overwrite a command buffer in flight — the
    // same hazard the windowed path solves with a fence per slot. Here the
    // timeline already counts every submit, so remembering which serial last
    // used a slot is enough.

    // Records that `serial` is the newest submit occupying the current ring slot.
    void note_slot_submit(std::uint64_t serial)
    {
        if (slot_serial_.size() != frames_in_flight_)
        {
            slot_serial_.assign(frames_in_flight_, 0);
        }
        slot_serial_[frame_serial_ % frames_in_flight_] = serial;
    }

    // Blocks until the submit that last used the current ring slot has finished.
    // Cheap when the slot is free: a timeline wait on a value already reached
    // returns immediately, and 0 is always reached.
    void wait_for_slot()
    {
        if (slot_serial_.size() != frames_in_flight_)
        {
            return;
        }
        // Frame pacing: a failure here surfaces at the next submit, which is
        // where a caller can be told about it.
        static_cast<void>(wait_for_serial(slot_serial_[frame_serial_ % frames_in_flight_]));
    }

    // Blocks until everything this Context started has finished — the uploads
    // still decoding on the worker as well as every submit — then reclaims what
    // the deletion queue was holding for them.
    //
    // This is the one wait verb. A caller who wants less waits on the resource
    // (Buffer::wait, Image::wait); there is nothing that wants more, because
    // vkDeviceWaitIdle would also stall the other Contexts sharing the device.
    std::expected<void, Error> wait_for_submits()
    {
        // Uploads first: a job still in the decode queue has no serial yet, so
        // waiting the timeline before it submits would miss it.
        if (upload_manager_)
        {
            upload_manager_->wait_all();
        }
        auto r = wait_for_serial(submit_serial_.load());
        flush_deletion_queue();
        return r;
    }

    // Submits one already-recorded command buffer on the graphics queue and
    // signals the submission timeline with the serial it returns. Every
    // one-shot submit goes through here — deferred_submit for the main thread,
    // the upload worker for its own — so there is one description of what a
    // submit signals. Takes the queue mutex: the worker is not the only thread
    // that submits.
    //
    // Who frees the command buffer afterwards is the caller's business, and it
    // differs: the main thread parks it in the deletion queue, while the worker
    // must free it back into its own pool from its own thread.
    // after: a serial this work must not start before, or 0 for "no ordering".
    //
    // Submitting in order does NOT execute in order: two submits on one queue
    // overlap unless something says otherwise, and the spec is explicit about it.
    // That is what `after` is for — the upload worker promises that two updates of
    // one image land in call order, and one thread submitting them in sequence is
    // not enough to keep the promise.
    std::expected<std::uint64_t, Error> submit_one_shot(VkCommandBuffer cmd, std::uint64_t after = 0)
    {
        std::lock_guard lock(queue_mutex_);
        const std::uint64_t serial = advance_submit_serial();

        VkSemaphore timeline = submit_timeline_;
        // TRANSFER, not TOP_OF_PIPE: an upload's first real work is a copy, and
        // there is nothing before it worth letting run early.
        const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        const bool ordered = after != 0;
        VkTimelineSemaphoreSubmitInfo timelineInfo{
            .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .waitSemaphoreValueCount = ordered ? 1u : 0u,
            .pWaitSemaphoreValues = ordered ? &after : nullptr,
            .signalSemaphoreValueCount = 1,
            .pSignalSemaphoreValues = &serial};
        VkSubmitInfo submitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = &timelineInfo,
            .waitSemaphoreCount = ordered ? 1u : 0u,
            .pWaitSemaphores = ordered ? &timeline : nullptr,
            .pWaitDstStageMask = ordered ? &wait_stage : nullptr,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmd,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &timeline};
        if (auto e = check(
                vk_.vkQueueSubmit(graphics_queue(), 1, &submitInfo, VK_NULL_HANDLE),
                "submit one-shot command buffer",
                ErrorCode::Resource))
        {
            return std::unexpected(*e);
        }
        return serial;
    }

    // The one place that blocks on the submission timeline. Everything that
    // waits for GPU work — a frame's ring slot, an image upload, a readback —
    // comes through here, so a wait is never wider than the work it waits for.
    std::expected<void, Error> wait_for_serial(std::uint64_t serial)
    {
        if (serial == 0)
        {
            return {};
        }
        VkSemaphore timeline = submit_timeline_;
        VkSemaphoreWaitInfo waitInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            .pNext = nullptr,
            .flags = 0,
            .semaphoreCount = 1,
            .pSemaphores = &timeline,
            .pValues = &serial};
        if (auto e = check(
                vk_.vkWaitSemaphores(vkb_device_.device, &waitInfo, UINT64_MAX),
                "wait for submitted GPU work",
                ErrorCode::Resource))
        {
            return std::unexpected(*e);
        }
        return {};
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

    // An upload that did NOT go through the worker: create_buffer and
    // create_image(array) hand over bytes that are already decoded, so they
    // submit on the calling thread and only skip the wait. It still counts as
    // an upload, so it joins the worker's batch instead of being tracked
    // beside it.
    void note_upload_serial(std::uint64_t serial)
    {
        if (upload_manager_)
        {
            upload_manager_->note_direct_upload(serial);
        }
    }

    // ── Hot reload ────────────────────────────────────────────────────────────
    //
    // Null unless the Context was created with hot_reload=True. The frame path
    // (Context::begin_frame) and the headless submit both drain it on the main
    // thread.
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
    std::expected<std::shared_ptr<Sampler>, Error> get_sampler(const SamplerDesc& desc, const std::string& name = {})
    {
        // Full Vulkan always allows both of these, so they are questions only on a
        // portability driver — and there the layers report a sampler the driver
        // then ignores. Refusing here turns "the shadows look wrong on a Mac" into
        // a sentence naming the capability to ask about (0.22).
        if (desc.compare && !supports(Feature::COMPARISON_SAMPLER))
        {
            return std::unexpected(err_resource(
                "create_sampler(compare=) needs the COMPARISON_SAMPLER feature, which this driver does not "
                "offer. Ask ctx.supports(bz.Feature.COMPARISON_SAMPLER) and sample the depth texture "
                "yourself where it answers False."));
        }
        if (desc.mip_lod_bias != 0.0f && !supports(Feature::SAMPLER_MIP_LOD_BIAS))
        {
            return std::unexpected(err_resource(
                "create_sampler(mip_lod_bias=) needs the SAMPLER_MIP_LOD_BIAS feature, which this driver does "
                "not offer. Ask ctx.supports(bz.Feature.SAMPLER_MIP_LOD_BIAS), or bias the level in the "
                "shader with textureLod()."));
        }

        const std::uint32_t key = sampler_cache_key(desc);
        // The key hashes a float, so equality of keys is not equality of
        // descriptions: compare the description before handing the handle back.
        if (auto it = sampler_cache_.find(key); it != sampler_cache_.end() && it->second->desc() == desc)
        {
            // A cache hit with a new name renames the shared object to list both
            // users. See Sampler::add_debug_name for why that beats dropping the
            // name or splitting the cache entry.
            if (it->second->add_debug_name(name))
            {
                set_debug_name(
                    VK_OBJECT_TYPE_SAMPLER,
                    reinterpret_cast<std::uint64_t>(it->second->get()),
                    it->second->debug_name());
            }
            return it->second;
        }

        const bool anisotropy = desc.anisotropy && supports(Feature::ANISOTROPIC_FILTERING);
        const VkFilter filter = to_vk_filter(desc.filter);
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
            case AddressMode::CLAMP_TO_BORDER:
                address = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
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
            .mipLodBias = desc.mip_lod_bias,
            .anisotropyEnable = anisotropy ? VK_TRUE : VK_FALSE,
            .maxAnisotropy = anisotropy ? 16.0f : 1.0f,
            .compareEnable = desc.compare ? VK_TRUE : VK_FALSE,
            .compareOp = desc.compare ? to_vk(*desc.compare) : VK_COMPARE_OP_ALWAYS,
            .minLod = 0.0f,
            // The whole mip chain. The old per-texture sampler had maxLod = 0,
            // which would have clamped every mip away the moment mips existed.
            .maxLod = VK_LOD_CLAMP_NONE,
            // Float rather than int: every sampled format bazalt exposes reads
            // as floats, and an int border on a float image is undefined.
            .borderColor = desc.border_color == BorderColor::OPAQUE_WHITE ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE
                                                                          : VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
            .unnormalizedCoordinates = VK_FALSE};

        VkSampler handle = VK_NULL_HANDLE;
        if (auto e = check(
                vk_.vkCreateSampler(vkb_device_.device, &info, nullptr, &handle),
                "create sampler",
                ErrorCode::Resource))
        {
            return std::unexpected(*e);
        }

        auto sampler = std::make_shared<Sampler>(handle, desc);
        if (sampler->add_debug_name(name))
        {
            set_debug_name(VK_OBJECT_TYPE_SAMPLER, reinterpret_cast<std::uint64_t>(handle), sampler->debug_name());
        }
        sampler_cache_.insert_or_assign(key, sampler);
        return sampler;
    }

    // ── Memory and device introspection ───────────────────────────────────────

    // How much GPU memory this Context has allocated, and how much the driver
    // says is left, in bytes. "Am I leaking, and how much room is there" is a
    // question a prototype asks constantly, and the answer used to need an
    // external tool.
    //
    // VMA already keeps the numbers, so this is a read rather than new
    // bookkeeping. Summed across heaps: per-heap detail is a different question
    // (which heap is a driver decision), and a single pair is what the question
    // above actually wants.
    struct MemoryStats
    {
        std::uint64_t used = 0;     // bytes VMA has allocated for this Context
        std::uint64_t reserved = 0; // bytes VMA has reserved from the driver
        std::uint64_t budget = 0;   // bytes the driver says the process may use
    };

    MemoryStats memory_stats() const
    {
        VkPhysicalDeviceMemoryProperties memory_props{};
        vkGetPhysicalDeviceMemoryProperties(vkb_physical_device_.physical_device, &memory_props);

        std::vector<VmaBudget> budgets(memory_props.memoryHeapCount);
        vmaGetHeapBudgets(allocator_, budgets.data());

        MemoryStats stats;
        for (std::uint32_t i = 0; i < memory_props.memoryHeapCount; ++i)
        {
            stats.used += budgets[i].statistics.allocationBytes;
            stats.reserved += budgets[i].statistics.blockBytes;
            stats.budget += budgets[i].budget;
        }
        return stats;
    }

    // The subgroup width this GPU runs shaders at, or 0 where the driver does not
    // report one.
    //
    // A compute shader doing a subgroupAdd reduction has to size its workgroup
    // against this number, and there was no way to ask. Vulkan 1.1 core, so no
    // negotiation and no Feature: the property either has a value or it does not.
    std::uint32_t subgroup_size() const
    {
        VkPhysicalDeviceSubgroupProperties subgroup{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES, .pNext = nullptr};
        VkPhysicalDeviceProperties2 props{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &subgroup};
        vkGetPhysicalDeviceProperties2(vkb_physical_device_.physical_device, &props);
        return subgroup.subgroupSize;
    }

    // ── Debug object names ────────────────────────────────────────────────────
    //
    // Attach `name` to a Vulkan handle so validation messages name the culprit
    // (the Filar A philosophy: diagnostics should say who). A silent no-op when
    // the name is empty or VK_EXT_debug_utils is not enabled — vk-bootstrap only
    // requests it when a debug callback is set, i.e. when validation is on, and
    // volk then leaves vkSetDebugUtilsObjectNameEXT null. So names cost nothing
    // in a release run and simply do not appear.
    //
    // This one stays on the global rather than moving to vk(): debug utils is an
    // INSTANCE extension, so vkGetInstanceProcAddr is the sanctioned way to fetch
    // it (volkLoadInstanceOnly does) and vkGetDeviceProcAddr may legitimately
    // return null for it. The pointer is a loader trampoline dispatching on the
    // VkDevice argument, so it is correct for every Context in the process.
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
        if (volkInitialize() != VK_SUCCESS)
        {
            return std::unexpected(err_no_vulkan_loader());
        }

        // printf is implemented by the validation layers, so the two settings
        // contradict each other. Say so instead of building a Context whose
        // shader_printf silently prints nothing.
        if (config.shader_printf && config.validation == ValidationMode::Off)
        {
            return std::unexpected(err_init(
                "shader_printf=True needs the validation layers, which validation=\"off\" turns off. "
                "Use validation=\"on\" (or leave it at \"auto\")."));
        }
        ctx.shader_printf_ = config.shader_printf;

        auto system_info = vkb::SystemInfo::get_system_info();
        if (!system_info)
        {
            return std::unexpected(err_init("Vulkan: " + system_info.error().message()));
        }

        // Take 1.3 where it exists, otherwise 1.2. Requiring 1.3 outright — as this
        // used to — rejects older Intel iGPUs, MoltenVK and any driver still on 1.2,
        // which is a large slice of real machines. Everything bazalt needs from 1.3
        // is available on 1.2 through VK_KHR_dynamic_rendering.
        // A test/CI knob, not public API (same species as BAZALT_HOT_RELOAD_POLL_MS):
        // negotiate 1.2 even where 1.3 exists, so the 1.2 + VK_KHR_dynamic_rendering
        // path is reachable on a 1.3 machine. Debt #2 — that path had no coverage
        // anywhere since 0.5, because both CI (lavapipe) and the dev GPU report 1.3+.
        // It forces the whole negotiation, not just the aliasing: nulling the core
        // entry points alone would only crash, since a device that never enabled the
        // KHR extension has no KHR entry points to alias to either.
        const char* force_1_2 = std::getenv("BAZALT_FORCE_VULKAN_1_2");
        const bool has_1_3 = system_info->is_instance_version_available(1, 3) && !(force_1_2 && force_1_2[0] == '1');
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

            // Shader printf is a validation-layer service, so it rides the same
            // instance the layers are on. The layer reports the output at INFO
            // severity, which is why the messenger mask has to open up for it —
            // and why debug_callback drops every OTHER info message, so asking
            // for printf does not also subscribe to loader chatter.
            if (config.shader_printf)
            {
                inst_builder.add_validation_feature_enable(VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT);
                // validation="auto" only *requests* the layers, so on a machine
                // with no SDK installed printf has nothing behind it. That is not
                // an error — auto exists precisely to keep running — but it must
                // not look like a shader that never printed.
                if (!system_info->validation_layers_available && logger)
                {
                    logger->log(
                        Severity::Warning,
                        Source::Shader,
                        "shader_printf=True, but the Vulkan validation layers are not installed on this "
                        "machine, so debugPrintfEXT() output cannot be delivered.");
                }
            }

            VkDebugUtilsMessageSeverityFlagsEXT severities = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
                                                             VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
            if (config.shader_printf)
            {
                severities |= VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
            }

            inst_builder.set_debug_callback(debug_callback)
                .set_debug_callback_user_data_pointer(logger.get())
                .set_debug_messenger_severity(severities)
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
        // ...Only, not volkLoadInstance: the device-level globals must stay null
        // so a call site that skipped Context::vk() fails loudly (see vk()). What
        // this does load is instance-level — loader trampolines that dispatch on
        // the handle passed in, hence correct no matter which Context created
        // them last. It also loads the global vkGetDeviceProcAddr that
        // volkLoadDeviceTable needs, so it must run before create_device_.
        volkLoadInstanceOnly(ctx.vkb_instance_.instance);

        return target_api;
    }

    static std::expected<void, Error> select_physical_device_(Context& ctx, const ContextConfig& config)
    {
        // Nothing is required here beyond the API version: swapchain support used to
        // be a required extension, which rejected headless-only GPUs outright and
        // made the require_present(false) on the next line pointless. Optional bits
        // are enabled per-device in configure_features_, once we know what this
        // device actually has.
        //
        // The minimum is the BASELINE, not the version the instance negotiated, and
        // the difference is a real machine rather than a hypothetical one. A device
        // may be older than its loader: on macOS the LunarG loader reports 1.4 while
        // MoltenVK's device reports 1.2. Selecting with the instance's version — which
        // vk-bootstrap also does by default — rejected that device outright, and 0.22
        // found the whole macOS suite failing with "no suitable GPU found". Only the
        // DEVICE version may decide the 1.3-or-KHR path, and configure_features_ has
        // always read it off the device (`device_has_1_3`). This one line was asking
        // the wrong object.
        auto selector = vkb::PhysicalDeviceSelector{ctx.vkb_instance_}
                            .set_minimum_version(1, 2)
                            .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
                            .require_present(false);

        // Required features must gate *selection*, so a machine with two GPUs picks
        // the one that can do the job rather than failing on the preferred one.
        //
        // Only the base-feature column can do that, and the reason is vk-bootstrap's
        // rather than ours: set_required_features_11/12 put a feature struct into the
        // library's own pNext chain, and build() then refuses a chain that also holds
        // ours (VkPhysicalDeviceFeatures2_in_pNext_chain_while_using_add_required_
        // extension_features). Moving every struct into vkb's chain would rewrite the
        // 1.2/1.3 negotiation for a machine that has two GPUs of which only the
        // non-preferred one has descriptor indexing. A required pNext feature is
        // therefore diagnosed in configure_features_ instead, by name.
        DeviceFeatures required_features{};
        for (Feature feature : config.required)
        {
            enable_feature(required_features, feature);
        }
        selector.set_required_features(required_features.core);

        // An explicitly chosen GPU still has to pass the same suitability gate —
        // required features and API version are not preferences. select_devices()
        // is select() without the "and now pick the best one" step, so the choice
        // is the caller's while the filtering stays ours.
        if (config.device)
        {
            auto all = selector.select_devices();
            if (all)
            {
                for (auto& candidate : all.value())
                {
                    if (device_uuid(vkGetPhysicalDeviceProperties2, candidate.physical_device) == *config.device)
                    {
                        ctx.vkb_physical_device_ = candidate;
                        return {};
                    }
                }
            }
            return std::unexpected(err_init(
                "Vulkan: the selected GPU is not available to this Context. It may have "
                "been removed since list_devices(), or it may not meet the requirements "
                "(check device.supports() for every feature passed as required)."));
        }

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
        // not the available ones, so ask the driver directly. One query for all
        // three structs, through the same helper list_devices uses.
        // vk-bootstrap enables VK_KHR_portability_subset by default when the device
        // reports it, so the struct is legal to ask for exactly when this is true.
        ctx.portability_subset_ =
            ctx.vkb_physical_device_.is_extension_present(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
        const DeviceFeatures available = query_device_features(
            vkGetPhysicalDeviceFeatures2, ctx.vkb_physical_device_.physical_device, ctx.portability_subset_);

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

        // debugPrintfEXT() compiles to OpExtInst against the NonSemantic.DebugPrintf
        // instruction set, which the device must permit through
        // VK_KHR_shader_non_semantic_info — core in 1.3, an extension on the 1.2
        // path. Same "one capability, two spellings" shape as dynamic rendering
        // above, so it is resolved here and nothing downstream asks the version.
        //
        // Enabled whenever the device has it, not only for a printf Context. A
        // shader that prints is legal to COMPILE anywhere -- it simply prints
        // nothing without the layers -- and vkCreateShaderModule refuses SPIR-V
        // that declares SPV_KHR_non_semantic_info unless the extension is on. So a
        // printing shader compiled in an ordinary Context was a validation error on
        // every 1.2 device, which before macOS meant a path nothing in CI ran.
        const bool non_semantic_info =
            ctx.negotiated_api_version_ >= VK_API_VERSION_1_3 ||
            ctx.vkb_physical_device_.enable_extension_if_present(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
        if (ctx.shader_printf_ && !non_semantic_info)
        {
            return std::unexpected(err_init(
                "shader_printf=True needs VK_KHR_shader_non_semantic_info (or Vulkan 1.3), which this "
                "GPU/driver does not offer. Updating the graphics driver usually fixes this."));
        }

        // Optional features: enable what this device happens to have, and record
        // what stuck so supports() can answer honestly.
        DeviceFeatures& enabled_features = ctx.negotiated_features_;
        for (Feature feature : config.required)
        {
            // A required feature in a pNext struct could not gate device selection
            // (see select_physical_device_), so this is where it is caught. The
            // base-feature ones cannot reach here missing, because the selector
            // already rejected every device without them.
            if (!feature_available(available, feature))
            {
                return std::unexpected(err_init(
                    std::format(
                        "Vulkan: the required feature {} is not supported by this GPU. Pass it as "
                        "optional and ask ctx.supports() instead, or choose a device whose "
                        "device.supports() answers True.",
                        feature_name(feature))));
            }
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

        // Two capabilities are on by default when present, because bazalt itself
        // uses them and asking for them would be a knob with one sensible setting.
        // ANISOTROPIC_FILTERING used to be *required*, which turned a nicety into a
        // reach blocker. MULTIVIEW joins it in 0.21: target.all_layers() has offered
        // it since 0.13 without an opt-in, and making the Feature row the way to ASK
        // must not also make it a thing to request.
        for (Feature implicit : {Feature::ANISOTROPIC_FILTERING, Feature::MULTIVIEW})
        {
            if (feature_available(available, implicit))
            {
                enable_feature(enabled_features, implicit);
                ctx.enabled_features_.insert(implicit);
            }
        }

        // The portability rows are lifted restrictions, not opt-in capabilities, so
        // they go on whenever the device has them. Nobody would ask for "samplers
        // may compare" — full Vulkan never made it a question — and leaving one off
        // is how a comparison sampler becomes a validation error on a driver that
        // could have done it. Enabling costs nothing: the bit is the driver's own
        // statement that the restriction does not apply.
        //
        // The set is recorded either way (see the loop condition), so supports()
        // answers True on every non-portability driver without a struct to read.
        for (Feature portability :
             {Feature::COMPARISON_SAMPLER, Feature::SAMPLER_MIP_LOD_BIAS, Feature::MULTISAMPLE_ARRAYS})
        {
            if (feature_available(available, portability))
            {
                enable_feature(enabled_features, portability);
                ctx.enabled_features_.insert(portability);
            }
            else if (logger)
            {
                logger->log(
                    Severity::Info,
                    Source::General,
                    std::format(
                        "Vulkan: this driver is a Vulkan portability subset and does not offer {}",
                        feature_name(portability)));
            }
        }

        // descriptorIndexing is a roll-up: the spec says enabling it "does not imply
        // the other minimum descriptor indexing features are also enabled", so the
        // bits an array binding actually needs are turned on one by one.
        if (ctx.enabled_features_.contains(Feature::BINDLESS))
        {
            enable_descriptor_indexing(enabled_features, available);
        }

        // Timeline semaphores pace the deletion queue and async uploads. They are
        // core in 1.2 (our floor), so the entry points are always loaded — but the
        // feature bit still has to be enabled at device creation, and checking it
        // here turns a cryptic device-creation error code into a sentence. Not a
        // Feature row: bazalt does not work without it, so there is nothing to ask.
        enabled_features.v12.timelineSemaphore = VK_TRUE;
        if (!available.v12.timelineSemaphore)
        {
            return std::unexpected(err_init(
                "Vulkan: this GPU/driver does not support timeline semaphores, which "
                "bazalt requires (they are mandatory in conformant Vulkan 1.2 drivers). "
                "Updating the graphics driver usually fixes this."));
        }

        // Computed here, after every insertion into enabled_features_, because
        // that set is the only honest source: a barrier mask may name the
        // tessellation or geometry stage only when the feature behind it is on.
        ctx.all_shader_stages_ = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        if (ctx.enabled_features_.contains(Feature::TESSELLATION))
        {
            ctx.all_shader_stages_ |= VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT |
                                      VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT;
        }
        if (ctx.enabled_features_.contains(Feature::GEOMETRY_SHADER))
        {
            ctx.all_shader_stages_ |= VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT;
        }

        ctx.vkb_physical_device_.enable_features_if_present(enabled_features.core);
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

        // The 1.1 and 1.2 structs come from configure_features_ rather than being
        // spelled again here: it is the only place that knows what the device
        // offered and what the caller asked for, and a second literal would be a
        // second answer. Both rode this path before 0.21 too — timelineSemaphore
        // (core in 1.2, so no KHR aliasing, unlike dynamic rendering) and multiview
        // are simply two of the bits in them now.
        VkPhysicalDeviceVulkan12Features features12 = ctx.negotiated_features_.v12;
        VkPhysicalDeviceVulkan11Features features11 = ctx.negotiated_features_.v11;
        VkPhysicalDevicePortabilitySubsetFeaturesKHR portability = ctx.negotiated_features_.portability;

        auto dev_builder = vkb::DeviceBuilder{ctx.vkb_physical_device_};
        dev_builder.add_pNext(&features12);
        dev_builder.add_pNext(&features11);
        // Only when the device claims the extension. Chaining it otherwise is a
        // struct for an extension nobody enabled, and the layers say so. When the
        // device DOES claim it, this chain is the whole point: an unlisted
        // portability feature defaults to off, so a driver that supports comparison
        // samplers still refuses them if nobody asked.
        if (ctx.portability_subset_)
        {
            dev_builder.add_pNext(&portability);
        }
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
        // Into this Context's own table, never into volk's globals — see vk().
        volkLoadDeviceTable(&ctx.vk_, ctx.vkb_device_.device);
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
                ctx.vk_.vkCreateCommandPool(ctx.vkb_device_.device, &poolInfo, nullptr, &ctx.command_pool_),
                "create command pool"))
        {
            return std::unexpected(*e);
        }

        // Which combined depth/stencil format this device gets. The float
        // variant comes first where it exists: bazalt's plain depth format is
        // D32F, and a depth_bias tuned against a float buffer would mean
        // something else entirely on a 24-bit integer one (the bias is scaled in
        // units of the format). Falling back the other way is still correct —
        // the spec promises at least one of the two.
        for (const VkFormat candidate : {VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT})
        {
            VkFormatProperties props{};
            vkGetPhysicalDeviceFormatProperties(ctx.vkb_physical_device_.physical_device, candidate, &props);
            if (props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
            {
                ctx.depth_stencil_format_ = candidate;
                break;
            }
        }

        // The pipeline cache is an optimization, so a failure here is not an
        // error: vkCreate*Pipelines takes VK_NULL_HANDLE and compiles from
        // scratch, which is exactly what happened before this existed.
        VkPipelineCacheCreateInfo cacheInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .initialDataSize = 0,
            .pInitialData = nullptr};
        if (ctx.vk_.vkCreatePipelineCache(ctx.vkb_device_.device, &cacheInfo, nullptr, &ctx.pipeline_cache_) !=
            VK_SUCCESS)
        {
            ctx.pipeline_cache_ = VK_NULL_HANDLE;
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
                ctx.vk_.vkCreateSemaphore(ctx.vkb_device_.device, &timelineInfo, nullptr, &ctx.submit_timeline_),
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
        if (!vk_.vkCmdBeginRendering && vk_.vkCmdBeginRenderingKHR)
        {
            vk_.vkCmdBeginRendering = vk_.vkCmdBeginRenderingKHR;
        }
        if (!vk_.vkCmdEndRendering && vk_.vkCmdEndRenderingKHR)
        {
            vk_.vkCmdEndRendering = vk_.vkCmdEndRenderingKHR;
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
        if (!logger)
        {
            return VK_FALSE;
        }

        // Shader printf arrives on this same channel, and it is NOT a validation
        // finding: it is the shader talking. Routing it as Source::Shader is what
        // keeps the validation-as-assert test fixture (which fails a test on any
        // Source::VALIDATION error) from treating a print as a bug.
        //
        // Matched on the message id rather than on the text, because the id is
        // structured data and the text is not. The layer has spelled it
        // UNASSIGNED-DEBUG-PRINTF and WARNING-DEBUG-PRINTF across versions, so the
        // stable part is the suffix.
        const std::string_view id_name{callback_data->pMessageIdName ? callback_data->pMessageIdName : ""};
        if (id_name.find("DEBUG-PRINTF") != std::string_view::npos)
        {
            // The layer wraps the shader's own words in its report boilerplate:
            // "Validation Information: [ WARNING-DEBUG-PRINTF ] | MessageID = 0x…
            //  | vkQueueSubmit(): pSubmits[0] DebugPrintf:\n<the print>".
            // Handing that whole blob back per print would bury the value the
            // user asked for behind two hundred characters of object handles,
            // once per print — for a print inside a loop, the difference between
            // a tool and a wall of text.
            //
            // "DebugPrintf:" is the marker this layer puts immediately before the
            // shader's own words, so it is tried first. The two generic
            // separators stay as a fallback for versions that do not emit it, and
            // a message matching none of them passes through whole: peeling is a
            // convenience, and losing the text would be worse than an ugly line.
            std::string_view text{callback_data->pMessage ? callback_data->pMessage : ""};
            if (const auto marker = text.rfind("DebugPrintf:"); marker != std::string_view::npos)
            {
                text.remove_prefix(marker + std::string_view("DebugPrintf:").size());
            }
            else
            {
                if (const auto bar = text.rfind(" | "); bar != std::string_view::npos)
                {
                    text.remove_prefix(bar + 3);
                }
                if (const auto call = text.find("(): "); call != std::string_view::npos)
                {
                    text.remove_prefix(call + 4);
                }
            }
            // The marker is followed by a newline, and the generic separators by
            // padding spaces.
            while (!text.empty() && (text.front() == ' ' || text.front() == '\n' || text.front() == '\t'))
            {
                text.remove_prefix(1);
            }
            while (!text.empty() && (text.back() == ' ' || text.back() == '\n' || text.back() == '\t'))
            {
                text.remove_suffix(1);
            }
            // log_always, not log: the layer reports printf at INFO and the default
            // floor is Warning, so the filter would swallow the channel the user
            // switched on by name.
            logger->log_always(Severity::Info, Source::Shader, std::string(text));
            return VK_FALSE;
        }

        // Everything else at INFO is loader and layer chatter. The messenger only
        // subscribes to INFO when shader_printf is on, so dropping it here is what
        // stops "I want prints" from also meaning "I want that".
        if (!(message_severity &
              (VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)))
        {
            return VK_FALSE;
        }

        // The severity travels as data. It used to be glued onto the front of
        // the text as "ERROR: ", which is why the callback named on_error was
        // receiving warnings and info messages indistinguishably.
        const Severity severity =
            (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ? Severity::Error : Severity::Warning;
        logger->log(severity, Source::Validation, callback_data->pMessage);
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
    VkPipelineCache pipeline_cache_ = VK_NULL_HANDLE;
    VkFormat depth_stencil_format_ = VK_FORMAT_UNDEFINED;

    // Zero-initialized: an entry point the device does not have stays null, which
    // is what makes the KHR aliasing below (and the null checks) work.
    VolkDeviceTable vk_{};

    std::set<Feature> enabled_features_;
    // The same answer as enabled_features_, in the shape vkCreateDevice wants.
    // configure_features_ fills it and create_device_ hands it over, so the set
    // Python asks about and the structs the device was built with cannot drift.
    DeviceFeatures negotiated_features_;
    bool headless_ = false;
    bool swapchain_supported_ = false;
    bool dynamic_rendering_khr_ = false;
    // Whether the device is a Vulkan portability subset (MoltenVK on macOS is the
    // one that exists). Decides whether the portability feature struct may be read
    // and whether it must be chained at device creation.
    bool portability_subset_ = false;

    // Set by configure_features_ from enabled_features_. The default is the mask
    // every conformant device has, so a Context that somehow skipped the
    // computation is narrow (a missing barrier) rather than illegal (a validation
    // error on every barrier).
    VkPipelineStageFlags all_shader_stages_ = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    // Set by configure_features_: 1.3 on the core path, 1.2 on the KHR path.
    std::uint32_t negotiated_api_version_ = VK_API_VERSION_1_2;

    std::uint32_t frames_in_flight_ = 2;
    bool auto_barriers_ = true;
    bool gpu_timing_ = false;
    bool shader_printf_ = false;
    std::uint64_t frame_serial_ = 0;

    VkSemaphore submit_timeline_ = VK_NULL_HANDLE;
    std::atomic<std::uint64_t> submit_serial_{0};

    // Which submit serial last used each ring slot. Only an asynchronous
    // headless submit fills it; a blocking one has already waited.
    std::vector<std::uint64_t> slot_serial_;

    std::mutex deletion_mutex_;
    std::deque<std::pair<std::uint64_t, std::function<void()>>> deletion_queue_;

    std::unordered_map<std::uint32_t, std::shared_ptr<Sampler>> sampler_cache_;
    std::unique_ptr<UploadManagerBase> upload_manager_;
    std::unique_ptr<HotReloadBase> hot_reload_;
};
