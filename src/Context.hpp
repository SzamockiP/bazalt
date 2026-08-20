#pragma once
#include <volk.h>
#include <VkBootstrap.h>
#include <vk_mem_alloc.h>

#include <atomic>
#include <deque>
#include <expected>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
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

// The async upload machinery behind ctx.load_image(). The concrete class lives
// in UploadManager.hpp, which needs Context.hpp (queues, timeline, deletion
// queue), so this header only names it — every Context call into it is in
// Context.cpp. bindings/ContextBind.cpp creates the UploadManager right after
// Context::create, so it is never null and there is one place that counts
// uploads.
class UploadManager;

// Why a GPU query has no number yet, or no number at all (0.24). One nullopt
// used to cover all of these, so a caller could not tell "wait longer" from
// "this GPU cannot" — and the two want opposite reactions. The binding layer
// turns the first three into exceptions and leaves None meaning exactly one
// thing.
//
// Here rather than on CommandBuffer because both askers need it and Renderer.hpp
// only forward-declares CommandBuffer. Namespace scope in the header that owns
// the subject, like Access in ResourceTracker.hpp — and Context is the owner:
// it holds the timeline the answers are paced by and the gpu_timing flag that
// produces Disabled.
enum class QueryStatus
{
    Ok,
    // The device has no usable timestamps: timestampPeriod is 0, or the graphics
    // family reports timestampValidBits == 0. Never true of an occlusion query,
    // which is core with no feature behind it.
    Unsupported,
    // The device could, but nobody asked: Context(gpu_timing=True) is what turns
    // the swapchain renderer's per-frame timestamps on. Only
    // SwapchainRenderer.gpu_time_ms produces this — cmd.timer() needs no flag,
    // because it costs nothing until a recording asks for a timer.
    Disabled,
    // The handle predates a begin(), so its slots now hold a different query's
    // data. A sequencing mistake, not a device limit.
    Superseded,
    // The submit has not finished. The one answer that means "ask again".
    NotReady
};

// The hot-reload file watcher — forward-declared like UploadManager (the
// concrete HotReloadWatcher lives in HotReload.hpp, which needs the full
// Pipeline/ShaderModule/Image/UploadManager definitions, so only Context.cpp
// may see it). Created in the Context binding when hot_reload=True.
//
// Its watch_* register a resource for watching (called from the Python
// bindings on the main thread). drain() applies whatever changed since the
// last call and is MAIN THREAD ONLY — it recompiles shaders (the includer is
// unlocked) and calls vkCreate*. The frame path and the headless submit path
// both drain it.
class HotReloadWatcher;

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
        const ContextConfig& config = {});

    // Defined in Context.cpp: upload_manager_ and hot_reload_ are incomplete
    // types here, and destroying them needs the concrete classes.
    ~Context();

    // Deterministic teardown of everything this Context owns EXCLUSIVELY: the
    // hot-reload watcher, the upload worker, and the GPU work still in flight.
    // Idempotent, and `with bz.Context() as ctx:` is a call to it at the end of
    // the block.
    //
    // What it deliberately does NOT do is destroy the device. The first version
    // did, and the validation-as-assert referee refused it in one line —
    // "vkDestroyDevice(): VkDevice has 3 leaked objects" — because a resource
    // the user still holds a name for is a live child of that device. There are
    // only two ways to satisfy the spec there: free the resource out from under
    // a live Python reference, or refuse to close while any name is bound. The
    // first is a use-after-free with extra steps and the second makes `with`
    // useless in a notebook, which is the surface this feature exists for. So
    // the device follows shared_ptr: it goes when the last resource built from
    // it is dropped, which is what the destructor above is.
    //
    // That leaves an honest contract rather than a smaller one. The threads are
    // what actually accumulate across notebook cell re-runs — a second decoder
    // and a second watcher per run, each with a queue — and they go now. The
    // memory goes when you stop holding it, which is the only point at which
    // freeing it could ever have been correct.
    void close();

    // True once close() has run. Cached values (frames_in_flight, device_name,
    // headless) still answer; anything that would start new GPU work is refused
    // at the binding layer with StateError. Some of it could still physically
    // run — the device outlives close() — and refusing anyway is the point:
    // "closed" that means closed for some verbs and not others is two contracts.
    bool closed() const
    {
        return closed_;
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
    std::uint32_t max_samples() const;

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
    void begin_frame();

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

    std::uint64_t completed_submit_serial() const;

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
    void note_slot_submit(std::uint64_t serial);

    // Blocks until the submit that last used the current ring slot has finished.
    // Cheap when the slot is free: a timeline wait on a value already reached
    // returns immediately, and 0 is always reached.
    void wait_for_slot();

    // Blocks until everything this Context started has finished — the uploads
    // still decoding on the worker as well as every submit — then reclaims what
    // the deletion queue was holding for them.
    //
    // This is the one wait verb. A caller who wants less waits on the resource
    // (Buffer::wait, Image::wait); there is nothing that wants more, because
    // vkDeviceWaitIdle would also stall the other Contexts sharing the device.
    std::expected<void, Error> wait_for_submits();

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
    std::expected<std::uint64_t, Error> submit_one_shot(VkCommandBuffer cmd, std::uint64_t after = 0);

    // The one place that blocks on the submission timeline. Everything that
    // waits for GPU work — a frame's ring slot, an image upload, a readback —
    // comes through here, so a wait is never wider than the work it waits for.
    std::expected<void, Error> wait_for_serial(std::uint64_t serial);

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
    void defer_destroy(std::function<void()> fn);

    void flush_deletion_queue();

    // ── Async uploads ─────────────────────────────────────────────────────────

    UploadManager* upload_manager() const
    {
        return upload_manager_.get();
    }
    // Out of line: assigning the unique_ptr destroys any previous value, which
    // needs the complete class.
    void set_upload_manager(std::unique_ptr<UploadManager> manager);

    // An upload that did NOT go through the worker: create_buffer and
    // create_image(array) hand over bytes that are already decoded, so they
    // submit on the calling thread and only skip the wait. It still counts as
    // an upload, so it joins the worker's batch instead of being tracked
    // beside it.
    void note_upload_serial(std::uint64_t serial);

    // ── Hot reload ────────────────────────────────────────────────────────────
    //
    // Null unless the Context was created with hot_reload=True. The frame path
    // (Context::begin_frame) and the headless submit both drain it on the main
    // thread.
    HotReloadWatcher* hot_reload() const
    {
        return hot_reload_.get();
    }
    void set_hot_reload(std::unique_ptr<HotReloadWatcher> watcher);

    // ── Sampler cache ─────────────────────────────────────────────────────────
    //
    // Identical descriptions share one VkSampler; Texture used to create a
    // fresh sampler per texture. Cached handles live until ~Context — the
    // descriptor space is a handful of combinations, never worth evicting.
    std::expected<std::shared_ptr<Sampler>, Error> get_sampler(const SamplerDesc& desc, const std::string& name = {});

    // ── Memory and device introspection ───────────────────────────────────────

    // How much GPU memory this Context has allocated, and how much the driver
    // says is left, in bytes. "Am I leaking, and how much room is there" is a
    // question a prototype asks constantly, and the answer used to need an
    // external tool.
    //
    // VMA already keeps the numbers, so this is a read rather than new
    // bookkeeping.
    //
    // Summed over the DEVICE_LOCAL heaps only, since 0.26. It used to sum every
    // heap, which on a laptop includes the system memory the GPU may spill
    // into — so an 8 GiB card reported a budget of 18.9 GiB, and code that
    // sized a load against it filled VRAM and then crawled over PCIe. Whichever
    // heap a given allocation lands in is a driver decision, but "how much fits
    // on the GPU" is not a question about the host's RAM, and that is the only
    // question these three numbers are asked.
    struct MemoryStats
    {
        std::uint64_t used = 0;     // bytes VMA has allocated in device-local heaps
        std::uint64_t reserved = 0; // bytes VMA has reserved from those heaps
        std::uint64_t budget = 0;   // bytes of those heaps the driver says the process may use
    };

    MemoryStats memory_stats() const;

    // The subgroup width this GPU runs shaders at, or 0 where the driver does not
    // report one.
    //
    // A compute shader doing a subgroupAdd reduction has to size its workgroup
    // against this number, and there was no way to ask. Vulkan 1.1 core, so no
    // negotiation and no Feature: the property either has a value or it does not.
    //
    // Reads the cache since 0.26 rather than querying per call. The RANGE a
    // driver may be pinned to lives on limits() instead of here, so each number
    // still has one spelling.
    std::uint32_t subgroup_size() const
    {
        return limits_.subgroup_size;
    }

    // The numbers this device holds a caller to (0.26). Queried once at
    // creation, because a property does not change under a live device.
    const DeviceLimits& limits() const
    {
        return limits_;
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
    void set_debug_name(VkObjectType type, std::uint64_t handle, const std::string& name);

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
    //
    // The bodies — and the rationale that goes with them — live in Context.cpp.

    // volk + instance (with validation and the headless fallback). Returns the
    // negotiated instance API version.
    static std::expected<std::uint32_t, Error> create_instance_(
        Context& ctx,
        const ContextConfig& config,
        const std::shared_ptr<Logger>& logger);

    static std::expected<void, Error> select_physical_device_(Context& ctx, const ContextConfig& config);

    static void enable_extension_for(Context& ctx, Feature feature);

    static std::expected<void, Error> configure_features_(
        Context& ctx,
        const ContextConfig& config,
        const std::shared_ptr<Logger>& logger,
        std::uint32_t target_api);

    static std::expected<void, Error> create_device_(Context& ctx);

    static std::expected<void, Error> create_allocator_and_pool_(Context& ctx);

    // Points the core dynamic-rendering names at the KHR implementations on the
    // 1.2 path — see the definition.
    void alias_dynamic_rendering_entry_points();

    static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
        VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
        VkDebugUtilsMessageTypeFlagsEXT message_type,
        const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
        void* user_data);

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
    // Filled beside the feature query and never touched again: a device's
    // limits are fixed for its lifetime (0.26).
    DeviceLimits limits_;
    bool headless_ = false;
    std::atomic<bool> closed_ = false;
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
    std::unique_ptr<UploadManager> upload_manager_;
    std::unique_ptr<HotReloadWatcher> hot_reload_;
};
