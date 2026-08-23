#pragma once
#include <volk.h>
#include <VkBootstrap.h>
#include <vk_mem_alloc.h>

#include <expected>
#include <format>
#include <memory>
#include <span>
#include <mutex>
#include <optional>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>

#include "Context.hpp"
#include "RenderTarget.hpp"
#include "SurfaceProvider.hpp"

class Graph;

struct SwapchainSupportDetails
{
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> present_modes;
};

SwapchainSupportDetails query_swapchain_support(VkPhysicalDevice device, VkSurfaceKHR surface);

VkSurfaceFormatKHR choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& availableFormats);

// FIFO is the only mode the spec guarantees; the others are preferences that
// fall back to FIFO (with an Info log) when the surface can't do them. An enum
// rather than a vsync bool: a bool cannot spell IMMEDIATE, and a second knob
// added later would be two ways to say one thing.
enum class PresentMode
{
    FIFO,         // vsync — capped to refresh rate
    MAILBOX,      // uncapped, no tearing (the default preference)
    IMMEDIATE,    // uncapped, tearing possible; for measurements
    FIFO_RELAXED, // vsync, but a LATE frame presents immediately (tearing once)
};

inline constexpr VkPresentModeKHR to_vk(PresentMode mode)
{
    switch (mode)
    {
        case PresentMode::FIFO:
            return VK_PRESENT_MODE_FIFO_KHR;
        case PresentMode::MAILBOX:
            return VK_PRESENT_MODE_MAILBOX_KHR;
        case PresentMode::IMMEDIATE:
            return VK_PRESENT_MODE_IMMEDIATE_KHR;
        case PresentMode::FIFO_RELAXED:
            return VK_PRESENT_MODE_FIFO_RELAXED_KHR;
    }
    // Not std::unreachable(): pybind enums accept arbitrary ints.
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkPresentModeKHR choose_swap_present_mode(
    const std::vector<VkPresentModeKHR>& availablePresentModes,
    PresentMode preferred);

VkExtent2D choose_swap_extent(const VkSurfaceCapabilitiesKHR& capabilities, int width, int height);

class SwapchainRenderer : public RenderTarget
{
public:
    static std::expected<std::unique_ptr<SwapchainRenderer>, Error> create(
        const std::shared_ptr<Context>& context,
        SurfaceProvider surface_provider,
        PresentMode present_mode = PresentMode::MAILBOX,
        std::uint32_t samples = 1,
        bool stencil = false);

    ~SwapchainRenderer();

    SwapchainRenderer(const SwapchainRenderer&) = delete;
    SwapchainRenderer& operator=(const SwapchainRenderer&) = delete;

    const Context* owner() const override
    {
        return context_.get();
    }

    std::shared_ptr<Context> context() const
    {
        return context_;
    }
    VkSwapchainKHR swapchain() const
    {
        return swapchain_;
    }

    VkImage depth_image() const override
    {
        return depth_image_;
    }

    // ── RenderTarget ──────────────────────────────────────────────────────────
    //
    // A swapchain hands out a different image each frame, which is exactly why
    // these are resolved at replay time rather than baked in when recording.

    std::uint32_t color_count() const override
    {
        return 1;
    }
    // With MSAA the multisampled image is rendered into and the acquired swapchain
    // image is its resolve target; without it, the swapchain image is drawn into
    // directly (msaa_color_image_ is null).
    VkImage color_image(std::uint32_t) const override
    {
        return msaa_color_image_ ? msaa_color_image_ : swapchain_images_[image_index_];
    }
    VkImageView color_view(std::uint32_t) const override
    {
        return msaa_color_view_ ? msaa_color_view_ : swapchain_image_views_[image_index_];
    }
    VkFormat color_format(std::uint32_t) const override
    {
        return swapchain_format_;
    }
    VkSampleCountFlagBits samples() const override
    {
        return samples_;
    }
    VkImage color_resolve_image(std::uint32_t) const override
    {
        return msaa_color_image_ ? swapchain_images_[image_index_] : VK_NULL_HANDLE;
    }
    VkImageView color_resolve_view(std::uint32_t) const override
    {
        return msaa_color_image_ ? swapchain_image_views_[image_index_] : VK_NULL_HANDLE;
    }
    VkImageView depth_view() const override
    {
        return depth_image_view_;
    }
    VkFormat depth_format() const override
    {
        return depth_format_;
    }
    VkExtent2D extent() const override
    {
        return swapchain_extent_;
    }

    // The one line that used to be hardcoded inside every end_rendering.
    VkImageLayout final_layout() const override
    {
        return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    }

    std::uint32_t current_frame() const
    {
        return context_->frame_index();
    }

    // The mode actually in use (post-fallback), not the requested preference.
    PresentMode present_mode() const;

    // Switch vsync at runtime. The present mode is a swapchain creation
    // parameter, so this is the preference plus the recreation that already
    // exists for a resize. Read present_mode() back afterwards: the request is
    // a preference, and a driver that lacks the mode falls back to FIFO.
    std::expected<void, Error> set_present_mode(PresentMode mode);

    // Take the display outright instead of drawing through the compositor
    // (0.25). What it buys is latency and the right to change the display mode;
    // what it costs is that alt-tab becomes a mode switch.
    //
    // A property of the SWAPCHAIN, not a fifth WindowMode: the window is already
    // fullscreen or this does nothing, and the enum stayed out of it for that
    // reason since 0.16. Needs Feature.EXCLUSIVE_FULLSCREEN, which is Win32-only
    // in practice.
    //
    // Refused between acquire() and present() for the reason set_present_mode is:
    // it recreates the swapchain, and an acquired image would be freed under the
    // frame that holds it.
    std::expected<void, Error> set_fullscreen_exclusive(bool enable);

    // Whether the display was actually taken. Not the same question as what was
    // asked for: a driver may refuse, and that is a normal outcome.
    bool fullscreen_exclusive() const
    {
        return exclusive_active_;
    }

    // ── GPU timing ────────────────────────────────────────────────────────────
    //
    // The GPU duration of the frame submitted `frames_in_flight` frames ago (a
    // timestamp pair around each submit, read back once its fence is signalled).
    // None for the first frames_in_flight frames, and on devices without
    // timestamp support. Windowed only: the headless submit is a blocking
    // wait-idle, where wall-clock time already is the GPU time.
    std::optional<double> gpu_time_ms() const
    {
        return last_gpu_time_ms_;
    }

    bool timestamps_supported() const
    {
        return timestamp_pool_ != VK_NULL_HANDLE;
    }

    // Why gpu_time_ms has no number, when it has none. Same three-way split as
    // cmd.timer(), for the same reason: None used to mean "off", "this GPU
    // cannot" and "not measured yet" at once, and only the last one is worth
    // waiting through.
    //
    // Ok here means "a measurement is available or will be" — the value itself
    // may still be absent for the first frames_in_flight frames, which is the
    // NotReady the caller sees as None.
    QueryStatus timing_status() const;

    VkQueryPool timestamp_pool() const
    {
        return timestamp_pool_;
    }
    // submit() calls this after recording the timestamp pair for current_frame(),
    // so acquire() knows the slot has results to read next time round.
    void mark_timestamp_written()
    {
        slot_written_[current_frame()] = true;
    }

    // Fails rather than raises: the caller reached here with the GIL released,
    // and the recording it drives can fail on a lost device.
    std::expected<void, Error> present(
        std::shared_ptr<Graph> graph,
        const QueueSerials& upload_wait = {},
        bool capture = false);

    // What the last present signalled, per queue. The caller hands it back to
    // the graph so its next replay waits for itself.
    QueueSerials last_signalled() const
    {
        return last_signalled_;
    }

    // ── Readback ─────────────────────────────────────────────────────────────
    //
    // A screenshot of a window. Only offscreen targets could be read back before
    // 0.18, so a windowed prototype could not save the picture it was drawing —
    // the one thing a prototype exists to do.
    //
    // It takes TWO calls, and that is not an oversight. A presentable image may
    // only be touched between vkAcquireNextImageKHR and vkQueuePresentKHR, so
    // "read the last frame" is illegal by the spec: after present the
    // compositor owns the image, and the validation layer says so. The copy
    // therefore rides the frame's OWN submit — present(capture=True) records it
    // while the image is still ours — and read_pixels() collects the result
    // afterwards. The frame that captures pays for a copy; every other frame
    // pays nothing.

    // Records the copy into the frame's own command buffer, from record_frame and
    // nowhere else: that is the one point between acquire and present.
    void record_capture(VkCommandBuffer cmd);

    // The captured frame as RGBA8 bytes. Blocks until that frame's submit has
    // completed, because the copy is part of it.
    std::expected<std::vector<std::byte>, Error> read_pixels();

    VkExtent2D capture_extent() const
    {
        return capture_extent_;
    }

    // Take the next swapchain image for THIS window, within the frame the
    // Context already opened. True when an image is ready to render into;
    // false when this window sits the frame out (minimized, mid-resize) —
    // which with N windows must not stop the others, hence a per-window
    // answer rather than a per-frame one.
    std::expected<bool, Error> acquire();

    // present() may only consume an image acquire() actually handed over, and
    // only once — the semaphores and the in-flight fence are per (window, slot).
    std::expected<void, Error> check_presentable() const;

    // upload_wait: per queue, the highest submission-timeline value this
    // frame's resources depend on (async uploads). 0 waits for nothing — a
    // timeline wait for 0 is trivially satisfied, so no branching is needed.
    // previous_replay: what this graph's own previous submit left running on
    // each queue, so a batch waits for the other queues' half of it.
    // Returns what this submit signalled, valid even when it failed halfway.
    QueueSerials end_frame(
        std::span<const Context::SubmitBatch> batches,
        const QueueSerials& upload_wait,
        const QueueSerials& previous_replay);

private:
    // Gives up an acquired frame that will never be submitted, and puts the slot
    // back where end_frame leaves it.
    //
    // acquire() resets this slot's in-flight fence, and only a submit signals it
    // again. So a frame that is acquired and then abandoned leaves a fence that
    // never signals, and the next acquire() on this slot waits on it with no
    // timeout — a hang, several frames after the call that actually failed. An
    // empty submit signals the fence, and it consumes the acquire semaphore as
    // well, which vkAcquireNextImageKHR needs unsignalled the next time round.
    //
    // Then the swapchain goes: Vulkan releases an acquired image when it is
    // presented or when the swapchain is destroyed, and this frame does neither.
    // If the empty submit fails too, the device is out of memory at a depth
    // nothing here can recover from.
    // wait_acquire: whether the acquire semaphore is still unconsumed. A submit
    // that succeeded already waited it, and waiting a binary semaphore twice
    // is a deadlock rather than an error.
    //
    // signal_fence: whether the slot's fence still needs signalling. A partial
    // submit failure can leave it already handed to a batch that WAS accepted,
    // and signalling one fence from two submits is a validation error — the
    // accepted batch will signal it, so the next acquire cannot hang either
    // way.
    void abandon_frame_(bool wait_acquire = true, bool signal_fence = true);

    // image_acquired_ with the Context's counter kept in step — the counter is
    // what refuses a headless ctx.submit() while a window holds an image.
    void set_acquired_(bool acquired);

    SwapchainRenderer(std::shared_ptr<Context> context, SurfaceProvider surface_provider);

    QueueSerials last_signalled_{};

    std::shared_ptr<Context> context_;
    SurfaceProvider surface_provider_;
    // What the caller asked for, and what the swapchain actually got. The two
    // differ when the driver refuses, which is a normal outcome rather than an
    // error: another application can hold the display.
    bool fullscreen_exclusive_ = false;
    bool exclusive_active_ = false;

    // The frame serial this window last acquired at, and whether that acquire
    // produced an image still waiting to be presented. Together they are the
    // whole of "did the caller drive this window correctly this frame".
    std::uint64_t acquired_serial_ = 0;
    bool image_acquired_ = false;

    // The capture staging buffer and which frame slot filled it. Allocated on
    // the first present(capture=True) and reallocated when the window resizes.
    VkBuffer capture_buffer_ = VK_NULL_HANDLE;
    VmaAllocation capture_alloc_ = VK_NULL_HANDLE;
    VkDeviceSize capture_size_ = 0;
    VkExtent2D capture_extent_{};
    std::uint32_t capture_slot_ = 0;
    bool captured_ = false;
    // Whether the surface let the swapchain carry TRANSFER_SRC. A compositor may
    // refuse, and a swapchain that fails to create would take the window down,
    // so the capability is recorded and read_pixels reports it.
    bool supports_readback_ = false;

    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat swapchain_format_ = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchain_extent_{};

    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkQueue present_queue_ = VK_NULL_HANDLE;

    std::vector<VkImage> swapchain_images_;
    std::vector<VkImageView> swapchain_image_views_;

    std::vector<VkSemaphore> image_available_semaphores_;
    std::vector<VkSemaphore> render_finished_semaphores_;
    std::vector<VkFence> in_flight_fences_;

    std::uint32_t image_index_ = 0;

    VkImage depth_image_ = VK_NULL_HANDLE;
    VmaAllocation depth_image_allocation_ = VK_NULL_HANDLE;
    VkImageView depth_image_view_ = VK_NULL_HANDLE;
    VkFormat depth_format_ = VK_FORMAT_UNDEFINED;

    // MSAA: a presentable image is always single-sample, so windowed MSAA renders
    // into this multisampled colour image and resolves into the swapchain image.
    // The depth image above simply becomes multisampled too (samples_). One shared
    // image, like depth — sized to the swapchain, recreated with it.
    VkSampleCountFlagBits samples_ = VK_SAMPLE_COUNT_1_BIT;
    VkImage msaa_color_image_ = VK_NULL_HANDLE;
    VmaAllocation msaa_color_allocation_ = VK_NULL_HANDLE;
    VkImageView msaa_color_view_ = VK_NULL_HANDLE;

    bool frame_skipped_ = false;

    PresentMode preferred_present_mode_ = PresentMode::MAILBOX;
    VkPresentModeKHR active_present_mode_ = VK_PRESENT_MODE_FIFO_KHR;

    // GPU timing: two timestamp queries per in-flight frame (start/end).
    // timestamp_pool_ stays null on devices without support → gpu_time_ms is
    // None. slot_written_ gates reads so a never-submitted slot is not queried.
    VkQueryPool timestamp_pool_ = VK_NULL_HANDLE;
    float timestamp_period_ = 0.0f;
    std::uint32_t timestamp_valid_bits_ = 0;
    std::vector<bool> slot_written_;
    std::optional<double> last_gpu_time_ms_;

    // Best-effort: any missing capability leaves timestamp_pool_ null.
    void create_timestamp_pool_();

    // After the fence wait: read this slot's previous timestamp pair with no
    // WAIT_BIT (the fence already proved completion). Unwritten slot or a
    // not-ready result → None.
    void read_timestamps_();

    std::expected<void, Error> create_swapchain_manually(
        int width,
        int height,
        VkSwapchainKHR old_swapchain = VK_NULL_HANDLE);

    void recreate_swapchain();

    // Depth image + view sized to the current swapchain extent. Shared by first
    // creation and every recreate — it used to be ~50 lines duplicated verbatim
    // between the two.
    std::expected<void, Error> create_depth_resources();

    // Tear down the MSAA colour image + view (destructor and every swapchain
    // recreate). No-op when not multisampled.
    void destroy_msaa_color_();
};
