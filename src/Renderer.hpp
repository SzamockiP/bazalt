#pragma once
#include <volk.h>
#include <VkBootstrap.h>
#include <vk_mem_alloc.h>

#include <expected>
#include <format>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>

#include "Context.hpp"
#include "RenderTarget.hpp"
#include "SurfaceProvider.hpp"

class CommandBuffer;

struct SwapchainSupportDetails
{
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> present_modes;
};

inline SwapchainSupportDetails query_swapchain_support(VkPhysicalDevice device, VkSurfaceKHR surface)
{
    SwapchainSupportDetails details;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
    if (formatCount != 0)
    {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
    if (presentModeCount != 0)
    {
        details.present_modes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.present_modes.data());
    }
    return details;
}

inline VkSurfaceFormatKHR choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& availableFormats)
{
    auto it = std::ranges::find_if(
        availableFormats,
        [](const VkSurfaceFormatKHR& f)
        { return f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR; });
    return it != availableFormats.end() ? *it : availableFormats[0];
}

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

inline VkPresentModeKHR choose_swap_present_mode(
    const std::vector<VkPresentModeKHR>& availablePresentModes,
    PresentMode preferred)
{
    const VkPresentModeKHR wanted = to_vk(preferred);
    return std::ranges::contains(availablePresentModes, wanted) ? wanted : VK_PRESENT_MODE_FIFO_KHR;
}

inline VkExtent2D choose_swap_extent(const VkSurfaceCapabilitiesKHR& capabilities, int width, int height)
{
    if (capabilities.currentExtent.width != (std::numeric_limits<uint32_t>::max)())
    {
        return capabilities.currentExtent;
    }
    else
    {
        VkExtent2D actualExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
        actualExtent.width = (std::max)(capabilities.minImageExtent.width,
                                        (std::min)(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = (std::max)(capabilities.minImageExtent.height,
                                         (std::min)(capabilities.maxImageExtent.height, actualExtent.height));
        return actualExtent;
    }
}

class SwapchainRenderer : public RenderTarget
{
public:
    static std::expected<std::unique_ptr<SwapchainRenderer>, Error> create(
        std::shared_ptr<Context> context,
        SurfaceProvider surface_provider,
        PresentMode present_mode = PresentMode::MAILBOX,
        std::uint32_t samples = 1,
        bool stencil = false)
    {
        if (!context->swapchain_supported())
        {
            return std::unexpected(err_window(
                "Vulkan: this Context has no swapchain support, so it cannot present. "
                "Render to a bazalt.RenderTarget instead, or check ctx.headless."));
        }
        auto vk_samples = validate_sample_count(samples, *context);
        if (!vk_samples)
        {
            return std::unexpected(vk_samples.error());
        }

        // 0.14 lifted the one-renderer-per-Context restriction: everything below
        // is per-renderer already (swapchain, surface, semaphores, fences, depth,
        // MSAA colour, timestamp pool), and the one genuinely shared thing — the
        // frame ring — moved to ctx.begin_frame(), so N windows advance it once.
        auto renderer = std::unique_ptr<SwapchainRenderer>(new SwapchainRenderer(context, std::move(surface_provider)));
        // The PREFERENCE is stored, not the resolved mode: swapchain recreation
        // re-negotiates, because availability can change with the surface.
        renderer->preferred_present_mode_ = present_mode;
        renderer->samples_ = *vk_samples;

        // Surface — created via the SurfaceProvider callback
        VkSurfaceKHR surface = renderer->surface_provider_.create_surface(context->instance());
        if (surface == VK_NULL_HANDLE)
        {
            // Window rather than Initialization: this fails when the window/HWND is
            // unusable, which the caller can fix without rebuilding the Context.
            return std::unexpected(err_window("Vulkan: Failed to create window surface"));
        }
        renderer->surface_ = surface;

        // Verify present support on the graphics queue family
        VkBool32 presentSupport = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(
            context->physical_device(), context->graphics_queue_family(), surface, &presentSupport);
        if (!presentSupport)
        {
            return std::unexpected(err_init("Vulkan: Graphics queue does not support present to this surface"));
        }
        renderer->present_queue_ = context->graphics_queue();

        // Swapchain
        auto [width, height] = renderer->surface_provider_.get_framebuffer_size();
        if (auto r = renderer->create_swapchain_manually(width, height); !r)
        {
            // Propagate the real Error: this used to collapse to a bare
            // "Failed to create swapchain", discarding the VkResult that says why.
            return std::unexpected(r.error());
        }

        // Sync Objects
        VkSemaphoreCreateInfo semaphoreInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = nullptr, .flags = 0};

        VkFenceCreateInfo fenceInfo{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .pNext = nullptr, .flags = VK_FENCE_CREATE_SIGNALED_BIT};

        renderer->image_available_semaphores_.resize(context->frames_in_flight(), VK_NULL_HANDLE);
        renderer->in_flight_fences_.resize(context->frames_in_flight(), VK_NULL_HANDLE);
        for (size_t i = 0; i < context->frames_in_flight(); i++)
        {
            if (auto e = check(
                    context->vk().vkCreateSemaphore(
                        context->device(), &semaphoreInfo, nullptr, &renderer->image_available_semaphores_[i]),
                    "create image available semaphore"))
            {
                return std::unexpected(*e);
            }
            if (auto e = check(
                    context->vk().vkCreateFence(
                        context->device(), &fenceInfo, nullptr, &renderer->in_flight_fences_[i]),
                    "create in flight fence"))
            {
                return std::unexpected(*e);
            }
        }

        renderer->render_finished_semaphores_.resize(renderer->swapchain_images_.size());
        for (size_t i = 0; i < renderer->swapchain_images_.size(); i++)
        {
            if (auto e = check(
                    context->vk().vkCreateSemaphore(
                        context->device(), &semaphoreInfo, nullptr, &renderer->render_finished_semaphores_[i]),
                    "create render finished semaphore"))
            {
                return std::unexpected(*e);
            }
        }

        // Depth Image. The window's depth buffer is scratch either way; asking
        // for a stencil aspect is what makes a masked pass (an outline, a
        // portal) possible on screen rather than only offscreen.
        renderer->depth_format_ = stencil ? context->depth_stencil_format() : VK_FORMAT_D32_SFLOAT;
        if (renderer->depth_format_ == VK_FORMAT_UNDEFINED)
        {
            return std::unexpected(err_init("This device supports no depth/stencil format"));
        }
        if (auto r = renderer->create_depth_resources(); !r)
        {
            return std::unexpected(r.error());
        }

        // Best-effort: a device without timestamp support just reports
        // gpu_time_ms as None, never an error.
        renderer->create_timestamp_pool_();

        return renderer;
    }

    ~SwapchainRenderer()
    {
        if (!context_)
        {
            return;
        }

        if (context_->device())
        {
            std::lock_guard lock(context_->queue_mutex());
            context_->vk().vkDeviceWaitIdle(context_->device());
        }

        for (size_t i = 0; i < image_available_semaphores_.size(); ++i)
        {
            if (image_available_semaphores_[i])
                context_->vk().vkDestroySemaphore(context_->device(), image_available_semaphores_[i], nullptr);
            if (in_flight_fences_[i])
                context_->vk().vkDestroyFence(context_->device(), in_flight_fences_[i], nullptr);
        }

        for (auto sem : render_finished_semaphores_)
        {
            if (sem)
                context_->vk().vkDestroySemaphore(context_->device(), sem, nullptr);
        }

        if (timestamp_pool_)
        {
            context_->vk().vkDestroyQueryPool(context_->device(), timestamp_pool_, nullptr);
        }

        // Direct, not deferred: the wait-idle above proves nothing is reading it.
        if (capture_buffer_ != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(context_->allocator(), capture_buffer_, capture_alloc_);
        }

        if (depth_image_view_)
        {
            context_->vk().vkDestroyImageView(context_->device(), depth_image_view_, nullptr);
        }
        if (depth_image_ && depth_image_allocation_)
        {
            vmaDestroyImage(context_->allocator(), depth_image_, depth_image_allocation_);
        }
        destroy_msaa_color_();

        for (auto iv : swapchain_image_views_)
        {
            context_->vk().vkDestroyImageView(context_->device(), iv, nullptr);
        }

        if (swapchain_)
        {
            context_->vk().vkDestroySwapchainKHR(context_->device(), swapchain_, nullptr);
        }

        if (surface_)
        {
            vkDestroySurfaceKHR(context_->instance(), surface_, nullptr);
        }
    }

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
    VkFormat swapchain_format() const
    {
        return swapchain_format_;
    }
    VkExtent2D swapchain_extent() const
    {
        return swapchain_extent_;
    }
    const std::vector<VkImage>& swapchain_images() const
    {
        return swapchain_images_;
    }
    const std::vector<VkImageView>& swapchain_image_views() const
    {
        return swapchain_image_views_;
    }

    VkImage depth_image() const override
    {
        return depth_image_;
    }
    VkImageView depth_image_view() const
    {
        return depth_image_view_;
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
    std::uint64_t current_serial() const
    {
        return context_->frame_serial();
    }
    std::uint32_t current_image_index() const
    {
        return image_index_;
    }
    bool frame_skipped() const
    {
        return frame_skipped_;
    }

    // The mode actually in use (post-fallback), not the requested preference.
    PresentMode present_mode() const
    {
        switch (active_present_mode_)
        {
            case VK_PRESENT_MODE_MAILBOX_KHR:
                return PresentMode::MAILBOX;
            case VK_PRESENT_MODE_IMMEDIATE_KHR:
                return PresentMode::IMMEDIATE;
            case VK_PRESENT_MODE_FIFO_RELAXED_KHR:
                return PresentMode::FIFO_RELAXED;
            default:
                return PresentMode::FIFO;
        }
    }

    // Switch vsync at runtime. The present mode is a swapchain creation
    // parameter, so this is the preference plus the recreation that already
    // exists for a resize. Read present_mode() back afterwards: the request is
    // a preference, and a driver that lacks the mode falls back to FIFO.
    std::expected<void, Error> set_present_mode(PresentMode mode)
    {
        if (mode == preferred_present_mode_)
        {
            return {};
        }
        // Recreation destroys the swapchain an acquired image belongs to, so
        // between acquire() and present() this is a use-after-free waiting for a
        // driver to notice. Same shape as the double-acquire guard: name the
        // mistake instead of letting validation describe the symptom.
        if (image_acquired_)
        {
            return std::unexpected(err_state(
                "set_present_mode() recreates the swapchain, so it cannot run between "
                "acquire() and present(). Call it before ctx.begin_frame(), or after present()."));
        }
        preferred_present_mode_ = mode;
        recreate_swapchain();
        return {};
    }

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
    std::expected<void, Error> set_fullscreen_exclusive(bool enable)
    {
        if (enable && !context_->supports(Feature::EXCLUSIVE_FULLSCREEN))
        {
            return std::unexpected(err_unsupported(
                "exclusive fullscreen needs Feature.EXCLUSIVE_FULLSCREEN, which this driver does "
                "not offer (VK_EXT_full_screen_exclusive is a Windows extension). Ask "
                "ctx.supports(bz.Feature.EXCLUSIVE_FULLSCREEN), and use "
                "window.set_mode(bz.WindowMode.FULLSCREEN) where it answers False."));
        }
        if (image_acquired_)
        {
            return std::unexpected(err_state(
                "set_fullscreen_exclusive() recreates the swapchain, so it cannot run between "
                "acquire() and present(). Call it before ctx.begin_frame(), or after present()."));
        }
        if (enable == fullscreen_exclusive_)
        {
            return {};
        }
        // Released before the swapchain goes, not after: the mode belongs to the
        // swapchain that holds it, and releasing a destroyed one is nothing.
        //
        // Guarded by the platform macro rather than by a runtime check, because
        // VolkDeviceTable does not even DECLARE these members off Win32 — the
        // whole extension lives behind VK_USE_PLATFORM_WIN32_KHR in the Vulkan
        // headers. The runtime check stays inside it: the member can exist and
        // still be null.
#ifdef VK_USE_PLATFORM_WIN32_KHR
        if (!enable && exclusive_active_ && context_->vk().vkReleaseFullScreenExclusiveModeEXT)
        {
            context_->vk().vkReleaseFullScreenExclusiveModeEXT(context_->device(), swapchain_);
            exclusive_active_ = false;
        }
#endif
        fullscreen_exclusive_ = enable;
        recreate_swapchain();
        return {};
    }

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
    QueryStatus timing_status() const
    {
        if (!context_->gpu_timing())
        {
            return QueryStatus::Disabled;
        }
        // The pool is created once, at renderer construction, and stays null only
        // when the device failed the timestampPeriod / timestampValidBits check.
        if (timestamp_pool_ == VK_NULL_HANDLE)
        {
            return QueryStatus::Unsupported;
        }
        return last_gpu_time_ms_.has_value() ? QueryStatus::Ok : QueryStatus::NotReady;
    }
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
        std::shared_ptr<CommandBuffer> cmd,
        std::uint64_t upload_wait_serial = 0,
        bool capture = false);

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
    void record_capture(VkCommandBuffer cmd)
    {
        if (!supports_readback_)
        {
            return;
        }
        const VkExtent2D extent = swapchain_extent_;
        const VkDeviceSize needed = static_cast<VkDeviceSize>(extent.width) * extent.height * 4;
        if (capture_buffer_ == VK_NULL_HANDLE || capture_size_ < needed)
        {
            if (capture_buffer_ != VK_NULL_HANDLE)
            {
                // Deferred: a previous frame's copy may still be in flight.
                context_->defer_destroy(
                    [allocator = context_->allocator(), buffer = capture_buffer_, alloc = capture_alloc_]
                    { vmaDestroyBuffer(allocator, buffer, alloc); });
                capture_buffer_ = VK_NULL_HANDLE;
            }
            auto staging = create_staging_buffer(*context_, needed, Staging::Readback);
            if (!staging)
            {
                capture_buffer_ = VK_NULL_HANDLE;
                return;
            }
            std::tie(capture_buffer_, capture_alloc_) = *staging;
            capture_size_ = needed;
        }

        VkImage source = swapchain_images_[image_index_];
        // end_rendering has already retired it to PRESENT_SRC, and it goes back
        // there: the compositor takes it from that layout a moment later.
        record_image_transition(
            context_->vk(),
            cmd,
            source,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);

        VkBufferImageCopy region{
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageOffset = {0, 0, 0},
            .imageExtent = {extent.width, extent.height, 1}};
        context_->vk().vkCmdCopyImageToBuffer(
            cmd, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, capture_buffer_, 1, &region);

        record_image_transition(
            context_->vk(),
            cmd,
            source,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_ACCESS_TRANSFER_READ_BIT,
            VK_ACCESS_MEMORY_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        capture_extent_ = extent;
        capture_slot_ = current_frame();
        captured_ = true;
    }

    // The captured frame as RGBA8 bytes. Blocks until that frame's submit has
    // completed, because the copy is part of it.
    std::expected<std::vector<std::byte>, Error> read_pixels()
    {
        if (!supports_readback_)
        {
            return std::unexpected(err_unsupported(
                "renderer.read_pixels(): this surface does not allow the swapchain images to be copied "
                "from (the compositor refused VK_IMAGE_USAGE_TRANSFER_SRC_BIT). Render into a "
                "bz.RenderTarget and read that instead."));
        }
        if (!captured_ || capture_buffer_ == VK_NULL_HANDLE)
        {
            return std::unexpected(err_state(
                "renderer.read_pixels(): no frame has been captured. Ask for one with "
                "renderer.present(cmd, capture=True) — a presentable image may only be copied while it "
                "is acquired, so the copy has to ride that frame's own submit."));
        }

        // The copy is part of that frame's submit, so this is where it is waited on.
        context_->vk().vkWaitForFences(context_->device(), 1, &in_flight_fences_[capture_slot_], VK_TRUE, UINT64_MAX);

        const std::size_t size = static_cast<std::size_t>(capture_extent_.width) * capture_extent_.height * 4;
        std::vector<std::byte> out(size);
        void* mapped = nullptr;
        if (auto e = check(
                vmaMapMemory(context_->allocator(), capture_alloc_, &mapped),
                "map the swapchain capture buffer",
                ErrorCode::Resource))
        {
            return std::unexpected(*e);
        }
        std::memcpy(out.data(), mapped, size);
        vmaUnmapMemory(context_->allocator(), capture_alloc_);

        // BGRA is what most compositors hand out. Swapping here keeps the channel
        // order one thing rather than a property of the machine, so out[y, x, 0] is
        // red everywhere.
        if (swapchain_format_ == VK_FORMAT_B8G8R8A8_UNORM || swapchain_format_ == VK_FORMAT_B8G8R8A8_SRGB)
        {
            for (std::size_t i = 0; i + 2 < out.size(); i += 4)
            {
                std::swap(out[i], out[i + 2]);
            }
        }
        return out;
    }

    VkExtent2D capture_extent() const
    {
        return capture_extent_;
    }

    // Take the next swapchain image for THIS window, within the frame the
    // Context already opened. True when an image is ready to render into;
    // false when this window sits the frame out (minimized, mid-resize) —
    // which with N windows must not stop the others, hence a per-window
    // answer rather than a per-frame one.
    std::expected<bool, Error> acquire()
    {
        // Acquiring twice on one ring slot means either a genuine double
        // acquire or — far more likely — a loop that forgot ctx.begin_frame().
        // Both leave this window rendering into a slot whose previous submit
        // may still be in flight, so name them together.
        if (acquired_serial_ == context_->frame_serial())
        {
            return std::unexpected(err_state(
                "This window already acquired an image for the current frame. Call "
                "ctx.begin_frame() once per frame, then acquire() once per window."));
        }
        acquired_serial_ = context_->frame_serial();
        image_acquired_ = false;
        frame_skipped_ = false;

        // Check framebuffer size — return false if minimized (0x0)
        auto [width, height] = surface_provider_.get_framebuffer_size();
        if (width == 0 || height == 0)
        {
            frame_skipped_ = true;
            return false;
        }

        context_->vk().vkWaitForFences(context_->device(), 1, &in_flight_fences_[current_frame()], VK_TRUE, UINT64_MAX);

        // The fence proves this slot's previous submission finished, so its
        // timestamp pair is ready to read (frames_in_flight frames of latency).
        read_timestamps_();

        VkResult result = context_->vk().vkAcquireNextImageKHR(
            context_->device(),
            swapchain_,
            UINT64_MAX,
            image_available_semaphores_[current_frame()],
            VK_NULL_HANDLE,
            &image_index_);

        if (result == VK_ERROR_OUT_OF_DATE_KHR)
        {
            recreate_swapchain();
            frame_skipped_ = true;
            return false;
        }
        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        {
            if (auto l = context_->logger())
                l->log(
                    Severity::Error,
                    Source::Device,
                    std::format("Failed to acquire swapchain image ({})", vk_result_name(result)));
            frame_skipped_ = true;
            return false;
        }

        context_->vk().vkResetFences(context_->device(), 1, &in_flight_fences_[current_frame()]);
        image_acquired_ = true;
        return true;
    }

    // present() may only consume an image acquire() actually handed over, and
    // only once — the semaphores and the in-flight fence are per (window, slot).
    std::expected<void, Error> check_presentable() const
    {
        if (!image_acquired_)
        {
            return std::unexpected(err_state(
                "This window has no acquired swapchain image. Call acquire() first, "
                "and skip present() when it returns False (minimized or resizing)."));
        }
        return {};
    }

    // upload_wait_serial: the highest submission-timeline value this frame's
    // resources depend on (async uploads). 0 waits for nothing — a timeline
    // wait for 0 is trivially satisfied, so no branching is needed.
    void end_frame(VkCommandBuffer cmd, std::uint64_t upload_wait_serial = 0)
    {
        // The image is consumed here; a second present() on it would submit
        // against semaphores this one already signalled.
        image_acquired_ = false;

        VkSemaphore waitSemaphores[] = {image_available_semaphores_[current_frame()], context_->submit_timeline()};
        VkPipelineStageFlags waitStages[] = {
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
        std::uint64_t waitValues[] = {0, upload_wait_serial}; // binary sem value ignored

        VkSemaphore signalSemaphores[] = {render_finished_semaphores_[image_index_], context_->submit_timeline()};
        VkSwapchainKHR swapchains[] = {swapchain_};

        VkPresentInfoKHR presentInfo{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = nullptr,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = signalSemaphores,
            .swapchainCount = 1,
            .pSwapchains = swapchains,
            .pImageIndices = &image_index_,
            .pResults = nullptr};

        // The lock ends before recreate_swapchain below: that path takes the
        // device idle, which must not happen while holding the queue mutex.
        VkResult result = VK_SUCCESS;
        bool submitted = false;
        {
            std::lock_guard lock(context_->queue_mutex());

            // Every submit signals the timeline; the serial is reserved under
            // the same lock that orders the submits.
            std::uint64_t signalValues[] = {0, context_->advance_submit_serial()};

            VkTimelineSemaphoreSubmitInfo timelineInfo{
                .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                .pNext = nullptr,
                .waitSemaphoreValueCount = 2,
                .pWaitSemaphoreValues = waitValues,
                .signalSemaphoreValueCount = 2,
                .pSignalSemaphoreValues = signalValues};

            VkSubmitInfo submitInfo{
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .pNext = &timelineInfo,
                .waitSemaphoreCount = 2,
                .pWaitSemaphores = waitSemaphores,
                .pWaitDstStageMask = waitStages,
                .commandBufferCount = 1,
                .pCommandBuffers = &cmd,
                .signalSemaphoreCount = 2,
                .pSignalSemaphores = signalSemaphores};

            if (VkResult submit_result = context_->vk().vkQueueSubmit(
                    context_->graphics_queue(), 1, &submitInfo, in_flight_fences_[current_frame()]);
                submit_result != VK_SUCCESS)
            {
                if (auto l = context_->logger())
                    l->log(
                        Severity::Error,
                        Source::Device,
                        std::format("Failed to submit draw command buffer ({})", vk_result_name(submit_result)));
            }
            else
            {
                submitted = true;
                result = context_->vk().vkQueuePresentKHR(present_queue_, &presentInfo);
            }
        }

        if (!submitted)
        {
            // A submit that fails signals nothing, so presenting would wait on a
            // render-finished semaphore nobody is going to signal, and the slot's
            // fence is still as acquire() left it. Give the frame back instead.
            //
            // The reserved timeline serial is dropped with it, and that needs no
            // repair: a timeline signal only has to be GREATER than the current
            // value, and every wait is "value >= N", so the next submit's higher
            // signal satisfies anything that was waiting for the skipped one.
            abandon_frame_();
            return;
        }

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ||
            surface_provider_.consume_resize_flag())
        {
            recreate_swapchain();
        }
    }

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
    void abandon_frame_()
    {
        image_acquired_ = false;
        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = nullptr,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &image_available_semaphores_[current_frame()],
            .pWaitDstStageMask = &wait_stage,
            .commandBufferCount = 0,
            .pCommandBuffers = nullptr,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = nullptr};
        {
            // Released before recreate_swapchain: that path takes the device idle,
            // which must not happen while holding the queue mutex.
            std::lock_guard lock(context_->queue_mutex());
            context_->vk().vkQueueSubmit(
                context_->graphics_queue(), 1, &submitInfo, in_flight_fences_[current_frame()]);
        }
        recreate_swapchain();
    }

    SwapchainRenderer(std::shared_ptr<Context> context, SurfaceProvider surface_provider)
        : context_(context),
          surface_provider_(std::move(surface_provider))
    {
    }

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
    void create_timestamp_pool_()
    {
        // Opt-in only. Off (the default), no pool exists, so submit records no
        // timestamps, acquire() reads none, and gpu_time_ms stays None — no
        // per-frame timestamp work at all, which is the point of the default.
        if (!context_->gpu_timing())
        {
            return;
        }

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(context_->physical_device(), &props);
        if (props.limits.timestampPeriod <= 0.0f)
        {
            return;
        }

        std::uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(context_->physical_device(), &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(context_->physical_device(), &family_count, families.data());
        const std::uint32_t graphics_family = context_->graphics_queue_family();
        if (graphics_family >= family_count || families[graphics_family].timestampValidBits == 0)
        {
            return;
        }

        VkQueryPoolCreateInfo poolInfo{
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = 2 * context_->frames_in_flight(),
            .pipelineStatistics = 0};
        if (context_->vk().vkCreateQueryPool(context_->device(), &poolInfo, nullptr, &timestamp_pool_) != VK_SUCCESS)
        {
            timestamp_pool_ = VK_NULL_HANDLE;
            return;
        }
        timestamp_period_ = props.limits.timestampPeriod;
        timestamp_valid_bits_ = families[graphics_family].timestampValidBits;
        slot_written_.assign(context_->frames_in_flight(), false);
    }

    // After the fence wait: read this slot's previous timestamp pair with no
    // WAIT_BIT (the fence already proved completion). Unwritten slot or a
    // not-ready result → None.
    void read_timestamps_()
    {
        const std::uint32_t slot = current_frame();
        if (timestamp_pool_ == VK_NULL_HANDLE || slot >= slot_written_.size() || !slot_written_[slot])
        {
            last_gpu_time_ms_ = std::nullopt;
            return;
        }
        std::uint64_t ts[2] = {0, 0};
        if (context_->vk().vkGetQueryPoolResults(
                context_->device(),
                timestamp_pool_,
                2 * slot,
                2,
                sizeof(ts),
                ts,
                sizeof(std::uint64_t),
                VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
        {
            last_gpu_time_ms_ = std::nullopt;
            return;
        }
        const std::uint64_t mask = timestamp_valid_bits_ >= 64 ? ~std::uint64_t{0}
                                                               : ((std::uint64_t{1} << timestamp_valid_bits_) - 1);
        const std::uint64_t delta = (ts[1] - ts[0]) & mask;
        last_gpu_time_ms_ = static_cast<double>(delta) * static_cast<double>(timestamp_period_) / 1.0e6;
    }

    std::expected<void, Error> create_swapchain_manually(
        int width,
        int height,
        VkSwapchainKHR old_swapchain = VK_NULL_HANDLE)
    {
        auto details = query_swapchain_support(context_->physical_device(), surface_);
        auto surface_format = choose_swap_surface_format(details.formats);
        auto present_mode = choose_swap_present_mode(details.present_modes, preferred_present_mode_);
        if (present_mode != to_vk(preferred_present_mode_))
        {
            if (auto l = context_->logger())
                l->log(
                    Severity::Info,
                    Source::Device,
                    "Requested present mode is not supported by this surface. Bazalt uses FIFO (vsync) instead");
        }
        active_present_mode_ = present_mode;
        auto extent = choose_swap_extent(details.capabilities, width, height);

        uint32_t image_count = details.capabilities.minImageCount + 1;
        if (details.capabilities.maxImageCount > 0 && image_count > details.capabilities.maxImageCount)
        {
            image_count = details.capabilities.maxImageCount;
        }

        // TRANSFER_SRC is what renderer.read_pixels() copies out of, and it is
        // asked for only where the surface allows it: a compositor is entitled
        // to refuse, and a swapchain that fails to create takes the window with
        // it. So the capability is recorded and read_pixels says so instead.
        supports_readback_ = (details.capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0;
        VkImageUsageFlags image_usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (supports_readback_)
        {
            image_usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        }

        // Exclusive fullscreen, when the Context negotiated it and the caller
        // asked. APPLICATION_CONTROLLED rather than ALLOWED: the caller said when
        // to take the display, so the acquire below is theirs to time, and a
        // driver-decided mode would make set_fullscreen_exclusive(True) a
        // suggestion.
        //
        // The whole extension is Win32-only in the Vulkan headers, structs
        // included, so everything about it is compiled out elsewhere. Nothing is
        // lost: Feature::EXCLUSIVE_FULLSCREEN keys on a device extension no other
        // platform reports, so set_fullscreen_exclusive already refuses there.
        const void* swapchain_next = nullptr;
        exclusive_active_ = false;
#ifdef VK_USE_PLATFORM_WIN32_KHR
        // The Win32 struct is not optional beside the first
        // (VUID-VkSwapchainCreateInfoKHR-pNext-02679): a Win32 surface must name
        // the monitor, which is what the SurfaceProvider hands out.
        VkSurfaceFullScreenExclusiveInfoEXT exclusive_info{
            .sType = VK_STRUCTURE_TYPE_SURFACE_FULL_SCREEN_EXCLUSIVE_INFO_EXT,
            .pNext = nullptr,
            .fullScreenExclusive = VK_FULL_SCREEN_EXCLUSIVE_APPLICATION_CONTROLLED_EXT};
        VkSurfaceFullScreenExclusiveWin32InfoEXT exclusive_win32{
            .sType = VK_STRUCTURE_TYPE_SURFACE_FULL_SCREEN_EXCLUSIVE_WIN32_INFO_EXT,
            .pNext = nullptr,
            .hmonitor = nullptr};
        if (fullscreen_exclusive_)
        {
            // The surface has to be able to name its display. A renderer built on
            // a raw HWND has no Window behind it to ask, which is the one case
            // where the Feature is present and the mode still cannot be taken.
            void* monitor = surface_provider_.get_win32_monitor ? surface_provider_.get_win32_monitor() : nullptr;
            if (!monitor)
            {
                if (auto l = context_->logger())
                {
                    l->log(
                        Severity::Warning,
                        Source::Device,
                        "Exclusive fullscreen was asked for, but this surface cannot name the display "
                        "it is on, so the swapchain stays composited.");
                }
            }
            else
            {
                exclusive_win32.hmonitor = static_cast<HMONITOR>(monitor);
                exclusive_info.pNext = &exclusive_win32;
                swapchain_next = &exclusive_info;
                exclusive_active_ = true;
            }
        }
#endif

        VkSwapchainCreateInfoKHR createInfo{
            .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .pNext = swapchain_next,
            .flags = 0,
            .surface = surface_,
            .minImageCount = image_count,
            .imageFormat = surface_format.format,
            .imageColorSpace = surface_format.colorSpace,
            .imageExtent = extent,
            .imageArrayLayers = 1,
            .imageUsage = image_usage,
            .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .preTransform = details.capabilities.currentTransform,
            .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = present_mode,
            .clipped = VK_TRUE,
            .oldSwapchain = old_swapchain};

        // On some systems compositeAlpha might not support OPAQUE, so select the first supported one
        if (!(details.capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR))
        {
            if (details.capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR)
            {
                createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
            }
            else if (details.capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR)
            {
                createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR;
            }
            else if (details.capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR)
            {
                createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR;
            }
        }

        VkSwapchainKHR new_swapchain;
        if (auto e = check(
                context_->vk().vkCreateSwapchainKHR(context_->device(), &createInfo, nullptr, &new_swapchain),
                "create swapchain"))
        {
            return std::unexpected(*e);
        }

        swapchain_ = new_swapchain;
        swapchain_format_ = surface_format.format;
        swapchain_extent_ = extent;

        // The acquire has to follow every swapchain creation, not only the first:
        // a resize or a present-mode change builds a new swapchain, and the mode
        // belongs to the swapchain rather than to the surface. Failure is not
        // fatal — another application may hold the display — so it is logged and
        // the composited swapchain keeps working.
#ifdef VK_USE_PLATFORM_WIN32_KHR
        // The entry point can be null even when the extension is available: volk
        // loads a device function only if the extension was enabled at device
        // creation, and a null call is a crash with no diagnostic — the 0.15
        // lesson about device-level globals, in a place that has one pointer
        // rather than a whole table.
        if (exclusive_active_ && !context_->vk().vkAcquireFullScreenExclusiveModeEXT)
        {
            exclusive_active_ = false;
            if (auto l = context_->logger())
            {
                l->log(
                    Severity::Warning,
                    Source::Device,
                    "VK_EXT_full_screen_exclusive is present but its entry points were not loaded, so "
                    "the swapchain stays composited.");
            }
        }
        if (exclusive_active_)
        {
            const VkResult acquired =
                context_->vk().vkAcquireFullScreenExclusiveModeEXT(context_->device(), swapchain_);
            if (acquired != VK_SUCCESS)
            {
                exclusive_active_ = false;
                if (auto l = context_->logger())
                {
                    l->log(
                        Severity::Warning,
                        Source::Device,
                        std::format(
                            "Exclusive fullscreen was refused ({}), so the swapchain stays composited. "
                            "The window must already be fullscreen on that display, and no other "
                            "application may hold it.",
                            vk_result_name(acquired)));
                }
            }
        }
#endif

        // Retrieve swapchain images
        uint32_t actual_image_count;
        context_->vk().vkGetSwapchainImagesKHR(context_->device(), swapchain_, &actual_image_count, nullptr);
        swapchain_images_.resize(actual_image_count);
        context_->vk().vkGetSwapchainImagesKHR(
            context_->device(), swapchain_, &actual_image_count, swapchain_images_.data());

        // Create swapchain image views
        swapchain_image_views_.resize(actual_image_count);
        for (size_t i = 0; i < actual_image_count; i++)
        {
            VkImageViewCreateInfo viewInfo{
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .image = swapchain_images_[i],
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = swapchain_format_,
                .components =
                    {.r = VK_COMPONENT_SWIZZLE_IDENTITY,
                     .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                     .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                     .a = VK_COMPONENT_SWIZZLE_IDENTITY},
                .subresourceRange = {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1}};

            if (auto e = check(
                    context_->vk().vkCreateImageView(
                        context_->device(), &viewInfo, nullptr, &swapchain_image_views_[i]),
                    "create swapchain image view"))
            {
                return std::unexpected(*e);
            }
        }

        return {};
    }

    void recreate_swapchain()
    {
        {
            std::lock_guard lock(context_->queue_mutex());
            context_->vk().vkDeviceWaitIdle(context_->device());
        }

        // Destroy old depth resources
        if (depth_image_view_)
        {
            context_->vk().vkDestroyImageView(context_->device(), depth_image_view_, nullptr);
            depth_image_view_ = VK_NULL_HANDLE;
        }
        if (depth_image_ && depth_image_allocation_)
        {
            vmaDestroyImage(context_->allocator(), depth_image_, depth_image_allocation_);
            depth_image_ = VK_NULL_HANDLE;
            depth_image_allocation_ = VK_NULL_HANDLE;
        }
        destroy_msaa_color_();

        // Destroy old render-finished semaphores
        for (auto sem : render_finished_semaphores_)
        {
            if (sem)
                context_->vk().vkDestroySemaphore(context_->device(), sem, nullptr);
        }
        render_finished_semaphores_.clear();

        // Destroy old swapchain image views
        for (auto iv : swapchain_image_views_)
        {
            context_->vk().vkDestroyImageView(context_->device(), iv, nullptr);
        }
        swapchain_image_views_.clear();
        swapchain_images_.clear();

        VkSwapchainKHR old_swapchain = swapchain_;
        auto [width, height] = surface_provider_.get_framebuffer_size();
        if (auto r = create_swapchain_manually(width, height, old_swapchain); !r)
        {
            // This runs mid-frame, so it keeps the log-and-bail contract — but it
            // logs the propagated Error (VkResult name included), not a hand-written
            // string.
            if (auto l = context_->logger())
                l->log(Severity::Error, Source::Device, "Failed to recreate swapchain: " + r.error().message);
            return;
        }

        if (old_swapchain != VK_NULL_HANDLE)
        {
            context_->vk().vkDestroySwapchainKHR(context_->device(), old_swapchain, nullptr);
        }

        // Recreate render-finished semaphores (one per swapchain image)
        VkSemaphoreCreateInfo semaphoreInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = nullptr, .flags = 0};
        render_finished_semaphores_.resize(swapchain_images_.size());
        for (size_t i = 0; i < swapchain_images_.size(); i++)
        {
            if (context_->vk().vkCreateSemaphore(
                    context_->device(), &semaphoreInfo, nullptr, &render_finished_semaphores_[i]) != VK_SUCCESS)
            {
                if (auto l = context_->logger())
                    l->log(Severity::Error, Source::Device, "Failed to recreate render finished semaphores");
                return;
            }
        }

        // Recreate depth image
        if (auto r = create_depth_resources(); !r)
        {
            if (auto l = context_->logger())
                l->log(Severity::Error, Source::Device, "Failed to recreate depth resources: " + r.error().message);
            return;
        }

        if (auto l = context_->logger())
            l->log(
                Severity::Info,
                Source::Device,
                std::format("Swapchain recreated ({}x{})", swapchain_extent_.width, swapchain_extent_.height));
    }

    // Depth image + view sized to the current swapchain extent. Shared by first
    // creation and every recreate — it used to be ~50 lines duplicated verbatim
    // between the two.
    std::expected<void, Error> create_depth_resources()
    {
        VkImageCreateInfo imageInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = depth_format_,
            .extent = {swapchain_extent_.width, swapchain_extent_.height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = samples_,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};

        VmaAllocationCreateInfo allocImageInfo = {};
        allocImageInfo.usage = VMA_MEMORY_USAGE_AUTO;

        if (auto e = check(
                vmaCreateImage(
                    context_->allocator(),
                    &imageInfo,
                    &allocImageInfo,
                    &depth_image_,
                    &depth_image_allocation_,
                    nullptr),
                "create depth image"))
        {
            return std::unexpected(*e);
        }

        VkImageViewCreateInfo viewInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = depth_image_,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = depth_format_,
            .components =
                {VK_COMPONENT_SWIZZLE_IDENTITY,
                 VK_COMPONENT_SWIZZLE_IDENTITY,
                 VK_COMPONENT_SWIZZLE_IDENTITY,
                 VK_COMPONENT_SWIZZLE_IDENTITY},
            .subresourceRange = {// Both aspects when the format has both: the view is used for
                                 // the depth AND the stencil attachment of the same pass.
                                 .aspectMask = aspect_mask_for(depth_format_),
                                 .baseMipLevel = 0,
                                 .levelCount = 1,
                                 .baseArrayLayer = 0,
                                 .layerCount = 1}};

        if (auto e = check(
                context_->vk().vkCreateImageView(context_->device(), &viewInfo, nullptr, &depth_image_view_),
                "create depth image view"))
        {
            return std::unexpected(*e);
        }

        // MSAA colour image: same extent/format as the swapchain but multisampled.
        // The swapchain image is single-sample (it must present) and serves as the
        // resolve target instead.
        if (samples_ != VK_SAMPLE_COUNT_1_BIT)
        {
            VkImageCreateInfo colorInfo{
                .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .imageType = VK_IMAGE_TYPE_2D,
                .format = swapchain_format_,
                .extent = {swapchain_extent_.width, swapchain_extent_.height, 1},
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = samples_,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};

            VmaAllocationCreateInfo allocColorInfo = {};
            allocColorInfo.usage = VMA_MEMORY_USAGE_AUTO;

            if (auto e = check(
                    vmaCreateImage(
                        context_->allocator(),
                        &colorInfo,
                        &allocColorInfo,
                        &msaa_color_image_,
                        &msaa_color_allocation_,
                        nullptr),
                    "create MSAA colour image"))
            {
                return std::unexpected(*e);
            }

            VkImageViewCreateInfo colorViewInfo{
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .image = msaa_color_image_,
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = swapchain_format_,
                .components =
                    {VK_COMPONENT_SWIZZLE_IDENTITY,
                     VK_COMPONENT_SWIZZLE_IDENTITY,
                     VK_COMPONENT_SWIZZLE_IDENTITY,
                     VK_COMPONENT_SWIZZLE_IDENTITY},
                .subresourceRange = {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1}};

            if (auto e = check(
                    context_->vk().vkCreateImageView(context_->device(), &colorViewInfo, nullptr, &msaa_color_view_),
                    "create MSAA colour image view"))
            {
                return std::unexpected(*e);
            }
        }

        return {};
    }

    // Tear down the MSAA colour image + view (destructor and every swapchain
    // recreate). No-op when not multisampled.
    void destroy_msaa_color_()
    {
        if (msaa_color_view_)
        {
            context_->vk().vkDestroyImageView(context_->device(), msaa_color_view_, nullptr);
            msaa_color_view_ = VK_NULL_HANDLE;
        }
        if (msaa_color_image_ && msaa_color_allocation_)
        {
            vmaDestroyImage(context_->allocator(), msaa_color_image_, msaa_color_allocation_);
            msaa_color_image_ = VK_NULL_HANDLE;
            msaa_color_allocation_ = VK_NULL_HANDLE;
        }
    }
};
