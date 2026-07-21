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

struct SwapchainSupportDetails {
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> present_modes;
};

inline SwapchainSupportDetails query_swapchain_support(VkPhysicalDevice device, VkSurfaceKHR surface) {
	SwapchainSupportDetails details;
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

	uint32_t formatCount;
	vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
	if (formatCount != 0) {
		details.formats.resize(formatCount);
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
	}

	uint32_t presentModeCount;
	vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
	if (presentModeCount != 0) {
		details.present_modes.resize(presentModeCount);
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.present_modes.data());
	}
	return details;
}

inline VkSurfaceFormatKHR choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
	auto it = std::ranges::find_if(availableFormats, [](const VkSurfaceFormatKHR& f) {
		return f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
	});
	return it != availableFormats.end() ? *it : availableFormats[0];
}

// FIFO is the only mode the spec guarantees; the others are preferences that
// fall back to FIFO (with an Info log) when the surface can't do them. An enum
// rather than a vsync bool: a bool cannot spell IMMEDIATE, and a second knob
// added later would be two ways to say one thing.
enum class PresentMode {
	FIFO,       // vsync — capped to refresh rate
	MAILBOX,    // uncapped, no tearing (the default preference)
	IMMEDIATE,  // uncapped, tearing possible; for measurements
};

inline constexpr VkPresentModeKHR to_vk(PresentMode mode) {
	switch (mode) {
		case PresentMode::FIFO:      return VK_PRESENT_MODE_FIFO_KHR;
		case PresentMode::MAILBOX:   return VK_PRESENT_MODE_MAILBOX_KHR;
		case PresentMode::IMMEDIATE: return VK_PRESENT_MODE_IMMEDIATE_KHR;
	}
	// Not std::unreachable(): pybind enums accept arbitrary ints.
	return VK_PRESENT_MODE_FIFO_KHR;
}

inline VkPresentModeKHR choose_swap_present_mode(const std::vector<VkPresentModeKHR>& availablePresentModes,
                                                 PresentMode preferred) {
	const VkPresentModeKHR wanted = to_vk(preferred);
	return std::ranges::contains(availablePresentModes, wanted)
		? wanted
		: VK_PRESENT_MODE_FIFO_KHR;
}

inline VkExtent2D choose_swap_extent(const VkSurfaceCapabilitiesKHR& capabilities, int width, int height) {
	if (capabilities.currentExtent.width != (std::numeric_limits<uint32_t>::max)()) {
		return capabilities.currentExtent;
	} else {
		VkExtent2D actualExtent = {
			static_cast<uint32_t>(width),
			static_cast<uint32_t>(height)
		};
		actualExtent.width = (std::max)(capabilities.minImageExtent.width, (std::min)(capabilities.maxImageExtent.width, actualExtent.width));
		actualExtent.height = (std::max)(capabilities.minImageExtent.height, (std::min)(capabilities.maxImageExtent.height, actualExtent.height));
		return actualExtent;
	}
}

class SwapchainRenderer : public RenderTarget
{
public:
	static std::expected<std::unique_ptr<SwapchainRenderer>, Error> create(std::shared_ptr<Context> context, SurfaceProvider surface_provider,
	                                                                       PresentMode present_mode = PresentMode::MAILBOX)
	{
		if (!context->swapchain_supported())
		{
			return std::unexpected(err_window(
				"Vulkan: this Context has no swapchain support, so it cannot present. "
				"Render to a bazalt.RenderTarget instead, or check ctx.headless."));
		}

		// The frame index lives on the Context and is advanced by whichever renderer
		// ends a frame, so two renderers sharing a Context silently corrupt each
		// other's in-flight tracking. Say so instead of misrendering.
		// 0.5 moves the ring onto Context::begin_frame(), which is the real fix and
		// is what lets this restriction be lifted in 0.6.
		if (context->has_swapchain_renderer())
		{
			return std::unexpected(err_window(
				"This Context already has a SwapchainRenderer. Two renderers sharing "
				"one Context would corrupt each other's frame index. Multi-window "
				"support is planned once frame tracking moves off the Context."));
		}

		auto renderer = std::unique_ptr<SwapchainRenderer>(new SwapchainRenderer(context, std::move(surface_provider)));
		// The PREFERENCE is stored, not the resolved mode: swapchain recreation
		// re-negotiates, because availability can change with the surface.
		renderer->preferred_present_mode_ = present_mode;

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
			context->physical_device(),
			context->graphics_queue_family(),
			surface,
			&presentSupport
		);
		if (!presentSupport)
		{
			return std::unexpected(err_init("Vulkan: Graphics queue does not support present to this surface"));
		}
		renderer->present_queue_ = context->graphics_queue();

		// Swapchain
		auto [width, height] = renderer->surface_provider_.get_framebuffer_size();
		if (auto r = renderer->create_swapchain_manually(width, height); !r) {
			// Propagate the real Error: this used to collapse to a bare
			// "Failed to create swapchain", discarding the VkResult that says why.
			return std::unexpected(r.error());
		}

		// Sync Objects
		VkSemaphoreCreateInfo semaphoreInfo{
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0
		};

		VkFenceCreateInfo fenceInfo{
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.pNext = nullptr,
			.flags = VK_FENCE_CREATE_SIGNALED_BIT
		};

		renderer->image_available_semaphores_.resize(context->frames_in_flight(), VK_NULL_HANDLE);
		renderer->in_flight_fences_.resize(context->frames_in_flight(), VK_NULL_HANDLE);
		for (size_t i = 0; i < context->frames_in_flight(); i++)
		{
			if (auto e = check(vkCreateSemaphore(context->device(), &semaphoreInfo, nullptr, &renderer->image_available_semaphores_[i]),
			                   "create image available semaphore"))
			{
				return std::unexpected(*e);
			}
			if (auto e = check(vkCreateFence(context->device(), &fenceInfo, nullptr, &renderer->in_flight_fences_[i]),
			                   "create in flight fence"))
			{
				return std::unexpected(*e);
			}
		}

		renderer->render_finished_semaphores_.resize(renderer->swapchain_images_.size());
		for (size_t i = 0; i < renderer->swapchain_images_.size(); i++)
		{
			if (auto e = check(vkCreateSemaphore(context->device(), &semaphoreInfo, nullptr, &renderer->render_finished_semaphores_[i]),
			                   "create render finished semaphore"))
			{
				return std::unexpected(*e);
			}
		}

		// Depth Image
		renderer->depth_format_ = VK_FORMAT_D32_SFLOAT;
		if (auto r = renderer->create_depth_resources(); !r)
		{
			return std::unexpected(r.error());
		}

		// Best-effort: a device without timestamp support just reports
		// gpu_time_ms as None, never an error.
		renderer->create_timestamp_pool_();

		context->set_has_swapchain_renderer(true);
		return renderer;
	}

	~SwapchainRenderer()
	{
		if (!context_)
		{
			return;
		}
		context_->set_has_swapchain_renderer(false);

		if (context_->device())
		{
			std::lock_guard lock(context_->queue_mutex());
			vkDeviceWaitIdle(context_->device());
		}

		for (size_t i = 0; i < image_available_semaphores_.size(); ++i)
		{
			if (image_available_semaphores_[i]) vkDestroySemaphore(context_->device(), image_available_semaphores_[i], nullptr);
			if (in_flight_fences_[i]) vkDestroyFence(context_->device(), in_flight_fences_[i], nullptr);
		}
		
		for (auto sem : render_finished_semaphores_)
		{
			if (sem) vkDestroySemaphore(context_->device(), sem, nullptr);
		}

		if (timestamp_pool_) {
			vkDestroyQueryPool(context_->device(), timestamp_pool_, nullptr);
		}

		if (depth_image_view_) {
			vkDestroyImageView(context_->device(), depth_image_view_, nullptr);
		}
		if (depth_image_ && depth_image_allocation_) {
			vmaDestroyImage(context_->allocator(), depth_image_, depth_image_allocation_);
		}

		for (auto iv : swapchain_image_views_)
		{
			vkDestroyImageView(context_->device(), iv, nullptr);
		}

		if (swapchain_) {
			vkDestroySwapchainKHR(context_->device(), swapchain_, nullptr);
		}

		if (surface_)
		{
			vkDestroySurfaceKHR(context_->instance(), surface_, nullptr);
		}
	}

	SwapchainRenderer(const SwapchainRenderer&) = delete;
	SwapchainRenderer& operator=(const SwapchainRenderer&) = delete;

	std::shared_ptr<Context> context() const { return context_; }
	VkSwapchainKHR swapchain() const { return swapchain_; }
	VkFormat swapchain_format() const { return swapchain_format_; }
	VkExtent2D swapchain_extent() const { return swapchain_extent_; }
	const std::vector<VkImage>& swapchain_images() const { return swapchain_images_; }
	const std::vector<VkImageView>& swapchain_image_views() const { return swapchain_image_views_; }
	VkImage depth_image() const override { return depth_image_; }
	VkImageView depth_image_view() const { return depth_image_view_; }

	// ── RenderTarget ──────────────────────────────────────────────────────────
	//
	// A swapchain hands out a different image each frame, which is exactly why
	// these are resolved at replay time rather than baked in when recording.

	std::uint32_t color_count() const override { return 1; }
	VkImage color_image(std::uint32_t) const override { return swapchain_images_[image_index_]; }
	VkImageView color_view(std::uint32_t) const override { return swapchain_image_views_[image_index_]; }
	VkFormat color_format(std::uint32_t) const override { return swapchain_format_; }
	VkImageView depth_view() const override { return depth_image_view_; }
	VkFormat depth_format() const override { return depth_format_; }
	VkExtent2D extent() const override { return swapchain_extent_; }

	// The one line that used to be hardcoded inside every end_rendering.
	VkImageLayout final_layout() const override { return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; }

	std::uint32_t current_frame() const { return context_->frame_index(); }
	std::uint64_t current_serial() const { return context_->frame_serial(); }
	std::uint32_t current_image_index() const { return image_index_; }
	bool frame_skipped() const { return frame_skipped_; }

	// The mode actually in use (post-fallback), not the requested preference.
	PresentMode present_mode() const
	{
		switch (active_present_mode_)
		{
			case VK_PRESENT_MODE_MAILBOX_KHR:   return PresentMode::MAILBOX;
			case VK_PRESENT_MODE_IMMEDIATE_KHR: return PresentMode::IMMEDIATE;
			default:                            return PresentMode::FIFO;
		}
	}

	// ── GPU timing ────────────────────────────────────────────────────────────
	//
	// The GPU duration of the frame submitted `frames_in_flight` frames ago (a
	// timestamp pair around each submit, read back once its fence is signalled).
	// None for the first frames_in_flight frames, and on devices without
	// timestamp support. Windowed only: the headless submit is a blocking
	// wait-idle, where wall-clock time already is the GPU time.
	std::optional<double> gpu_time_ms() const { return last_gpu_time_ms_; }

	bool timestamps_supported() const { return timestamp_pool_ != VK_NULL_HANDLE; }
	VkQueryPool timestamp_pool() const { return timestamp_pool_; }
	// submit() calls this after recording the timestamp pair for current_frame(),
	// so begin_frame knows the slot has results to read next time round.
	void mark_timestamp_written() { slot_written_[current_frame()] = true; }

	void submit(std::shared_ptr<CommandBuffer> cmd, std::uint64_t upload_wait_serial = 0);

	// Returns true if a frame was successfully acquired and is ready for rendering.
	// Returns false if the frame was skipped (minimized, resize, etc.) — caller should skip rendering.
	bool begin_frame()
	{
		frame_skipped_ = false;

		// The frame ring advances at the START of a frame, on the Context — not
		// at the end, on the renderer, as it used to. Everything below indexes
		// per-frame state through current_frame(), so consistency within one
		// frame is all that matters. A skipped frame burns a ring slot, which is
		// harmless: nothing gets submitted under it.
		context_->advance_frame();

		// Apply any pending hot reloads before this frame records: pipeline
		// rebuilds are handle swaps and old handles retire through the deletion
		// queue keyed by the current submit serial, so in-flight frames are safe.
		if (auto* hr = context_->hot_reload())
		{
			hr->drain();
		}

		// Check framebuffer size — return false if minimized (0x0)
		auto [width, height] = surface_provider_.get_framebuffer_size();
		if (width == 0 || height == 0) {
			frame_skipped_ = true;
			return false;
		}

		vkWaitForFences(context_->device(), 1, &in_flight_fences_[current_frame()], VK_TRUE, UINT64_MAX);

		// The fence proves this slot's previous submission finished, so its
		// timestamp pair is ready to read (frames_in_flight frames of latency).
		read_timestamps_();

		// A frame boundary is the natural point to reclaim deferred handles;
		// the submission timeline says how far the GPU actually got.
		context_->flush_deletion_queue();

		VkResult result = vkAcquireNextImageKHR(
			context_->device(),
			swapchain_,
			UINT64_MAX,
			image_available_semaphores_[current_frame()],
			VK_NULL_HANDLE,
			&image_index_
		);

		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			recreate_swapchain();
			frame_skipped_ = true;
			return false;
		} else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			if (auto l = context_->logger()) l->log(Severity::Error, Source::Device,
				std::format("Failed to acquire swapchain image ({})", vk_result_name(result)));
			frame_skipped_ = true;
			return false;
		}

		vkResetFences(context_->device(), 1, &in_flight_fences_[current_frame()]);
		return true;
	}

	// upload_wait_serial: the highest submission-timeline value this frame's
	// resources depend on (async uploads). 0 waits for nothing — a timeline
	// wait for 0 is trivially satisfied, so no branching is needed.
	void end_frame(VkCommandBuffer cmd, std::uint64_t upload_wait_serial = 0)
	{
		VkSemaphore waitSemaphores[] = { image_available_semaphores_[current_frame()],
		                                 context_->submit_timeline() };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		                                      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT };
		std::uint64_t waitValues[] = { 0, upload_wait_serial };  // binary sem value ignored

		VkSemaphore signalSemaphores[] = { render_finished_semaphores_[image_index_],
		                                   context_->submit_timeline() };
		VkSwapchainKHR swapchains[] = { swapchain_ };

		VkPresentInfoKHR presentInfo{
			.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
			.pNext = nullptr,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = signalSemaphores,
			.swapchainCount = 1,
			.pSwapchains = swapchains,
			.pImageIndices = &image_index_,
			.pResults = nullptr
		};

		// The lock ends before recreate_swapchain below: that path takes the
		// device idle, which must not happen while holding the queue mutex.
		VkResult result;
		{
			std::lock_guard lock(context_->queue_mutex());

			// Every submit signals the timeline; the serial is reserved under
			// the same lock that orders the submits.
			std::uint64_t signalValues[] = { 0, context_->advance_submit_serial() };

			VkTimelineSemaphoreSubmitInfo timelineInfo{
				.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
				.pNext = nullptr,
				.waitSemaphoreValueCount = 2,
				.pWaitSemaphoreValues = waitValues,
				.signalSemaphoreValueCount = 2,
				.pSignalSemaphoreValues = signalValues
			};

			VkSubmitInfo submitInfo{
				.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
				.pNext = &timelineInfo,
				.waitSemaphoreCount = 2,
				.pWaitSemaphores = waitSemaphores,
				.pWaitDstStageMask = waitStages,
				.commandBufferCount = 1,
				.pCommandBuffers = &cmd,
				.signalSemaphoreCount = 2,
				.pSignalSemaphores = signalSemaphores
			};

			if (VkResult submit_result = vkQueueSubmit(context_->graphics_queue(), 1, &submitInfo, in_flight_fences_[current_frame()]);
			    submit_result != VK_SUCCESS) {
				if (auto l = context_->logger()) l->log(Severity::Error, Source::Device,
					std::format("Failed to submit draw command buffer ({})", vk_result_name(submit_result)));
			}

			result = vkQueuePresentKHR(present_queue_, &presentInfo);
		}

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || surface_provider_.consume_resize_flag()) {
			recreate_swapchain();
		}
	}

private:
	SwapchainRenderer(std::shared_ptr<Context> context, SurfaceProvider surface_provider)
		: context_(context), surface_provider_(std::move(surface_provider)) {}

	std::shared_ptr<Context> context_;
	SurfaceProvider surface_provider_;

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
		// timestamps, begin_frame reads none, and gpu_time_ms stays None — no
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
			.pipelineStatistics = 0
		};
		if (vkCreateQueryPool(context_->device(), &poolInfo, nullptr, &timestamp_pool_) != VK_SUCCESS)
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
		std::uint64_t ts[2] = { 0, 0 };
		if (vkGetQueryPoolResults(context_->device(), timestamp_pool_, 2 * slot, 2,
		                          sizeof(ts), ts, sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
		{
			last_gpu_time_ms_ = std::nullopt;
			return;
		}
		const std::uint64_t mask = timestamp_valid_bits_ >= 64
			? ~std::uint64_t{ 0 }
			: ((std::uint64_t{ 1 } << timestamp_valid_bits_) - 1);
		const std::uint64_t delta = (ts[1] - ts[0]) & mask;
		last_gpu_time_ms_ = static_cast<double>(delta) * static_cast<double>(timestamp_period_) / 1.0e6;
	}

	std::expected<void, Error> create_swapchain_manually(int width, int height, VkSwapchainKHR old_swapchain = VK_NULL_HANDLE)
	{
		auto details = query_swapchain_support(context_->physical_device(), surface_);
		auto surface_format = choose_swap_surface_format(details.formats);
		auto present_mode = choose_swap_present_mode(details.present_modes, preferred_present_mode_);
		if (present_mode != to_vk(preferred_present_mode_)) {
			if (auto l = context_->logger()) l->log(Severity::Info, Source::Device,
				"Requested present mode is not supported by this surface; falling back to FIFO (vsync)");
		}
		active_present_mode_ = present_mode;
		auto extent = choose_swap_extent(details.capabilities, width, height);

		uint32_t image_count = details.capabilities.minImageCount + 1;
		if (details.capabilities.maxImageCount > 0 && image_count > details.capabilities.maxImageCount) {
			image_count = details.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR createInfo{
			.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
			.pNext = nullptr,
			.flags = 0,
			.surface = surface_,
			.minImageCount = image_count,
			.imageFormat = surface_format.format,
			.imageColorSpace = surface_format.colorSpace,
			.imageExtent = extent,
			.imageArrayLayers = 1,
			.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
			.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.queueFamilyIndexCount = 0,
			.pQueueFamilyIndices = nullptr,
			.preTransform = details.capabilities.currentTransform,
			.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
			.presentMode = present_mode,
			.clipped = VK_TRUE,
			.oldSwapchain = old_swapchain
		};

		// On some systems compositeAlpha might not support OPAQUE, so select the first supported one
		if (!(details.capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)) {
			if (details.capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR) {
				createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
			} else if (details.capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR) {
				createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR;
			} else if (details.capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR) {
				createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR;
			}
		}

		VkSwapchainKHR new_swapchain;
		if (auto e = check(vkCreateSwapchainKHR(context_->device(), &createInfo, nullptr, &new_swapchain),
		                   "create swapchain")) {
			return std::unexpected(*e);
		}

		swapchain_ = new_swapchain;
		swapchain_format_ = surface_format.format;
		swapchain_extent_ = extent;

		// Retrieve swapchain images
		uint32_t actual_image_count;
		vkGetSwapchainImagesKHR(context_->device(), swapchain_, &actual_image_count, nullptr);
		swapchain_images_.resize(actual_image_count);
		vkGetSwapchainImagesKHR(context_->device(), swapchain_, &actual_image_count, swapchain_images_.data());

		// Create swapchain image views
		swapchain_image_views_.resize(actual_image_count);
		for (size_t i = 0; i < actual_image_count; i++) {
			VkImageViewCreateInfo viewInfo{
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.pNext = nullptr,
				.flags = 0,
				.image = swapchain_images_[i],
				.viewType = VK_IMAGE_VIEW_TYPE_2D,
				.format = swapchain_format_,
				.components = {
					.r = VK_COMPONENT_SWIZZLE_IDENTITY,
					.g = VK_COMPONENT_SWIZZLE_IDENTITY,
					.b = VK_COMPONENT_SWIZZLE_IDENTITY,
					.a = VK_COMPONENT_SWIZZLE_IDENTITY
				},
				.subresourceRange = {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1
				}
			};

			if (auto e = check(vkCreateImageView(context_->device(), &viewInfo, nullptr, &swapchain_image_views_[i]),
			                   "create swapchain image view")) {
				return std::unexpected(*e);
			}
		}

		return {};
	}

	void recreate_swapchain()
	{
		{
			std::lock_guard lock(context_->queue_mutex());
			vkDeviceWaitIdle(context_->device());
		}

		// Destroy old depth resources
		if (depth_image_view_) {
			vkDestroyImageView(context_->device(), depth_image_view_, nullptr);
			depth_image_view_ = VK_NULL_HANDLE;
		}
		if (depth_image_ && depth_image_allocation_) {
			vmaDestroyImage(context_->allocator(), depth_image_, depth_image_allocation_);
			depth_image_ = VK_NULL_HANDLE;
			depth_image_allocation_ = VK_NULL_HANDLE;
		}

		// Destroy old render-finished semaphores
		for (auto sem : render_finished_semaphores_) {
			if (sem) vkDestroySemaphore(context_->device(), sem, nullptr);
		}
		render_finished_semaphores_.clear();

		// Destroy old swapchain image views
		for (auto iv : swapchain_image_views_) {
			vkDestroyImageView(context_->device(), iv, nullptr);
		}
		swapchain_image_views_.clear();
		swapchain_images_.clear();

		VkSwapchainKHR old_swapchain = swapchain_;
		auto [width, height] = surface_provider_.get_framebuffer_size();
		if (auto r = create_swapchain_manually(width, height, old_swapchain); !r) {
			// This runs mid-frame, so it keeps the log-and-bail contract — but it
			// logs the propagated Error (VkResult name included), not a hand-written
			// string.
			if (auto l = context_->logger()) l->log(Severity::Error, Source::Device,
				"Failed to recreate swapchain: " + r.error().message);
			return;
		}

		if (old_swapchain != VK_NULL_HANDLE) {
			vkDestroySwapchainKHR(context_->device(), old_swapchain, nullptr);
		}

		// Recreate render-finished semaphores (one per swapchain image)
		VkSemaphoreCreateInfo semaphoreInfo{
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0
		};
		render_finished_semaphores_.resize(swapchain_images_.size());
		for (size_t i = 0; i < swapchain_images_.size(); i++) {
			if (vkCreateSemaphore(context_->device(), &semaphoreInfo, nullptr, &render_finished_semaphores_[i]) != VK_SUCCESS) {
				if (auto l = context_->logger()) l->log(Severity::Error, Source::Device, "Failed to recreate render finished semaphores");
				return;
			}
		}

		// Recreate depth image
		if (auto r = create_depth_resources(); !r) {
			if (auto l = context_->logger()) l->log(Severity::Error, Source::Device,
				"Failed to recreate depth resources: " + r.error().message);
			return;
		}

		if (auto l = context_->logger()) l->log(Severity::Info, Source::Device,
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
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.tiling = VK_IMAGE_TILING_OPTIMAL,
			.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.queueFamilyIndexCount = 0,
			.pQueueFamilyIndices = nullptr,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
		};

		VmaAllocationCreateInfo allocImageInfo = {};
		allocImageInfo.usage = VMA_MEMORY_USAGE_AUTO;

		if (auto e = check(vmaCreateImage(context_->allocator(), &imageInfo, &allocImageInfo, &depth_image_, &depth_image_allocation_, nullptr),
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
			.components = {
				VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
				VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY
			},
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1
			}
		};

		if (auto e = check(vkCreateImageView(context_->device(), &viewInfo, nullptr, &depth_image_view_),
		                   "create depth image view"))
		{
			return std::unexpected(*e);
		}

		return {};
	}
};
// One successfully acquired swapchain frame. begin_frame() hands this out
// instead of a bool: "a frame exists" and "here is the frame" are the same
// fact, and putting submit() on the frame makes submitting without having
// acquired one unrepresentable.
//
// The serial is a generation guard: a Frame held across ticks (the obvious
// PyQt mistake) fails with a readable error instead of a validation storm.
struct Frame
{
	std::shared_ptr<SwapchainRenderer> renderer;
	std::uint64_t serial = 0;
	std::uint32_t frame_index = 0;
	std::uint32_t image_index = 0;
	bool submitted = false;

	// The GPU time of the frame frames_in_flight ago, read back when this frame
	// was acquired; None until the ring has cycled once, and on unsupported
	// devices. Snapshotted at begin_frame so it is stable for the frame's life.
	std::optional<double> gpu_time_ms;

	// Records, submits and presents. Defined in main.cpp next to record_frame.
	std::expected<void, Error> submit(std::shared_ptr<CommandBuffer> cmd);
};
