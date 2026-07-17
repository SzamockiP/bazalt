#pragma once
#include <volk.h>
#include <vk_mem_alloc.h>

#include <cstdint>
#include <expected>
#include <memory>
#include <vector>

#include "Context.hpp"
#include "Error.hpp"
#include "ImmediateSubmit.hpp"

// Anything that can be drawn into.
//
// This interface exists to answer the only four questions a recorded command
// actually asks at replay time: which colour attachments, which depth
// attachment, how big, and what layout the result must end in. CommandBuffer
// used to take a `SwapchainRenderer&` for exactly that, which is why headless
// rendering, render-to-texture and MRT were all impossible at once, and why
// end_rendering could hardcode VK_IMAGE_LAYOUT_PRESENT_SRC_KHR.
//
// A swapchain is now one implementation of this, not the whole world.
class RenderTarget
{
public:
	virtual ~RenderTarget() = default;

	virtual std::uint32_t color_count() const = 0;

	// Queried at replay time, not record time: a swapchain hands out a different
	// image every frame.
	virtual VkImage color_image(std::uint32_t index) const = 0;
	virtual VkImageView color_view(std::uint32_t index) const = 0;
	virtual VkFormat color_format(std::uint32_t index) const = 0;

	// VK_NULL_HANDLE when the target has no depth attachment.
	virtual VkImage depth_image() const = 0;
	virtual VkImageView depth_view() const = 0;
	virtual VkFormat depth_format() const = 0;

	virtual VkExtent2D extent() const = 0;

	// The layout the colour attachments must be left in when rendering ends.
	// A swapchain needs PRESENT_SRC_KHR; an offscreen target that will be sampled
	// needs SHADER_READ_ONLY_OPTIMAL. This being a virtual is what removes the
	// hardcoded present transition from CommandBuffer.
	virtual VkImageLayout final_layout() const = 0;

	// Called by CommandBuffer when the end-of-rendering barrier is recorded into
	// a real submit. An OffscreenTarget uses this to learn that its image has
	// left UNDEFINED — the submit paths never see the target (it lives inside the
	// recorded lambdas), so the notification has to come from the recording
	// itself. No-op for targets that don't care.
	virtual void on_rendering_recorded() {}
};

// Everything a recorded command needs that isn't known until replay.
//
// Deliberately does NOT carry the target: begin_rendering captures its own, so a
// single command buffer can render into a shadow map and then a window. A target
// here would be both dead weight and a limit.
struct FrameContext
{
	std::uint32_t frame_index = 0;
};

// A render target backed by images this object owns, with no swapchain and no
// window involved. This is what makes headless rendering — and therefore the
// test suite — possible.
class OffscreenTarget : public RenderTarget, public std::enable_shared_from_this<OffscreenTarget>
{
public:
	static std::expected<std::shared_ptr<OffscreenTarget>, Error> create(
		Context& context, std::uint32_t width, std::uint32_t height, bool with_depth)
	{
		auto target = std::shared_ptr<OffscreenTarget>(new OffscreenTarget(context.shared_from_this()));
		target->extent_ = { width, height };

		// TRANSFER_SRC so read_pixels() can copy the result back out; SAMPLED so
		// the same image can be fed straight into a descriptor set (render-to-
		// texture needs no separate API — it's just this image).
		VkImageCreateInfo colorInfo{
			.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.imageType = VK_IMAGE_TYPE_2D,
			.format = target->color_format_,
			.extent = { width, height, 1 },
			.mipLevels = 1,
			.arrayLayers = 1,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.tiling = VK_IMAGE_TILING_OPTIMAL,
			.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
			         VK_IMAGE_USAGE_SAMPLED_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.queueFamilyIndexCount = 0,
			.pQueueFamilyIndices = nullptr,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
		};

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

		if (auto e = check(vmaCreateImage(context.allocator(), &colorInfo, &allocInfo,
		                                  &target->color_image_, &target->color_allocation_, nullptr),
		                   "create offscreen colour image", ErrorCode::Resource))
		{
			return std::unexpected(*e);
		}

		VkImageViewCreateInfo colorViewInfo{
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.image = target->color_image_,
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = target->color_format_,
			.components = {},
			.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
		};

		if (auto e = check(vkCreateImageView(context.device(), &colorViewInfo, nullptr, &target->color_view_),
		                   "create offscreen colour image view", ErrorCode::Resource))
		{
			return std::unexpected(*e);
		}

		if (with_depth)
		{
			VkImageCreateInfo depthInfo = colorInfo;
			depthInfo.format = target->depth_format_;
			depthInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

			if (auto e = check(vmaCreateImage(context.allocator(), &depthInfo, &allocInfo,
			                                  &target->depth_image_, &target->depth_allocation_, nullptr),
			                   "create offscreen depth image", ErrorCode::Resource))
			{
				return std::unexpected(*e);
			}

			VkImageViewCreateInfo depthViewInfo = colorViewInfo;
			depthViewInfo.image = target->depth_image_;
			depthViewInfo.format = target->depth_format_;
			depthViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

			if (auto e = check(vkCreateImageView(context.device(), &depthViewInfo, nullptr, &target->depth_view_),
			                   "create offscreen depth image view", ErrorCode::Resource))
			{
				return std::unexpected(*e);
			}
		}

		return target;
	}

	~OffscreenTarget() override
	{
		if (!context_)
		{
			return;
		}
		if (depth_view_) vkDestroyImageView(context_->device(), depth_view_, nullptr);
		if (depth_image_) vmaDestroyImage(context_->allocator(), depth_image_, depth_allocation_);
		if (color_view_) vkDestroyImageView(context_->device(), color_view_, nullptr);
		if (color_image_) vmaDestroyImage(context_->allocator(), color_image_, color_allocation_);
	}

	OffscreenTarget(const OffscreenTarget&) = delete;
	OffscreenTarget& operator=(const OffscreenTarget&) = delete;

	std::uint32_t color_count() const override { return 1; }
	VkImage color_image(std::uint32_t) const override { return color_image_; }
	VkImageView color_view(std::uint32_t) const override { return color_view_; }
	VkFormat color_format(std::uint32_t) const override { return color_format_; }
	VkImage depth_image() const override { return depth_image_; }
	VkImageView depth_view() const override { return depth_view_; }
	VkFormat depth_format() const override { return depth_image_ ? depth_format_ : VK_FORMAT_UNDEFINED; }
	VkExtent2D extent() const override { return extent_; }

	// Left ready to be sampled, so using the result as a texture needs no extra
	// step. read_pixels() transitions from here when it needs to.
	VkImageLayout final_layout() const override { return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; }

	// Copies the colour attachment back to host memory. Blocking, and it stalls
	// the GPU — acceptable because this is a debugging/test path, not a per-frame
	// one. 0.5 replaces the stall with the timeline-based UploadManager.
	std::expected<std::vector<std::uint8_t>, Error> read_pixels()
	{
		// Refuse rather than return uninitialised VRAM: before the first submit the
		// image is UNDEFINED, and a barrier claiming otherwise would let the driver
		// hand back garbage that *sometimes* looks right.
		if (!rendered_)
		{
			return std::unexpected(err_resource(
				"read_pixels() called on a RenderTarget that has never been rendered "
				"to; submit a command buffer targeting it first"));
		}

		const VkDeviceSize size = static_cast<VkDeviceSize>(extent_.width) * extent_.height * 4;

		VkBufferCreateInfo bufferInfo{
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.size = size,
			.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.queueFamilyIndexCount = 0,
			.pQueueFamilyIndices = nullptr
		};

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_TO_CPU;
		allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

		VkBuffer staging = VK_NULL_HANDLE;
		VmaAllocation staging_alloc = VK_NULL_HANDLE;
		if (auto e = check(vmaCreateBuffer(context_->allocator(), &bufferInfo, &allocInfo,
		                                   &staging, &staging_alloc, nullptr),
		                   "create readback buffer", ErrorCode::Resource))
		{
			return std::unexpected(*e);
		}

		auto submitted = immediate_submit(*context_, [&](VkCommandBuffer cmd) {
			// The image is in final_layout() after rendering; move it to TRANSFER_SRC
			// and back so the target stays usable for sampling afterwards.
			transition(cmd, final_layout(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

			VkBufferImageCopy region{
				.bufferOffset = 0,
				.bufferRowLength = 0,
				.bufferImageHeight = 0,
				.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
				.imageOffset = { 0, 0, 0 },
				.imageExtent = { extent_.width, extent_.height, 1 }
			};
			vkCmdCopyImageToBuffer(cmd, color_image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging, 1, &region);

			transition(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, final_layout());
		});
		if (!submitted)
		{
			vmaDestroyBuffer(context_->allocator(), staging, staging_alloc);
			return std::unexpected(submitted.error());
		}

		std::vector<std::uint8_t> pixels(static_cast<size_t>(size));
		void* mapped = nullptr;
		if (auto e = check(vmaMapMemory(context_->allocator(), staging_alloc, &mapped),
		                   "map readback buffer memory", ErrorCode::Resource))
		{
			vmaDestroyBuffer(context_->allocator(), staging, staging_alloc);
			return std::unexpected(*e);
		}
		std::memcpy(pixels.data(), mapped, static_cast<size_t>(size));
		vmaUnmapMemory(context_->allocator(), staging_alloc);
		vmaDestroyBuffer(context_->allocator(), staging, staging_alloc);

		return pixels;
	}

	bool rendered_at_least_once() const { return rendered_; }

	// Flipped by the end-of-rendering barrier at execute() time — i.e. only when
	// work targeting this image is actually submitted. This used to be a public
	// mark_rendered() that nothing ever called, which left rendered_ permanently
	// false and read_pixels() transitioning from UNDEFINED — a spec-sanctioned
	// licence for the driver to discard the just-rendered contents.
	void on_rendering_recorded() override { rendered_ = true; }

private:
	explicit OffscreenTarget(std::shared_ptr<Context> context) : context_(std::move(context)) {}

	void transition(VkCommandBuffer cmd, VkImageLayout from, VkImageLayout to)
	{
		// `from` is always truthful here: read_pixels() refuses to run before the
		// first render, so the image is guaranteed to be in final_layout() by the
		// time this barrier is recorded.
		VkImageMemoryBarrier barrier{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.pNext = nullptr,
			.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
			.oldLayout = from,
			.newLayout = to,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = color_image_,
			.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
		};
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
		                     0, 0, nullptr, 0, nullptr, 1, &barrier);
	}

	std::shared_ptr<Context> context_;
	VkExtent2D extent_{};

	// 0.5 makes these configurable; 0.4 only needs the abstraction to exist.
	VkFormat color_format_ = VK_FORMAT_R8G8B8A8_UNORM;
	VkFormat depth_format_ = VK_FORMAT_D32_SFLOAT;

	VkImage color_image_ = VK_NULL_HANDLE;
	VmaAllocation color_allocation_ = VK_NULL_HANDLE;
	VkImageView color_view_ = VK_NULL_HANDLE;

	VkImage depth_image_ = VK_NULL_HANDLE;
	VmaAllocation depth_allocation_ = VK_NULL_HANDLE;
	VkImageView depth_view_ = VK_NULL_HANDLE;

	bool rendered_ = false;
};
