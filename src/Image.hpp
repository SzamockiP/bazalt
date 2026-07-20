#pragma once
#include <volk.h>
#include <vk_mem_alloc.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <expected>
#include <format>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "stb_image.h"
#include "Context.hpp"
#include "Error.hpp"
#include "Format.hpp"
#include "ImmediateSubmit.hpp"

// Layout transition helper shared by uploads, mip generation and readback.
// Formerly a private of Texture; RenderTarget grew its own copy, which is
// exactly the drift this ends.
inline void record_image_transition(VkCommandBuffer cmd, VkImage image,
	VkImageLayout oldLayout, VkImageLayout newLayout,
	VkAccessFlags srcAccess, VkAccessFlags dstAccess,
	VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
	VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT,
	std::uint32_t baseMip = 0, std::uint32_t mipCount = 1)
{
	VkImageMemoryBarrier barrier{
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.pNext = nullptr,
		.srcAccessMask = srcAccess,
		.dstAccessMask = dstAccess,
		.oldLayout = oldLayout,
		.newLayout = newLayout,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image,
		.subresourceRange = {
			.aspectMask = aspect,
			.baseMipLevel = baseMip,
			.levelCount = mipCount,
			.baseArrayLayer = 0,
			.layerCount = 1
		}
	};
	vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

// A GPU image: VkImage + view + format. Nothing else — the sampler it used to
// be fused with lives in the Context's cache, and how the image is *used*
// (sampled, rendered into, read back) is the caller's business, not baked in
// at creation.
class Image : public std::enable_shared_from_this<Image>
{
public:
	Image(std::shared_ptr<Context> context, VkImage image, VmaAllocation allocation,
	      VkImageView view, Format format, std::uint32_t width, std::uint32_t height,
	      std::uint32_t mip_levels)
		: context_(std::move(context)), image_(image), allocation_(allocation),
		  view_(view), format_(format), width_(width), height_(height),
		  mip_levels_(mip_levels) {}

	// Deferred: an in-flight frame may still sample this image.
	~Image()
	{
		if (context_)
		{
			context_->defer_destroy(
				[device = context_->device(), allocator = context_->allocator(),
				 view = view_, image = image_, allocation = allocation_] {
					if (view != VK_NULL_HANDLE) vkDestroyImageView(device, view, nullptr);
					if (image != VK_NULL_HANDLE && allocation != VK_NULL_HANDLE)
					{
						vmaDestroyImage(allocator, image, allocation);
					}
				});
		}
	}

	Image(const Image&) = delete;
	Image& operator=(const Image&) = delete;

	VkImage vk_image() const { return image_; }
	VkImageView view() const { return view_; }
	Format format() const { return format_; }
	std::uint32_t width() const { return width_; }
	std::uint32_t height() const { return height_; }
	std::uint32_t mip_levels() const { return mip_levels_; }

	// "Has the GPU ever been given contents for this image" — uploaded, copied
	// from an array, or rendered into. Readback and sampling of a virgin image
	// are refused rather than returning driver-defined garbage (0.4.1 contract).
	bool has_contents() const { return has_contents_.load(); }
	VkImageLayout current_layout() const { return layout_; }
	void mark_has_contents(VkImageLayout layout)
	{
		layout_ = layout;
		has_contents_.store(true);
	}

	// ── The image IS the upload future ────────────────────────────────────────
	//
	// load_image returns immediately; the decode + copy runs on the upload
	// worker. The image is usable for *recording* right away — residency is
	// required only at submit, where the frame's GPU work waits on the
	// submission timeline (see require_resident). These members are the
	// explicit-control verbs.

	enum class UploadState { None, Pending, Submitted, Failed };

	// Non-blocking: is the pixel data on the GPU?
	bool ready() const
	{
		switch (upload_state_.load())
		{
			case UploadState::None: return has_contents_.load();
			case UploadState::Pending: return false;
			case UploadState::Failed: return false;
			case UploadState::Submitted:
				return context_->completed_submit_serial() >= upload_serial_.load();
		}
		return false;
	}

	// Block until this one image's upload has finished on the GPU.
	// A failed decode surfaces here as ResourceError.
	std::expected<void, Error> wait()
	{
		if (auto r = wait_submitted_(); !r)
		{
			return std::unexpected(r.error());
		}
		if (upload_state_.load() == UploadState::Submitted)
		{
			const std::uint64_t serial = upload_serial_.load();
			VkSemaphore timeline = context_->submit_timeline();
			VkSemaphoreWaitInfo waitInfo{
				.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
				.pNext = nullptr,
				.flags = 0,
				.semaphoreCount = 1,
				.pSemaphores = &timeline,
				.pValues = &serial
			};
			if (auto e = check(vkWaitSemaphores(context_->device(), &waitInfo, UINT64_MAX),
			                   "wait for image upload", ErrorCode::Resource))
			{
				return std::unexpected(*e);
			}
		}
		return {};
	}

	// Called at submit time for every image a command buffer references.
	// Returns the timeline serial the frame's GPU work must wait for (0 when
	// nothing is pending — RTT attachments and synchronously uploaded images
	// short-circuit here). CPU-blocks only while the worker is still decoding.
	std::expected<std::uint64_t, Error> require_resident()
	{
		if (upload_state_.load() == UploadState::None)
		{
			return 0;
		}
		if (auto r = wait_submitted_(); !r)
		{
			return std::unexpected(r.error());
		}
		return upload_serial_.load();
	}

	// Worker-side state transitions.
	void set_upload_pending() { upload_state_.store(UploadState::Pending); }
	void set_upload_submitted(std::uint64_t serial)
	{
		{
			std::lock_guard lock(upload_mutex_);
			upload_serial_.store(serial);
			mark_has_contents(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			upload_state_.store(UploadState::Submitted);
		}
		upload_cv_.notify_all();
	}
	void set_upload_failed(std::string message)
	{
		{
			std::lock_guard lock(upload_mutex_);
			upload_error_ = std::move(message);
			upload_state_.store(UploadState::Failed);
		}
		upload_cv_.notify_all();
	}

	// ── Creation ──────────────────────────────────────────────────────────────

	// Every usage the format legally supports, filtered through the driver's
	// format properties. There is no `usage=` parameter on purpose: forgetting
	// STORAGE_BIT (or TRANSFER_SRC, or SAMPLED) is a classic Vulkan paper cut
	// with no upside — the driver knows what the format can do, so ask it.
	static VkImageUsageFlags usage_for(Context& context, Format format)
	{
		VkFormatProperties props{};
		vkGetPhysicalDeviceFormatProperties(context.physical_device(), format_info(format).vk, &props);
		const VkFormatFeatureFlags feat = props.optimalTilingFeatures;

		VkImageUsageFlags usage = 0;
		if (feat & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
		if (feat & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT) usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		if (feat & VK_FORMAT_FEATURE_TRANSFER_DST_BIT) usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		if (feat & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		if (feat & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		if (feat & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) usage |= VK_IMAGE_USAGE_STORAGE_BIT;
		return usage;
	}

	// Mip generation needs to blit and to linearly filter the format.
	static bool can_generate_mips(Context& context, Format format)
	{
		if (format_info(format).depth)
		{
			return false;
		}
		VkFormatProperties props{};
		vkGetPhysicalDeviceFormatProperties(context.physical_device(), format_info(format).vk, &props);
		constexpr VkFormatFeatureFlags needed =
			VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT |
			VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
		return (props.optimalTilingFeatures & needed) == needed;
	}

	static std::uint32_t full_mip_count(std::uint32_t width, std::uint32_t height)
	{
		std::uint32_t mips = 1;
		std::uint32_t size = width > height ? width : height;
		while (size > 1)
		{
			size /= 2;
			++mips;
		}
		return mips;
	}

	// An empty image: no contents, layout UNDEFINED. The building block for
	// render-target attachments and array uploads.
	static std::expected<std::shared_ptr<Image>, Error> create_empty(
		Context& context, std::uint32_t width, std::uint32_t height, Format format,
		std::uint32_t mip_levels = 1)
	{
		if (width == 0 || height == 0)
		{
			return std::unexpected(err_resource(std::format(
				"Image dimensions must be non-zero, got {}x{}", width, height)));
		}
		const FormatInfo info = format_info(format);

		VkImageCreateInfo imageInfo{
			.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.imageType = VK_IMAGE_TYPE_2D,
			.format = info.vk,
			.extent = { width, height, 1 },
			.mipLevels = mip_levels,
			.arrayLayers = 1,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.tiling = VK_IMAGE_TILING_OPTIMAL,
			.usage = usage_for(context, format),
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.queueFamilyIndexCount = 0,
			.pQueueFamilyIndices = nullptr,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
		};

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;
		if (auto e = check(vmaCreateImage(context.allocator(), &imageInfo, &allocInfo,
		                                  &image, &allocation, nullptr),
		                   std::format("create {} image", format_name(format)), ErrorCode::Resource))
		{
			return std::unexpected(*e);
		}

		VkImageViewCreateInfo viewInfo{
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.image = image,
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = info.vk,
			.components = {},
			.subresourceRange = {
				static_cast<VkImageAspectFlags>(
					info.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT),
				0, mip_levels, 0, 1
			}
		};

		VkImageView view = VK_NULL_HANDLE;
		if (auto e = check(vkCreateImageView(context.device(), &viewInfo, nullptr, &view),
		                   std::format("create {} image view", format_name(format)), ErrorCode::Resource))
		{
			vmaDestroyImage(context.allocator(), image, allocation);
			return std::unexpected(*e);
		}

		return std::make_shared<Image>(context.shared_from_this(), image, allocation,
		                               view, format, width, height, mip_levels);
	}

	// From caller-provided pixels (numpy arrays land here). UNORM by default at
	// the binding layer: arrays are data, files are pictures. One mip — data
	// images don't want surprise filtering.
	static std::expected<std::shared_ptr<Image>, Error> create_from_pixels(
		Context& context, const void* pixels, std::uint32_t width, std::uint32_t height,
		Format format)
	{
		auto image = create_empty(context, width, height, format, 1);
		if (!image)
		{
			return image;
		}
		if (auto r = (*image)->upload_pixels(context, pixels, 1); !r)
		{
			return std::unexpected(r.error());
		}
		return image;
	}

	// From a file on disk: sRGB, full mip chain (when the format supports the
	// blits — otherwise silently one level, matching the anisotropy precedent:
	// a correct image that minifies slightly softer, not a failure).
	static std::expected<std::shared_ptr<Image>, Error> load_from_file(
		Context& context, const std::string& path)
	{
		int width = 0, height = 0, channels = 0;
		stbi_uc* pixels = stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb_alpha);
		if (!pixels)
		{
			// stb knows whether this was a missing file or a corrupt one; saying
			// only "Failed to load" throws that away.
			const char* reason = stbi_failure_reason();
			return std::unexpected(err_resource(reason
				? std::format("Failed to load image: {} ({})", path, reason)
				: std::format("Failed to load image: {}", path)));
		}

		const Format format = Format::RGBA8_SRGB;
		const std::uint32_t mips = can_generate_mips(context, format)
			? full_mip_count(static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height))
			: 1;

		auto image = create_empty(context, static_cast<std::uint32_t>(width),
		                          static_cast<std::uint32_t>(height), format, mips);
		if (!image)
		{
			stbi_image_free(pixels);
			return image;
		}

		auto uploaded = (*image)->upload_pixels(context, pixels, mips);
		stbi_image_free(pixels);
		if (!uploaded)
		{
			return std::unexpected(uploaded.error());
		}
		return image;
	}

	// ── Readback ──────────────────────────────────────────────────────────────

	// Copies mip 0 back to host memory. Blocking, stalls the GPU — a debugging
	// and test path, not a per-frame one. Size and dtype come from the format
	// table; the binding layer shapes the bytes into a numpy array.
	std::expected<std::vector<std::byte>, Error> read()
	{
		// A pending async upload is finished first — read() is blocking anyway.
		if (auto w = wait(); !w)
		{
			return std::unexpected(w.error());
		}
		if (!has_contents_.load())
		{
			return std::unexpected(err_resource(
				"read() called on an Image that has no contents yet; upload to it or "
				"render into it first"));
		}

		const FormatInfo info = format_info(format_);
		const VkDeviceSize size =
			static_cast<VkDeviceSize>(width_) * height_ * info.bytes_per_pixel;
		const VkImageAspectFlags aspect =
			info.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;

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
			record_image_transition(cmd, image_, layout_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
				VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
				aspect, 0, mip_levels_);

			VkBufferImageCopy region{
				.bufferOffset = 0,
				.bufferRowLength = 0,
				.bufferImageHeight = 0,
				.imageSubresource = { aspect, 0, 0, 1 },
				.imageOffset = { 0, 0, 0 },
				.imageExtent = { width_, height_, 1 }
			};
			vkCmdCopyImageToBuffer(cmd, image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging, 1, &region);

			record_image_transition(cmd, image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, layout_,
				VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_MEMORY_READ_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
				aspect, 0, mip_levels_);
		});
		if (!submitted)
		{
			vmaDestroyBuffer(context_->allocator(), staging, staging_alloc);
			return std::unexpected(submitted.error());
		}

		std::vector<std::byte> out(static_cast<std::size_t>(size));
		void* mapped = nullptr;
		if (auto e = check(vmaMapMemory(context_->allocator(), staging_alloc, &mapped),
		                   "map readback buffer memory", ErrorCode::Resource))
		{
			vmaDestroyBuffer(context_->allocator(), staging, staging_alloc);
			return std::unexpected(*e);
		}
		std::memcpy(out.data(), mapped, static_cast<std::size_t>(size));
		vmaUnmapMemory(context_->allocator(), staging_alloc);
		vmaDestroyBuffer(context_->allocator(), staging, staging_alloc);

		return out;
	}

	// Records the whole first upload: transition all mips to TRANSFER_DST from
	// UNDEFINED (there are no contents to preserve), copy the staging buffer into
	// mip 0, then either blit the chain or transition to SHADER_READ_ONLY. The
	// sync path replays this through immediate_submit; the upload worker records
	// it into its own command buffer.
	void record_upload_commands(VkCommandBuffer cmd, VkBuffer staging, std::uint32_t mips)
	{
		record_image_transition(cmd, image_,
			VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			0, VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_IMAGE_ASPECT_COLOR_BIT, 0, mips);

		record_copy_and_finalize_(cmd, staging, mips);
	}

	// Hot reload: the image already holds contents that in-flight frames may
	// still be sampling. Transition from SHADER_READ_ONLY with a fragment-shader
	// source scope, so the copy waits for those reads — a WAR execution
	// dependency against every frame already submitted on this queue, no CPU
	// sync needed. UNDEFINED (as in the first upload) would instead let the
	// driver discard the live contents mid-frame, which sync validation flags.
	void record_reload_commands(VkCommandBuffer cmd, VkBuffer staging, std::uint32_t mips)
	{
		record_image_transition(cmd, image_,
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_IMAGE_ASPECT_COLOR_BIT, 0, mips);

		record_copy_and_finalize_(cmd, staging, mips);
	}

	// Creates and fills a staging buffer for this image's mip 0.
	std::expected<std::pair<VkBuffer, VmaAllocation>, Error> create_filled_staging(
		Context& context, const void* pixels)
	{
		const FormatInfo info = format_info(format_);
		const VkDeviceSize size =
			static_cast<VkDeviceSize>(width_) * height_ * info.bytes_per_pixel;

		VkBufferCreateInfo stagingInfo{
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.size = size,
			.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.queueFamilyIndexCount = 0,
			.pQueueFamilyIndices = nullptr
		};
		VmaAllocationCreateInfo stagingAllocInfo{};
		stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
		stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

		VkBuffer staging = VK_NULL_HANDLE;
		VmaAllocation staging_alloc = VK_NULL_HANDLE;
		if (auto e = check(vmaCreateBuffer(context.allocator(), &stagingInfo, &stagingAllocInfo,
		                                   &staging, &staging_alloc, nullptr),
		                   "create staging buffer for image upload", ErrorCode::Resource))
		{
			return std::unexpected(*e);
		}

		void* mapped = nullptr;
		if (auto e = check(vmaMapMemory(context.allocator(), staging_alloc, &mapped),
		                   "map image staging buffer", ErrorCode::Resource))
		{
			vmaDestroyBuffer(context.allocator(), staging, staging_alloc);
			return std::unexpected(*e);
		}
		std::memcpy(mapped, pixels, static_cast<std::size_t>(size));
		vmaUnmapMemory(context.allocator(), staging_alloc);

		return std::pair{ staging, staging_alloc };
	}

private:
	// CPU-side half of a wait: block while the worker is still decoding, then
	// surface a failed decode as the error it is.
	std::expected<void, Error> wait_submitted_()
	{
		if (upload_state_.load() == UploadState::Pending ||
		    upload_state_.load() == UploadState::Failed)
		{
			std::unique_lock lock(upload_mutex_);
			upload_cv_.wait(lock, [&] { return upload_state_.load() != UploadState::Pending; });
			if (upload_state_.load() == UploadState::Failed)
			{
				return std::unexpected(err_resource(upload_error_));
			}
		}
		return {};
	}

	// Staging upload of mip 0, then either the blit chain filling the rest of
	// the levels or a single transition to SHADER_READ_ONLY. One synchronous
	// submit; the async UploadManager replaces the transport, not the recording.
	std::expected<void, Error> upload_pixels(Context& context, const void* pixels,
	                                         std::uint32_t mips)
	{
		auto staging = create_filled_staging(context, pixels);
		if (!staging)
		{
			return std::unexpected(staging.error());
		}
		auto [buffer, allocation] = *staging;

		auto submitted = immediate_submit(context, [&](VkCommandBuffer cmd) {
			record_upload_commands(cmd, buffer, mips);
		});

		vmaDestroyBuffer(context.allocator(), buffer, allocation);
		if (!submitted)
		{
			return std::unexpected(submitted.error());
		}

		mark_has_contents(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		return {};
	}

	// Copy staging into mip 0, then either blit the mip chain or transition mip 0
	// to SHADER_READ_ONLY. Shared by the first upload and hot reload — the image
	// must already be in TRANSFER_DST across all mips when this runs.
	void record_copy_and_finalize_(VkCommandBuffer cmd, VkBuffer staging, std::uint32_t mips)
	{
		VkBufferImageCopy region{
			.bufferOffset = 0,
			.bufferRowLength = 0,
			.bufferImageHeight = 0,
			.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
			.imageOffset = { 0, 0, 0 },
			.imageExtent = { width_, height_, 1 }
		};
		vkCmdCopyBufferToImage(cmd, staging, image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		if (mips > 1)
		{
			record_mip_generation(cmd, image_, width_, height_, mips);
		}
		else
		{
			record_image_transition(cmd, image_,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
		}
	}

	// The classic blit cascade: level i-1 (TRANSFER_DST after the copy above)
	// becomes TRANSFER_SRC, blits into level i, and retires to
	// SHADER_READ_ONLY; the last level retires after the loop. Every level ends
	// in SHADER_READ_ONLY.
	static void record_mip_generation(VkCommandBuffer cmd, VkImage image,
	                                  std::uint32_t width, std::uint32_t height,
	                                  std::uint32_t mips)
	{
		std::int32_t mip_width = static_cast<std::int32_t>(width);
		std::int32_t mip_height = static_cast<std::int32_t>(height);

		for (std::uint32_t i = 1; i < mips; ++i)
		{
			record_image_transition(cmd, image,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 1);

			const std::int32_t next_width = mip_width > 1 ? mip_width / 2 : 1;
			const std::int32_t next_height = mip_height > 1 ? mip_height / 2 : 1;

			VkImageBlit blit{
				.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1 },
				.srcOffsets = { { 0, 0, 0 }, { mip_width, mip_height, 1 } },
				.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1 },
				.dstOffsets = { { 0, 0, 0 }, { next_width, next_height, 1 } }
			};
			vkCmdBlitImage(cmd,
				image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1, &blit, VK_FILTER_LINEAR);

			record_image_transition(cmd, image,
				VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
				VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 1);

			mip_width = next_width;
			mip_height = next_height;
		}

		record_image_transition(cmd, image,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			VK_IMAGE_ASPECT_COLOR_BIT, mips - 1, 1);
	}

	std::shared_ptr<Context> context_;
	VkImage image_ = VK_NULL_HANDLE;
	VmaAllocation allocation_ = VK_NULL_HANDLE;
	VkImageView view_ = VK_NULL_HANDLE;
	Format format_ = Format::RGBA8;
	std::uint32_t width_ = 0;
	std::uint32_t height_ = 0;
	std::uint32_t mip_levels_ = 1;
	std::atomic<bool> has_contents_{ false };
	VkImageLayout layout_ = VK_IMAGE_LAYOUT_UNDEFINED;

	// Async upload state, written by the upload worker, read by the main
	// thread. The cv/mutex pair backs the CPU-side waits; the timeline serial
	// backs the GPU-side ones.
	std::atomic<UploadState> upload_state_{ UploadState::None };
	std::atomic<std::uint64_t> upload_serial_{ 0 };
	std::mutex upload_mutex_;
	std::condition_variable upload_cv_;
	std::string upload_error_;
};
