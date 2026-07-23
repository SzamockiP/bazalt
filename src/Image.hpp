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
inline void record_image_transition(
    VkCommandBuffer cmd,
    VkImage image,
    VkImageLayout oldLayout,
    VkImageLayout newLayout,
    VkAccessFlags srcAccess,
    VkAccessFlags dstAccess,
    VkPipelineStageFlags srcStage,
    VkPipelineStageFlags dstStage,
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT,
    std::uint32_t baseMip = 0,
    std::uint32_t mipCount = 1,
    std::uint32_t layerCount = 1)
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
            .layerCount = layerCount}};
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

// A GPU image: VkImage + view + format. Nothing else — the sampler it used to
// be fused with lives in the Context's cache, and how the image is *used*
// (sampled, rendered into, read back) is the caller's business, not baked in
// at creation.
class Image : public std::enable_shared_from_this<Image>
{
public:
    Image(
        std::shared_ptr<Context> context,
        VkImage image,
        VmaAllocation allocation,
        VkImageView view,
        Format format,
        std::uint32_t width,
        std::uint32_t height,
        std::uint32_t mip_levels,
        std::uint32_t array_layers = 1,
        bool cube = false,
        VkImageView storage_view = VK_NULL_HANDLE)
        : context_(std::move(context)),
          image_(image),
          allocation_(allocation),
          view_(view),
          storage_view_(storage_view),
          format_(format),
          width_(width),
          height_(height),
          mip_levels_(mip_levels),
          array_layers_(array_layers),
          cube_(cube)
    {
    }

    // Deferred: an in-flight frame may still sample this image.
    ~Image()
    {
        if (context_)
        {
            context_->defer_destroy(
                [device = context_->device(),
                 allocator = context_->allocator(),
                 view = view_,
                 storage_view = storage_view_,
                 image = image_,
                 allocation = allocation_]
                {
                    if (view != VK_NULL_HANDLE)
                        vkDestroyImageView(device, view, nullptr);
                    // A separate 2D_ARRAY view exists only for cubemaps (storage
                    // binding can't use a CUBE view); non-cube images share view_.
                    if (storage_view != VK_NULL_HANDLE)
                        vkDestroyImageView(device, storage_view, nullptr);
                    if (image != VK_NULL_HANDLE && allocation != VK_NULL_HANDLE)
                    {
                        vmaDestroyImage(allocator, image, allocation);
                    }
                });
        }
    }

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    VkImage vk_image() const
    {
        return image_;
    }
    VkImageView view() const
    {
        return view_;
    }
    // The view a storage-image descriptor binds. For a cubemap this is a
    // separate 2D_ARRAY view (a CUBE view is illegal as storage); every other
    // image samples and stores through the same view.
    VkImageView storage_view() const
    {
        return storage_view_ != VK_NULL_HANDLE ? storage_view_ : view_;
    }
    Format format() const
    {
        return format_;
    }
    std::uint32_t width() const
    {
        return width_;
    }
    std::uint32_t height() const
    {
        return height_;
    }
    std::uint32_t mip_levels() const
    {
        return mip_levels_;
    }
    std::uint32_t array_layers() const
    {
        return array_layers_;
    }
    bool is_cube() const
    {
        return cube_;
    }

    // "Has the GPU ever been given contents for this image" — uploaded, copied
    // from an array, or rendered into. Readback and sampling of a virgin image
    // are refused rather than returning driver-defined garbage (0.4.1 contract).
    bool has_contents() const
    {
        return has_contents_.load();
    }
    VkImageLayout current_layout() const
    {
        return layout_;
    }
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

    enum class UploadState
    {
        None,
        Pending,
        Submitted,
        Failed
    };

    // Non-blocking: is the pixel data on the GPU?
    bool ready() const
    {
        switch (upload_state_.load())
        {
            case UploadState::None:
                return has_contents_.load();
            case UploadState::Pending:
                return false;
            case UploadState::Failed:
                return false;
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
                .pValues = &serial};
            if (auto e = check(
                    vkWaitSemaphores(context_->device(), &waitInfo, UINT64_MAX),
                    "wait for image upload",
                    ErrorCode::Resource))
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
    void set_upload_pending()
    {
        upload_state_.store(UploadState::Pending);
    }
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
        if (feat & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)
            usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
        if (feat & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT)
            usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        if (feat & VK_FORMAT_FEATURE_TRANSFER_DST_BIT)
            usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if (feat & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)
            usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (feat & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
            usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        if (feat & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)
            usage |= VK_IMAGE_USAGE_STORAGE_BIT;
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
        constexpr VkFormatFeatureFlags needed = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT |
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
    // render-target attachments and array/cubemap uploads. `array_layers > 1`
    // makes it a texture array (view 2D_ARRAY); `cube` makes it a cubemap (6
    // layers, view CUBE + a second 2D_ARRAY view for storage-image writes).
    static std::expected<std::shared_ptr<Image>, Error> create_empty(
        Context& context,
        std::uint32_t width,
        std::uint32_t height,
        Format format,
        std::uint32_t mip_levels = 1,
        std::uint32_t array_layers = 1,
        bool cube = false)
    {
        if (width == 0 || height == 0)
        {
            return std::unexpected(
                err_resource(std::format("Image dimensions must be non-zero, got {}x{}", width, height)));
        }
        const FormatInfo info = format_info(format);
        const VkImageAspectFlags aspect = info.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        const VkImageViewType view_type =
            cube ? VK_IMAGE_VIEW_TYPE_CUBE : (array_layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D);

        VkImageCreateInfo imageInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = static_cast<VkImageCreateFlags>(cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0),
            .imageType = VK_IMAGE_TYPE_2D,
            .format = info.vk,
            .extent = {width, height, 1},
            .mipLevels = mip_levels,
            .arrayLayers = array_layers,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = usage_for(context, format),
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        VkImage image = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        if (auto e = check(
                vmaCreateImage(context.allocator(), &imageInfo, &allocInfo, &image, &allocation, nullptr),
                std::format("create {} image", format_name(format)),
                ErrorCode::Resource))
        {
            return std::unexpected(*e);
        }

        VkImageViewCreateInfo viewInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = image,
            .viewType = view_type,
            .format = info.vk,
            .components = {},
            .subresourceRange = {aspect, 0, mip_levels, 0, array_layers}};

        VkImageView view = VK_NULL_HANDLE;
        if (auto e = check(
                vkCreateImageView(context.device(), &viewInfo, nullptr, &view),
                std::format("create {} image view", format_name(format)),
                ErrorCode::Resource))
        {
            vmaDestroyImage(context.allocator(), image, allocation);
            return std::unexpected(*e);
        }

        // Storage images may not be bound through a CUBE view. Give a cubemap a
        // parallel 2D_ARRAY view so compute can write its faces (procedural
        // skyboxes); sampling still goes through the CUBE view above.
        VkImageView storage_view = VK_NULL_HANDLE;
        if (cube)
        {
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
            if (auto e = check(
                    vkCreateImageView(context.device(), &viewInfo, nullptr, &storage_view),
                    std::format("create {} storage view", format_name(format)),
                    ErrorCode::Resource))
            {
                vkDestroyImageView(context.device(), view, nullptr);
                vmaDestroyImage(context.allocator(), image, allocation);
                return std::unexpected(*e);
            }
        }

        return std::make_shared<Image>(
            context.shared_from_this(),
            image,
            allocation,
            view,
            format,
            width,
            height,
            mip_levels,
            array_layers,
            cube,
            storage_view);
    }

    // From caller-provided pixels (numpy arrays land here). UNORM by default at
    // the binding layer: arrays are data, files are pictures. One mip by default
    // — data images don't want surprise filtering — but `mipmaps` opts a numpy
    // texture into the full chain (falling back to one level when the format
    // can't be blitted, exactly like load_from_file).
    static std::expected<std::shared_ptr<Image>, Error> create_from_pixels(
        Context& context,
        const void* pixels,
        std::uint32_t width,
        std::uint32_t height,
        Format format,
        bool mipmaps = false)
    {
        const std::uint32_t mips = mipmaps && can_generate_mips(context, format) ? full_mip_count(width, height) : 1;
        auto image = create_empty(context, width, height, format, mips);
        if (!image)
        {
            return image;
        }
        if (auto r = (*image)->upload_pixels(context, pixels, mips); !r)
        {
            return std::unexpected(r.error());
        }
        return image;
    }

    // A texture array or cubemap from caller-provided pixels: `layers` images of
    // width×height laid out back to back (layer 0, layer 1, …). UNORM data, one
    // mip by default — same policy as create_from_pixels; `mipmaps` opts into the
    // full chain (one blit per level across every layer). `cube` picks the CUBE
    // view; callers (the binding layer) enforce layers==6 and square faces first.
    static std::expected<std::shared_ptr<Image>, Error> create_layered_from_pixels(
        Context& context,
        const void* pixels,
        std::uint32_t width,
        std::uint32_t height,
        std::uint32_t layers,
        bool cube,
        Format format,
        bool mipmaps = false)
    {
        const std::uint32_t mips = mipmaps && can_generate_mips(context, format) ? full_mip_count(width, height) : 1;
        auto image = create_empty(context, width, height, format, mips, layers, cube);
        if (!image)
        {
            return image;
        }
        if (auto r = (*image)->upload_pixels(context, pixels, mips); !r)
        {
            return std::unexpected(r.error());
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
        const VkDeviceSize size = static_cast<VkDeviceSize>(width_) * height_ * info.bytes_per_pixel;
        const VkImageAspectFlags aspect = info.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;

        VkBufferCreateInfo bufferInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = size,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr};

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_GPU_TO_CPU;
        allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

        VkBuffer staging = VK_NULL_HANDLE;
        VmaAllocation staging_alloc = VK_NULL_HANDLE;
        if (auto e = check(
                vmaCreateBuffer(context_->allocator(), &bufferInfo, &allocInfo, &staging, &staging_alloc, nullptr),
                "create readback buffer",
                ErrorCode::Resource))
        {
            return std::unexpected(*e);
        }

        auto submitted = immediate_submit(
            *context_,
            [&](VkCommandBuffer cmd)
            {
                record_image_transition(
                    cmd,
                    image_,
                    layout_,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_ACCESS_MEMORY_WRITE_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    aspect,
                    0,
                    mip_levels_);

                VkBufferImageCopy region{
                    .bufferOffset = 0,
                    .bufferRowLength = 0,
                    .bufferImageHeight = 0,
                    .imageSubresource = {aspect, 0, 0, 1},
                    .imageOffset = {0, 0, 0},
                    .imageExtent = {width_, height_, 1}};
                vkCmdCopyImageToBuffer(cmd, image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging, 1, &region);

                record_image_transition(
                    cmd,
                    image_,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    layout_,
                    VK_ACCESS_TRANSFER_READ_BIT,
                    VK_ACCESS_MEMORY_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    aspect,
                    0,
                    mip_levels_);
            });
        if (!submitted)
        {
            vmaDestroyBuffer(context_->allocator(), staging, staging_alloc);
            return std::unexpected(submitted.error());
        }

        std::vector<std::byte> out(static_cast<std::size_t>(size));
        void* mapped = nullptr;
        if (auto e = check(
                vmaMapMemory(context_->allocator(), staging_alloc, &mapped),
                "map readback buffer memory",
                ErrorCode::Resource))
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
        record_image_transition(
            cmd,
            image_,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            0,
            mips,
            array_layers_);

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
        record_image_transition(
            cmd,
            image_,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_ACCESS_SHADER_READ_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            0,
            mips,
            array_layers_);

        record_copy_and_finalize_(cmd, staging, mips);
    }

    // Standalone mip generation for cmd.generate_mipmaps(): mip 0 already holds
    // its final contents (in `src_layout` — GENERAL from a compute write, or
    // SHADER_READ_ONLY from an upload / prior bake), and the rest of the chain is
    // (re)generated by blitting down. Every level, every layer ends in
    // SHADER_READ_ONLY. The caller (CommandBuffer) has already checked
    // mip_levels_ > 1 and that the format can be blitted. The mip-0 transition's
    // src scope (stage + access) doubles as the RAW/WAR barrier against whatever
    // produced mip 0.
    void record_generate_mipmaps(
        VkCommandBuffer cmd,
        VkImageLayout src_layout,
        VkPipelineStageFlags src_stage,
        VkAccessFlags src_access)
    {
        // mip 0 -> TRANSFER_SRC (this transition is also the barrier waiting on
        // the producer of mip 0).
        record_image_transition(
            cmd,
            image_,
            src_layout,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            src_access,
            VK_ACCESS_TRANSFER_READ_BIT,
            src_stage,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            0,
            1,
            array_layers_);

        // The other levels hold nothing worth keeping -> discard into TRANSFER_DST.
        record_image_transition(
            cmd,
            image_,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            1,
            mip_levels_ - 1,
            array_layers_);

        std::int32_t mip_width = static_cast<std::int32_t>(width_);
        std::int32_t mip_height = static_cast<std::int32_t>(height_);

        for (std::uint32_t i = 1; i < mip_levels_; ++i)
        {
            const std::int32_t next_width = mip_width > 1 ? mip_width / 2 : 1;
            const std::int32_t next_height = mip_height > 1 ? mip_height / 2 : 1;

            // One blit with layerCount = array_layers_ fills level i for every
            // face/layer at once (they share mip dimensions).
            VkImageBlit blit{
                .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, array_layers_},
                .srcOffsets = {{0, 0, 0}, {mip_width, mip_height, 1}},
                .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, array_layers_},
                .dstOffsets = {{0, 0, 0}, {next_width, next_height, 1}}};
            vkCmdBlitImage(
                cmd,
                image_,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                image_,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &blit,
                VK_FILTER_LINEAR);

            // Level i-1 is done being read from -> retire to SHADER_READ_ONLY.
            record_image_transition(
                cmd,
                image_,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_ACCESS_TRANSFER_READ_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                i - 1,
                1,
                array_layers_);

            if (i + 1 < mip_levels_)
            {
                // Level i becomes the source for the next blit.
                record_image_transition(
                    cmd,
                    image_,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    i,
                    1,
                    array_layers_);
            }
            else
            {
                // The last level retires straight to SHADER_READ_ONLY.
                record_image_transition(
                    cmd,
                    image_,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    i,
                    1,
                    array_layers_);
            }

            mip_width = next_width;
            mip_height = next_height;
        }
    }

    // Creates and fills a staging buffer for this image's mip 0, all layers
    // (one image for a plain 2D texture; `array_layers_` images laid out back to
    // back for an array/cubemap).
    std::expected<std::pair<VkBuffer, VmaAllocation>, Error> create_filled_staging(Context& context, const void* pixels)
    {
        const FormatInfo info = format_info(format_);
        const VkDeviceSize size = static_cast<VkDeviceSize>(width_) * height_ * info.bytes_per_pixel * array_layers_;

        VkBufferCreateInfo stagingInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = size,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr};
        VmaAllocationCreateInfo stagingAllocInfo{};
        stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
        stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

        VkBuffer staging = VK_NULL_HANDLE;
        VmaAllocation staging_alloc = VK_NULL_HANDLE;
        if (auto e = check(
                vmaCreateBuffer(
                    context.allocator(), &stagingInfo, &stagingAllocInfo, &staging, &staging_alloc, nullptr),
                "create staging buffer for image upload",
                ErrorCode::Resource))
        {
            return std::unexpected(*e);
        }

        void* mapped = nullptr;
        if (auto e = check(
                vmaMapMemory(context.allocator(), staging_alloc, &mapped),
                "map image staging buffer",
                ErrorCode::Resource))
        {
            vmaDestroyBuffer(context.allocator(), staging, staging_alloc);
            return std::unexpected(*e);
        }
        std::memcpy(mapped, pixels, static_cast<std::size_t>(size));
        vmaUnmapMemory(context.allocator(), staging_alloc);

        return std::pair{staging, staging_alloc};
    }

private:
    // CPU-side half of a wait: block while the worker is still decoding, then
    // surface a failed decode as the error it is.
    std::expected<void, Error> wait_submitted_()
    {
        if (upload_state_.load() == UploadState::Pending || upload_state_.load() == UploadState::Failed)
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
    std::expected<void, Error> upload_pixels(Context& context, const void* pixels, std::uint32_t mips)
    {
        auto staging = create_filled_staging(context, pixels);
        if (!staging)
        {
            return std::unexpected(staging.error());
        }
        auto [buffer, allocation] = *staging;

        auto submitted =
            immediate_submit(context, [&](VkCommandBuffer cmd) { record_upload_commands(cmd, buffer, mips); });

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
        // One region with layerCount = array_layers_ copies every layer: the
        // staging buffer holds them consecutively, exactly what a layered copy
        // reads from bufferOffset 0.
        VkBufferImageCopy region{
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, array_layers_},
            .imageOffset = {0, 0, 0},
            .imageExtent = {width_, height_, 1}};
        vkCmdCopyBufferToImage(cmd, staging, image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        if (mips > 1)
        {
            record_mip_generation(cmd, image_, width_, height_, mips, array_layers_);
        }
        else
        {
            record_image_transition(
                cmd,
                image_,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                0,
                1,
                array_layers_);
        }
    }

    // The classic blit cascade: level i-1 (TRANSFER_DST after the copy above)
    // becomes TRANSFER_SRC, blits into level i, and retires to
    // SHADER_READ_ONLY; the last level retires after the loop. Every level ends
    // in SHADER_READ_ONLY.
    static void record_mip_generation(
        VkCommandBuffer cmd,
        VkImage image,
        std::uint32_t width,
        std::uint32_t height,
        std::uint32_t mips,
        std::uint32_t layers = 1)
    {
        std::int32_t mip_width = static_cast<std::int32_t>(width);
        std::int32_t mip_height = static_cast<std::int32_t>(height);

        // Every layer shares mip dimensions, so one blit with layerCount = layers
        // generates the whole level for all faces/layers at once.
        for (std::uint32_t i = 1; i < mips; ++i)
        {
            record_image_transition(
                cmd,
                image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                i - 1,
                1,
                layers);

            const std::int32_t next_width = mip_width > 1 ? mip_width / 2 : 1;
            const std::int32_t next_height = mip_height > 1 ? mip_height / 2 : 1;

            VkImageBlit blit{
                .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, layers},
                .srcOffsets = {{0, 0, 0}, {mip_width, mip_height, 1}},
                .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, layers},
                .dstOffsets = {{0, 0, 0}, {next_width, next_height, 1}}};
            vkCmdBlitImage(
                cmd,
                image,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &blit,
                VK_FILTER_LINEAR);

            record_image_transition(
                cmd,
                image,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_ACCESS_TRANSFER_READ_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                i - 1,
                1,
                layers);

            mip_width = next_width;
            mip_height = next_height;
        }

        record_image_transition(
            cmd,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            mips - 1,
            1,
            layers);
    }

    std::shared_ptr<Context> context_;
    VkImage image_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VkImageView view_ = VK_NULL_HANDLE;
    VkImageView storage_view_ = VK_NULL_HANDLE; // cube only; else null → view()
    Format format_ = Format::RGBA8;
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    std::uint32_t mip_levels_ = 1;
    std::uint32_t array_layers_ = 1;
    bool cube_ = false;
    std::atomic<bool> has_contents_{false};
    VkImageLayout layout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    // Async upload state, written by the upload worker, read by the main
    // thread. The cv/mutex pair backs the CPU-side waits; the timeline serial
    // backs the GPU-side ones.
    std::atomic<UploadState> upload_state_{UploadState::None};
    std::atomic<std::uint64_t> upload_serial_{0};
    std::mutex upload_mutex_;
    std::condition_variable upload_cv_;
    std::string upload_error_;
};
