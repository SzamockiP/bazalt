#pragma once
#include <volk.h>
#include <vk_mem_alloc.h>

#include <array>
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
#include "ResourceTracker.hpp"

// Layout transition helper shared by uploads, mip generation and readback.
// Formerly a private of Texture; RenderTarget grew its own copy, which is
// exactly the drift this ends.
inline void record_image_transition(
    const VolkDeviceTable& vk,
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
    std::uint32_t layerCount = 1,
    // Last and defaulted so every existing call site keeps its meaning. Only a
    // per-subresource transition (a split image, 0.18) ever names it.
    std::uint32_t baseLayer = 0)
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
            .baseArrayLayer = baseLayer,
            .layerCount = layerCount}};
    vk.vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

// Forward declaration: the copy/clear recorders below take Images, and they live
// here so that cmd.copy_image() records one call instead of thirty lines.
class Image;
void record_image_copy(
    const VolkDeviceTable& vk,
    VkCommandBuffer cmd,
    Image& src,
    Image& dst,
    VkImageLayout src_layout);
void record_image_blit(
    const VolkDeviceTable& vk,
    VkCommandBuffer cmd,
    Image& src,
    Image& dst,
    VkImageLayout src_layout,
    VkFilter filter);
void record_image_clear(const VolkDeviceTable& vk, VkCommandBuffer cmd, Image& image, std::array<float, 4> color);

// The layout of every (layer, mip) of one image, with a fast path for the case
// where they all agree.
//
// Vulkan tracks layout per subresource, and until 0.18 bazalt held ONE layout
// per Image. That was right while every write covered the whole image, and it
// stopped being right the moment a pass could render into one mip
// (`target.mip(m)`, 0.13): the whole image was then marked as rendered, so
// sampling a partially rendered target handed a stale oldLayout to the next
// barrier. The documented workaround was "render every layer and every mip
// before you sample", which rules out exactly the two things that make a mip
// chain worth having — a render into one level, and a copy into another.
//
// The collapse is what keeps this free for the 99% case. An image whose
// subresources all share a layout stores one value and emits one barrier, and a
// split image collapses back the moment the last odd subresource catches up. So
// rendering six cube faces one at a time ends uniform again, and the sample
// that follows costs exactly what it did before this existed.
class SubresourceLayouts
{
public:
    void configure(std::uint32_t layers, std::uint32_t mips)
    {
        layers_ = layers;
        mips_ = mips;
    }

    VkImageLayout get(std::uint32_t layer, std::uint32_t mip) const
    {
        if (split_.empty())
        {
            return uniform_;
        }
        return split_[index_(layer, mip)];
    }

    // The one layout the whole image is in, or nullopt when the subresources
    // disagree. A caller that can only emit one barrier asks this first.
    std::optional<VkImageLayout> uniform() const
    {
        return split_.empty() ? std::optional<VkImageLayout>{uniform_} : std::nullopt;
    }

    void set_all(VkImageLayout layout)
    {
        split_.clear();
        uniform_ = layout;
    }

    void set_range(
        VkImageLayout layout,
        std::uint32_t base_layer,
        std::uint32_t layer_count,
        std::uint32_t base_mip,
        std::uint32_t mip_count)
    {
        // VK_REMAINING_* are what a caller passes for "the rest of them", and
        // they arrive here as huge numbers rather than as a flag.
        const std::uint32_t last_layer = (std::ranges::min)(base_layer + layer_count, layers_);
        const std::uint32_t last_mip = (std::ranges::min)(base_mip + mip_count, mips_);

        if (base_layer == 0 && last_layer >= layers_ && base_mip == 0 && last_mip >= mips_)
        {
            set_all(layout);
            return;
        }
        if (split_.empty())
        {
            if (uniform_ == layout)
            {
                return; // nothing to split over
            }
            split_.assign(static_cast<std::size_t>(layers_) * mips_, uniform_);
        }
        for (std::uint32_t layer = base_layer; layer < last_layer; ++layer)
        {
            for (std::uint32_t mip = base_mip; mip < last_mip; ++mip)
            {
                split_[index_(layer, mip)] = layout;
            }
        }
        // Collapse the moment the odd one out catches up, so a target rendered
        // layer by layer is back to one barrier by the time it is sampled.
        if (std::ranges::all_of(split_, [&](VkImageLayout l) { return l == split_.front(); }))
        {
            set_all(split_.front());
        }
    }

    // Calls `fn(layout, layer, mip)` for each subresource whose layout differs
    // from `uniform()`. Only reached on a split image: the uniform case has one
    // barrier and never comes here.
    template <typename Fn>
    void for_each(Fn&& fn) const
    {
        for (std::uint32_t layer = 0; layer < layers_; ++layer)
        {
            for (std::uint32_t mip = 0; mip < mips_; ++mip)
            {
                fn(split_[index_(layer, mip)], layer, mip);
            }
        }
    }

private:
    std::size_t index_(std::uint32_t layer, std::uint32_t mip) const
    {
        return static_cast<std::size_t>(layer) * mips_ + mip;
    }

    std::uint32_t layers_ = 1;
    std::uint32_t mips_ = 1;
    VkImageLayout uniform_ = VK_IMAGE_LAYOUT_UNDEFINED;
    // Empty means "uniform_ covers everything" — the state an image lives in
    // unless something writes a strict subset of it.
    std::vector<VkImageLayout> split_;
};

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
        VkImageView storage_view = VK_NULL_HANDLE,
        VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT)
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
          cube_(cube),
          samples_(samples)
    {
        // The layout state needs the geometry to index a split; it starts
        // uniform-UNDEFINED, so nothing is allocated until something writes a
        // strict subset.
        layouts_.configure(array_layers, mip_levels);
    }

    // Deferred: an in-flight frame may still sample this image.
    ~Image()
    {
        if (context_)
        {
            context_->defer_destroy(
                [vk = &context_->vk(),
                 device = context_->device(),
                 allocator = context_->allocator(),
                 view = view_,
                 storage_view = storage_view_,
                 image = image_,
                 allocation = allocation_]
                {
                    if (view != VK_NULL_HANDLE)
                        vk->vkDestroyImageView(device, view, nullptr);
                    // A separate 2D_ARRAY view exists only for cubemaps (storage
                    // binding can't use a CUBE view); non-cube images share view_.
                    if (storage_view != VK_NULL_HANDLE)
                        vk->vkDestroyImageView(device, storage_view, nullptr);
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
    // The VkFormat this image was actually created with. Equal to
    // format_info(format()).vk for every format except DEPTH_STENCIL, which the
    // device resolves — so attachment infos, views and pipeline formats read it
    // from the image rather than re-deriving it and disagreeing.
    VkFormat vk_format() const
    {
        return context_ ? context_->vk_format(format_) : format_info(format_).vk;
    }
    // Colour, depth, or depth+stencil. One source for the views, the barriers
    // and the copies.
    VkImageAspectFlags aspect() const
    {
        return aspect_mask_for(vk_format());
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
    // MSAA sample count as a plain int (1/2/4/…). >1 means this is a multisampled
    // attachment owned by a RenderTarget: it's rendered into and resolved out, so
    // it cannot be sampled, uploaded to, or read back — read() refuses it.
    std::uint32_t samples() const
    {
        return static_cast<std::uint32_t>(samples_);
    }

    // "Has the GPU ever been given contents for this image" — uploaded, copied
    // from an array, or rendered into. Readback and sampling of a virgin image
    // are refused rather than returning driver-defined garbage (0.4.1 contract).
    bool has_contents() const
    {
        return has_contents_.load();
    }

    // The layout the whole image is in, or nullopt when its subresources
    // disagree — which happens exactly when a pass, a copy or an update wrote a
    // strict subset of it. A caller that emits ONE barrier must ask this and
    // fall back to per-subresource barriers on nullopt; `current_layout()` is
    // the convenience for the callers that only ever see whole images.
    std::optional<VkImageLayout> uniform_layout() const
    {
        return layouts_.uniform();
    }
    VkImageLayout layout_of(std::uint32_t layer, std::uint32_t mip) const
    {
        return layouts_.get(layer, mip);
    }
    VkImageLayout current_layout() const
    {
        return layouts_.get(0, 0);
    }

    void mark_has_contents(VkImageLayout layout)
    {
        layouts_.set_all(layout);
        has_contents_.store(true);
    }

    // The same statement about one part of the image. Used by a pass that
    // rendered into a single layer or mip, and by a copy that filled one
    // subresource: marking the whole image would be the 0.13 bug — a stale
    // oldLayout on the next barrier, reported as a validation error a long way
    // from its cause.
    //
    // has_contents stays whole-image on purpose. It answers "is reading this
    // meaningful at all", which is a question about the image, and a per-part
    // version would refuse a legitimate read of a fully written image whose
    // parts were written separately.
    void mark_subresource_contents(
        VkImageLayout layout,
        std::uint32_t base_layer,
        std::uint32_t layer_count,
        std::uint32_t base_mip,
        std::uint32_t mip_count)
    {
        layouts_.set_range(layout, base_layer, layer_count, base_mip, mip_count);
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
                    context_->vk().vkWaitSemaphores(context_->device(), &waitInfo, UINT64_MAX),
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
        vkGetPhysicalDeviceFormatProperties(context.physical_device(), context.vk_format(format), &props);
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

    // The usage an image of this format and sample count actually gets.
    //
    // Two narrowings on top of "everything the device supports". A multisampled
    // attachment is only rendered into and resolved out, so it keeps just the
    // attachment usage: STORAGE on a multisample image needs a feature we do not
    // enable, and SAMPLED/TRANSFER are dead weight (you sample the resolve).
    //
    // A combined depth/stencil image drops SAMPLED and STORAGE for a harder
    // reason: its view carries both aspects, and Vulkan forbids sampling
    // through such a view. Keeping the usage would make every DEPTH_STENCIL
    // view illegal at creation, which is a validation error at the target's
    // constructor rather than at the sample that was never going to work.
    static VkImageUsageFlags usage_for_image(Context& context, Format format, VkSampleCountFlagBits samples)
    {
        VkImageUsageFlags usage = usage_for(context, format);
        if (samples != VK_SAMPLE_COUNT_1_BIT)
        {
            return usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
        }
        if (has_stencil(context.vk_format(format)))
        {
            usage &= ~static_cast<VkImageUsageFlags>(VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
        }
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
        vkGetPhysicalDeviceFormatProperties(context.physical_device(), context.vk_format(format), &props);
        constexpr VkFormatFeatureFlags needed = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT |
                                                VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
        return (props.optimalTilingFeatures & needed) == needed;
    }

    // Whether this GPU can blit `from` into `to`. A blit is a transfer command
    // and yet it is NOT universally available: BLIT_SRC and BLIT_DST are format
    // features, and a linear filter needs the source to be filterable on top.
    // Same shape as can_generate_mips, which asks the same question of one
    // format — and the same reason for asking it here: the alternative is a
    // validation message about VkFormatFeatureFlags instead of a sentence
    // naming the two formats.
    static bool can_blit(Context& context, Format from, Format to)
    {
        VkFormatProperties src_props{};
        vkGetPhysicalDeviceFormatProperties(context.physical_device(), context.vk_format(from), &src_props);
        VkFormatProperties dst_props{};
        vkGetPhysicalDeviceFormatProperties(context.physical_device(), context.vk_format(to), &dst_props);

        constexpr VkFormatFeatureFlags src_needed = VK_FORMAT_FEATURE_BLIT_SRC_BIT |
                                                    VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
        return (src_props.optimalTilingFeatures & src_needed) == src_needed &&
               (dst_props.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) != 0;
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
        bool cube = false,
        VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT)
    {
        if (width == 0 || height == 0)
        {
            return std::unexpected(
                err_resource(std::format("Image dimensions must be non-zero, got {}x{}", width, height)));
        }
        // A multisampled image is an MSAA attachment and nothing else. It can be a
        // layered attachment (MSAA render-to-layer resolves per layer), but it can't
        // carry mips (no blitting between sample counts) or be a cubemap (you sample
        // the single-sample resolve, never a multisampled cube view). Fail here
        // rather than at vkCreateImage.
        if (samples != VK_SAMPLE_COUNT_1_BIT && (mip_levels != 1 || cube))
        {
            return std::unexpected(err_resource(
                "A multisampled image (samples>1) is a render-target attachment only: "
                "it cannot have mipmaps or be a cubemap"));
        }
        const FormatInfo info = format_info(format);
        const VkFormat vk_fmt = context.vk_format(format);
        if (vk_fmt == VK_FORMAT_UNDEFINED)
        {
            return std::unexpected(err_resource(std::format("This device supports no {} format", format_name(format))));
        }
        const VkImageAspectFlags aspect = aspect_mask_for(vk_fmt);
        const VkImageViewType view_type =
            cube ? VK_IMAGE_VIEW_TYPE_CUBE : (array_layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D);

        VkImageCreateInfo imageInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = static_cast<VkImageCreateFlags>(cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0),
            .imageType = VK_IMAGE_TYPE_2D,
            .format = vk_fmt,
            .extent = {width, height, 1},
            .mipLevels = mip_levels,
            .arrayLayers = array_layers,
            .samples = samples,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            // A multisampled attachment is only rendered into and resolved out, so
            // it keeps just the attachment usage: STORAGE on a multisample image
            // needs a feature we don't enable, and SAMPLED/TRANSFER are dead weight
            // (you sample the single-sample resolve, never this).
            .usage = usage_for_image(context, format, samples),
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
            .format = vk_fmt,
            .components = {},
            .subresourceRange = {aspect, 0, mip_levels, 0, array_layers}};

        VkImageView view = VK_NULL_HANDLE;
        if (auto e = check(
                context.vk().vkCreateImageView(context.device(), &viewInfo, nullptr, &view),
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
                    context.vk().vkCreateImageView(context.device(), &viewInfo, nullptr, &storage_view),
                    std::format("create {} storage view", format_name(format)),
                    ErrorCode::Resource))
            {
                context.vk().vkDestroyImageView(context.device(), view, nullptr);
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
            storage_view,
            samples);
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
    //
    // `all_layers` reads every array layer back to back (layer 0, layer 1, …),
    // which is exactly the layout create_layered_from_pixels expects — that is
    // what makes ctx_b.create_image(image_from_ctx_a) a cubemap on the other
    // side too. Python's image.read() keeps layer 0 only: a numpy array has no
    // place to put the cube-ness, so returning six faces stacked would be a
    // shape the caller has to guess at.
    std::expected<std::vector<std::byte>, Error> read(bool all_layers = false)
    {
        const std::uint32_t layers = all_layers ? array_layers_ : 1;
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
        if (samples_ != VK_SAMPLE_COUNT_1_BIT)
        {
            return std::unexpected(err_resource(
                "read() called on a multisampled image; read the target's resolved "
                "single-sample attachment (target.color[i] / target.depth) instead"));
        }

        const FormatInfo info = format_info(format_);
        // A packed or combined format has no single numpy dtype: one pixel of
        // DEPTH_STENCIL is depth AND stencil, one pixel of R11G11B10F is three
        // channels sharing 32 bits. Refuse here with the reason rather than hand
        // back bytes that mean nothing.
        if (info.numpy_dtype[0] == '\0')
        {
            return std::unexpected(err_resource(
                std::format(
                    "read() is not available for {}: the format packs several values into one "
                    "texel, so there is no array shape that describes it. Render it into an "
                    "RGBA target, or use a shader to unpack it.",
                    format_name(format_))));
        }
        const VkDeviceSize size = static_cast<VkDeviceSize>(width_) * height_ * info.bytes_per_pixel * layers;
        const VkImageAspectFlags aspect = aspect_mask_for(vk_format());

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
                // Whatever state the subresources are in: one barrier when they
                // agree, one each when they do not. A single barrier can only
                // name one oldLayout, and naming the wrong one is a validation
                // error plus undefined contents — which is precisely what
                // reading a partially rendered image used to do.
                record_transition_into_(
                    cmd,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_ACCESS_MEMORY_WRITE_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    aspect);

                VkBufferImageCopy region{
                    .bufferOffset = 0,
                    .bufferRowLength = 0,
                    .bufferImageHeight = 0,
                    .imageSubresource = {aspect, 0, 0, layers},
                    .imageOffset = {0, 0, 0},
                    .imageExtent = {width_, height_, 1}};
                context_->vk().vkCmdCopyImageToBuffer(
                    cmd, image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging, 1, &region);

                // …and put every subresource back where it was, which is what
                // makes read() non-destructive on a partially written image.
                record_transition_out_of_(
                    cmd,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_ACCESS_TRANSFER_READ_BIT,
                    VK_ACCESS_MEMORY_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    aspect);
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
            context_->vk(),
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
            context_->vk(),
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
            context_->vk(),
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
            context_->vk(),
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
            context_->vk().vkCmdBlitImage(
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
                context_->vk(),
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
                    context_->vk(),
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
                    context_->vk(),
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

    // Move the WHOLE image into `to`, whatever state its subresources are in.
    // One barrier when they agree, one per subresource when they do not — a
    // barrier names a single oldLayout, so a split image cannot be covered by
    // one. The layout state is left alone: the caller is a scoped operation
    // (readback, copy) that puts the image back with record_transition_out_of_.
    void record_transition_into_(
        VkCommandBuffer cmd,
        VkImageLayout to,
        VkAccessFlags src_access,
        VkAccessFlags dst_access,
        VkPipelineStageFlags src_stage,
        VkPipelineStageFlags dst_stage,
        VkImageAspectFlags aspect) const
    {
        if (const auto uniform = layouts_.uniform())
        {
            record_image_transition(
                context_->vk(),
                cmd,
                image_,
                *uniform,
                to,
                src_access,
                dst_access,
                src_stage,
                dst_stage,
                aspect,
                0,
                mip_levels_,
                array_layers_);
            return;
        }
        layouts_.for_each(
            [&](VkImageLayout from, std::uint32_t layer, std::uint32_t mip)
            {
                record_image_transition(
                    context_->vk(),
                    cmd,
                    image_,
                    from,
                    to,
                    src_access,
                    dst_access,
                    src_stage,
                    dst_stage,
                    aspect,
                    mip,
                    1,
                    1,
                    layer);
            });
    }

    // The same walk backwards: every subresource returns to the layout it held.
    void record_transition_out_of_(
        VkCommandBuffer cmd,
        VkImageLayout from,
        VkAccessFlags src_access,
        VkAccessFlags dst_access,
        VkPipelineStageFlags src_stage,
        VkPipelineStageFlags dst_stage,
        VkImageAspectFlags aspect) const
    {
        if (const auto uniform = layouts_.uniform())
        {
            record_image_transition(
                context_->vk(),
                cmd,
                image_,
                from,
                *uniform,
                src_access,
                dst_access,
                src_stage,
                dst_stage,
                aspect,
                0,
                mip_levels_,
                array_layers_);
            return;
        }
        layouts_.for_each(
            [&](VkImageLayout to, std::uint32_t layer, std::uint32_t mip)
            {
                record_image_transition(
                    context_->vk(),
                    cmd,
                    image_,
                    from,
                    to,
                    src_access,
                    dst_access,
                    src_stage,
                    dst_stage,
                    aspect,
                    mip,
                    1,
                    1,
                    layer);
            });
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
        context_->vk().vkCmdCopyBufferToImage(cmd, staging, image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        if (mips > 1)
        {
            record_mip_generation(context_->vk(), cmd, image_, width_, height_, mips, array_layers_);
        }
        else
        {
            record_image_transition(
                context_->vk(),
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
        const VolkDeviceTable& vk,
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
                vk,
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
            vk.vkCmdBlitImage(
                cmd,
                image,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &blit,
                VK_FILTER_LINEAR);

            record_image_transition(
                vk,
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
            vk,
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

public:
    // Which Context this object belongs to. Multi-context (0.15) made "a
    // resource from the other Context" a reachable mistake, and its symptom
    // without a check is a driver crash or a validation message from Vulkan
    // rather than from bazalt; the binding layer compares owners at record time.
    const Context* owner() const
    {
        return context_.get();
    }

private:
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
    VkSampleCountFlagBits samples_ = VK_SAMPLE_COUNT_1_BIT;
    std::atomic<bool> has_contents_{false};
    SubresourceLayouts layouts_;

    // Async upload state, written by the upload worker, read by the main
    // thread. The cv/mutex pair backs the CPU-side waits; the timeline serial
    // backs the GPU-side ones.
    std::atomic<UploadState> upload_state_{UploadState::None};
    std::atomic<std::uint64_t> upload_serial_{0};
    std::mutex upload_mutex_;
    std::condition_variable upload_cv_;
    std::string upload_error_;
};

// Copy the whole of `src` into `dst` — same size, same format, every layer of
// mip 0. Both ends are transitioned around the transfer and left in
// SHADER_READ_ONLY, which is the layout a copy exists to produce. `src_layout`
// is where the source currently is (see cmd.copy_image).
//
// Only mip 0 is copied: the destination's other levels, if it has any, are
// regenerated with cmd.generate_mipmaps. Copying a chain would be N regions for
// a case that has not come up.
inline void record_image_copy(
    const VolkDeviceTable& vk,
    VkCommandBuffer cmd,
    Image& src,
    Image& dst,
    VkImageLayout src_layout)
{
    const std::uint32_t layers = src.array_layers();
    // Every level the two images share. 0.17 copied mip 0 only and called the
    // rest a ceiling ("a full chain is N regions for a case that has not come
    // up"). The case came up: a copy that leaves levels 1..N holding the
    // destination's old pixels is not a copy of the image, it is a copy of its
    // top level, and the difference shows up the moment anything samples with a
    // mip bias. N regions in one vkCmdCopyImage is the same one call.
    const std::uint32_t mips = (std::ranges::min)(src.mip_levels(), dst.mip_levels());

    record_image_transition(
        vk,
        cmd,
        src.vk_image(),
        src_layout,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_TRANSFER_READ_BIT,
        kAllShaderStages,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        src.aspect(),
        0,
        mips,
        layers);
    // The destination is overwritten in full, so its old contents are discarded
    // rather than waited for — UNDEFINED is legal from any layout.
    record_image_transition(
        vk,
        cmd,
        dst.vk_image(),
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        0,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        kAllShaderStages,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        dst.aspect(),
        0,
        mips,
        layers);

    std::vector<VkImageCopy> regions;
    regions.reserve(mips);
    for (std::uint32_t mip = 0; mip < mips; ++mip)
    {
        // Level dimensions floor at 1, which is what the mip chain of a
        // non-square image does on its short axis before the long one.
        const std::uint32_t w = (std::ranges::max)(src.width() >> mip, 1u);
        const std::uint32_t h = (std::ranges::max)(src.height() >> mip, 1u);
        regions.push_back(
            VkImageCopy{
                .srcSubresource =
                    {.aspectMask = src.aspect(), .mipLevel = mip, .baseArrayLayer = 0, .layerCount = layers},
                .srcOffset = {0, 0, 0},
                .dstSubresource =
                    {.aspectMask = dst.aspect(), .mipLevel = mip, .baseArrayLayer = 0, .layerCount = layers},
                .dstOffset = {0, 0, 0},
                .extent = {w, h, 1}});
    }
    vk.vkCmdCopyImage(
        cmd,
        src.vk_image(),
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        dst.vk_image(),
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        static_cast<std::uint32_t>(regions.size()),
        regions.data());

    for (Image* image : {&src, &dst})
    {
        record_image_transition(
            vk,
            cmd,
            image->vk_image(),
            image == &src ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            image == &src ? VK_ACCESS_TRANSFER_READ_BIT : VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            kAllShaderStages,
            image->aspect(),
            0,
            mips,
            layers);
    }
}

// A copy that RESIZES. vkCmdBlitImage rather than vkCmdCopyImage, so the two
// images need not share an extent, and the filter chooses how the sampling is
// done on the way.
//
// copy_image (above) needs identical size and format; generate_mipmaps scales,
// but only within one image. Downsampling for bloom, upscaling a compute result
// and building a thumbnail all fell in the gap, and every one of them was a
// full graphics pass with a fullscreen shader. This is the same blit cascade
// generate_mipmaps already runs, generalized to two images.
//
// Mip 0 across every layer: a blit chain between two images is a different
// question, and generate_mipmaps answers it on the destination.
inline void record_image_blit(
    const VolkDeviceTable& vk,
    VkCommandBuffer cmd,
    Image& src,
    Image& dst,
    VkImageLayout src_layout,
    VkFilter filter)
{
    const std::uint32_t layers = (std::ranges::min)(src.array_layers(), dst.array_layers());

    record_image_transition(
        vk,
        cmd,
        src.vk_image(),
        src_layout,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_TRANSFER_READ_BIT,
        kAllShaderStages,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        src.aspect(),
        0,
        1,
        layers);
    record_image_transition(
        vk,
        cmd,
        dst.vk_image(),
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        0,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        kAllShaderStages,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        dst.aspect(),
        0,
        1,
        layers);

    VkImageBlit region{
        .srcSubresource = {.aspectMask = src.aspect(), .mipLevel = 0, .baseArrayLayer = 0, .layerCount = layers},
        .srcOffsets = {{0, 0, 0}, {static_cast<std::int32_t>(src.width()), static_cast<std::int32_t>(src.height()), 1}},
        .dstSubresource = {.aspectMask = dst.aspect(), .mipLevel = 0, .baseArrayLayer = 0, .layerCount = layers},
        .dstOffsets = {
            {0, 0, 0}, {static_cast<std::int32_t>(dst.width()), static_cast<std::int32_t>(dst.height()), 1}}};
    vk.vkCmdBlitImage(
        cmd,
        src.vk_image(),
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        dst.vk_image(),
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region,
        filter);

    for (Image* image : {&src, &dst})
    {
        record_image_transition(
            vk,
            cmd,
            image->vk_image(),
            image == &src ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            image == &src ? VK_ACCESS_TRANSFER_READ_BIT : VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            kAllShaderStages,
            image->aspect(),
            0,
            1,
            layers);
    }
}

// Fill every layer of mip 0 with one colour and leave the image sampleable.
// The contents are discarded on entry for the same reason a copy's destination
// is: the clear covers all of them.
inline void record_image_clear(const VolkDeviceTable& vk, VkCommandBuffer cmd, Image& image, std::array<float, 4> color)
{
    const std::uint32_t layers = image.array_layers();
    record_image_transition(
        vk,
        cmd,
        image.vk_image(),
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        0,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        kAllShaderStages,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        image.aspect(),
        0,
        image.mip_levels(),
        layers);

    const VkClearColorValue value{{color[0], color[1], color[2], color[3]}};
    VkImageSubresourceRange range{
        .aspectMask = image.aspect(),
        .baseMipLevel = 0,
        .levelCount = image.mip_levels(),
        .baseArrayLayer = 0,
        .layerCount = layers};
    vk.vkCmdClearColorImage(cmd, image.vk_image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &value, 1, &range);

    record_image_transition(
        vk,
        cmd,
        image.vk_image(),
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        kAllShaderStages,
        image.aspect(),
        0,
        image.mip_levels(),
        layers);
}
