#pragma once
#include <volk.h>
#include <vk_mem_alloc.h>

#include <algorithm>
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

// The size of one mip level along one axis. Floors at 1, which is what the
// chain of a non-square image does on its short axis before the long one.
inline constexpr std::uint32_t mip_extent(std::uint32_t base, std::uint32_t mip)
{
    const std::uint32_t v = base >> mip;
    return v > 0 ? v : 1;
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
        VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT,
        std::uint32_t depth = 1)
        : context_(std::move(context)),
          image_(image),
          allocation_(allocation),
          view_(view),
          storage_view_(storage_view),
          format_(format),
          width_(width),
          height_(height),
          depth_(depth),
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
    // The Z extent. 1 for every 2D image; >1 only for a volume created with
    // create_image(depth=). A volume always has exactly one array layer —
    // Vulkan's own rule — so depth and layers can never both be >1.
    std::uint32_t depth() const
    {
        return depth_;
    }
    bool is_3d() const
    {
        return depth_ > 1;
    }
    // The layer count a BARRIER on this image should name. A volume created
    // with 2D_ARRAY_COMPATIBLE must say VK_REMAINING_ARRAY_LAYERS: with
    // maintenance9 a count of 1 will mean one Z slice, and the layers already
    // warn about the narrower spelling. Copy and blit REGIONS keep the real
    // count — VkImageSubresourceLayers does not accept the sentinel.
    std::uint32_t barrier_layers(std::uint32_t narrow) const
    {
        return is_3d() ? VK_REMAINING_ARRAY_LAYERS : narrow;
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
        // A caller quoting a barrier range may say VK_REMAINING_ARRAY_LAYERS
        // (a 3D slice target does); the layout state wants the real count.
        layer_count = (std::ranges::min)(layer_count, array_layers_ - base_layer);
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
            return context_->wait_for_serial(upload_serial_.load());
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

    // The last upload submitted for this image, or 0. Read by the upload worker
    // so a queued job can be ordered behind an upload some OTHER thread made:
    // create_image(array) submits inline on the calling thread, so the worker's
    // own chain does not know about it.
    std::uint64_t upload_serial() const
    {
        return upload_serial_.load();
    }

    // Upload state transitions. Pending/Failed are worker-side; Submitted is
    // whoever made the submit — the worker for a decode, the calling thread for
    // create_image(array), which has nothing to decode (see upload_pixels).
    // Pending counts OUTSTANDING JOBS, it is not a flag (0.21). It was a flag,
    // and one image with several updates queued is exactly where that breaks: the
    // worker submitting the FIRST one flipped the state to Submitted, so wait()
    // stopped waiting while five jobs were still in the queue and read() returned
    // whichever one had landed. Six updates, the fifth one's pixels.
    //
    // Guarded by upload_mutex_ rather than atomic, because the condition variable
    // predicate has to read it under the same lock that the notify holds.
    void set_upload_pending()
    {
        {
            std::lock_guard lock(upload_mutex_);
            ++pending_uploads_;
            upload_state_.store(UploadState::Pending);
        }
        upload_cv_.notify_all();
    }
    // Worker thread. It sets has_contents_ (an atomic) but NOT the layout state:
    // that is a vector the main thread reads without a lock, and since 0.18 it
    // is per-subresource, so a worker-side set_all would both race the reader
    // and be wrong for a partial update — claiming every layer sampleable when
    // one was written. Whoever queues the job records the layout instead, on the
    // main thread, where the rest of that state already lives.
    void set_upload_submitted(std::uint64_t serial)
    {
        {
            std::lock_guard lock(upload_mutex_);
            upload_serial_.store(serial);
            has_contents_.store(true);
            // Zero already, and legitimately so, for create_image(array): that
            // path submits inline on the calling thread and never queues a job,
            // so there is nothing outstanding to retire.
            if (pending_uploads_ > 0)
            {
                --pending_uploads_;
            }
            // Still Pending while anything is queued behind this one. The serial
            // above is the newest submitted, so a waiter that gets through waits
            // for the last job rather than for whichever finished first.
            if (pending_uploads_ == 0 && upload_state_.load() != UploadState::Failed)
            {
                upload_state_.store(UploadState::Submitted);
            }
        }
        upload_cv_.notify_all();
    }
    void set_upload_failed(std::string message)
    {
        {
            std::lock_guard lock(upload_mutex_);
            upload_error_ = std::move(message);
            // Retires this job like a submit does, so a waiter is not left
            // counting one that will never arrive. Failed wins over Submitted
            // while anything is outstanding: a later job succeeding does not make
            // the earlier failure untrue, and the next set_upload_pending clears
            // it, which is what it did before it could be one job of several.
            if (pending_uploads_ > 0)
            {
                --pending_uploads_;
            }
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

    // The chain length Vulkan derives from an extent: floor(log2(max axis)) + 1.
    // Depth participates for a volume — a 1x1x64 image has 7 levels — and
    // defaults to 1 so every 2D caller reads as before.
    static std::uint32_t full_mip_count(std::uint32_t width, std::uint32_t height, std::uint32_t depth = 1)
    {
        std::uint32_t mips = 1;
        std::uint32_t size = (std::ranges::max)({width, height, depth});
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
    // layers, view CUBE + a second 2D_ARRAY view for storage-image writes);
    // `depth > 1` makes it a VK_IMAGE_TYPE_3D volume (view 3D, one layer).
    static std::expected<std::shared_ptr<Image>, Error> create_empty(
        Context& context,
        std::uint32_t width,
        std::uint32_t height,
        Format format,
        std::uint32_t mip_levels = 1,
        std::uint32_t array_layers = 1,
        bool cube = false,
        VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT,
        std::uint32_t depth = 1)
    {
        if (width == 0 || height == 0 || depth == 0)
        {
            return std::unexpected(
                err_resource(std::format("Image dimensions must be non-zero, got {}x{}x{}", width, height, depth)));
        }
        // Vulkan's own rule: a 3D image has exactly one array layer, so a
        // layered or cube volume is not a combination that exists. Multisampled
        // 3D images do not exist either (VUID-VkImageCreateInfo-samples-02257).
        if (depth > 1 && (array_layers > 1 || cube))
        {
            return std::unexpected(err_resource(
                "A 3D image (depth>1) cannot have layers or be a cubemap: Vulkan gives a "
                "volume exactly one array layer. Use depth= alone, or layers=/cube= on a "
                "2D image."));
        }
        if (depth > 1 && samples != VK_SAMPLE_COUNT_1_BIT)
        {
            return std::unexpected(err_resource("A 3D image (depth>1) cannot be multisampled"));
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
        // Layered MSAA is plain Vulkan and a portability driver may still refuse it
        // (Metal has no multisampled texture array). One feature to ask, rather than
        // a validation error at vkCreateImage (0.22).
        if (samples != VK_SAMPLE_COUNT_1_BIT && array_layers > 1 && !context.supports(Feature::MULTISAMPLE_ARRAYS))
        {
            return std::unexpected(err_unsupported(
                "A multisampled image with layers>1 needs the MULTISAMPLE_ARRAYS feature, which this driver "
                "does not offer. Ask ctx.supports(bz.Feature.MULTISAMPLE_ARRAYS), or render the layers one "
                "at a time into single-layer multisampled targets."));
        }
        const FormatInfo info = format_info(format);
        const VkFormat vk_fmt = context.vk_format(format);
        if (vk_fmt == VK_FORMAT_UNDEFINED)
        {
            return std::unexpected(
                err_unsupported(std::format("This device supports no {} format", format_name(format))));
        }
        const VkImageAspectFlags aspect = aspect_mask_for(vk_fmt);
        const VkImageViewType view_type = depth > 1 ? VK_IMAGE_VIEW_TYPE_3D
                                          : cube    ? VK_IMAGE_VIEW_TYPE_CUBE
                                                    : (array_layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                                                                        : VK_IMAGE_VIEW_TYPE_2D);

        // 2D_ARRAY_COMPATIBLE is what later lets a 2D view select one Z slice
        // of the volume as a render-target attachment. It costs nothing where
        // legal, and a portability driver that lacks imageView2DOn3DImage would
        // reject the flag itself — so it is set only where the feature answers.
        VkImageCreateFlags flags = cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
        if (depth > 1 && context.supports(Feature::IMAGE_VIEW_2D_ON_3D))
        {
            flags |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
        }

        VkImageCreateInfo imageInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = flags,
            .imageType = depth > 1 ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D,
            .format = vk_fmt,
            .extent = {width, height, depth},
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
            samples,
            depth);
    }

    // From caller-provided pixels (numpy arrays land here). UNORM by default at
    // the binding layer: arrays are data, files are pictures. One mip by default
    // — data images don't want surprise filtering — but `mipmaps` opts a numpy
    // texture into the full chain (falling back to one level when the format
    // can't be blitted, exactly like load_from_file). `depth > 1` makes it a
    // volume: the pixels are `depth` slices of width×height back to back, which
    // is exactly what a C-contiguous (d, h, w, c) numpy array is.
    static std::expected<std::shared_ptr<Image>, Error> create_from_pixels(
        Context& context,
        const void* pixels,
        std::uint32_t width,
        std::uint32_t height,
        Format format,
        bool mipmaps = false,
        std::uint32_t depth = 1)
    {
        const std::uint32_t mips = mipmaps && can_generate_mips(context, format) ? full_mip_count(width, height, depth)
                                                                                 : 1;
        auto image = create_empty(context, width, height, format, mips, 1, false, VK_SAMPLE_COUNT_1_BIT, depth);
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
    // side too. Python's image.read() keeps one layer: a numpy array has no
    // place to put the cube-ness, so returning six faces stacked would be a
    // shape the caller has to guess at.
    //
    // `layer` and `mip` pick WHICH subresource. Until 0.18 there was no choice:
    // read() meant mip 0 of layer 0, so a cube face could not be inspected and
    // "did generate_mipmaps actually compute anything" was a question with no
    // way to ask it. The transition covers exactly the subresource being read
    // and takes its oldLayout from the per-subresource state, so reading one
    // level of a partially written image is legal and leaves the rest alone.
    std::expected<std::vector<std::byte>, Error> read(
        bool all_layers = false,
        std::uint32_t layer = 0,
        std::uint32_t mip = 0)
    {
        const std::uint32_t layers = all_layers ? array_layers_ : 1;
        const std::uint32_t base_layer = all_layers ? 0 : layer;
        // A pending async upload is finished first — read() is blocking anyway.
        if (auto w = wait(); !w)
        {
            return std::unexpected(w.error());
        }
        if (!has_contents_.load())
        {
            return std::unexpected(err_resource(
                "read() called on an Image that has no contents yet. Upload to it or "
                "render into it first"));
        }
        if (samples_ != VK_SAMPLE_COUNT_1_BIT)
        {
            return std::unexpected(err_resource(
                "read() called on a multisampled image. Read the target's resolved "
                "single-sample attachment (target.color[i] / target.depth) instead"));
        }
        if (!fits_within(base_layer, layers, array_layers_))
        {
            return std::unexpected(
                err_resource(std::format("read(layer={}): this image has {} layer(s)", base_layer, array_layers_)));
        }
        if (mip >= mip_levels_)
        {
            return std::unexpected(
                err_resource(std::format("read(mip={}): this image has {} mip level(s)", mip, mip_levels_)));
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
        const std::uint32_t mip_width = mip_extent(width_, mip);
        const std::uint32_t mip_height = mip_extent(height_, mip);
        const std::uint32_t mip_depth = mip_extent(depth_, mip);
        const VkDeviceSize size = static_cast<VkDeviceSize>(mip_width) * mip_height * mip_depth * info.bytes_per_pixel *
                                  layers;
        const VkImageAspectFlags aspect = aspect_mask_for(vk_format());

        auto staging_pair = create_staging_buffer(*context_, size, Staging::Readback);
        if (!staging_pair)
        {
            return std::unexpected(staging_pair.error());
        }
        auto [staging, staging_alloc] = *staging_pair;

        auto submitted = immediate_submit(
            *context_,
            [&](VkCommandBuffer cmd)
            {
                // Exactly the subresources being read, each from the layout it
                // is actually in. A single barrier over the whole image can only
                // name one oldLayout, and naming the wrong one is a validation
                // error plus undefined contents — which is precisely what
                // reading a partially rendered image used to do.
                for_each_subresource_(
                    base_layer,
                    layers,
                    mip,
                    [&](std::uint32_t l, std::uint32_t m)
                    {
                        record_image_transition(
                            context_->vk(),
                            cmd,
                            image_,
                            layouts_.get(l, m),
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            VK_ACCESS_MEMORY_WRITE_BIT,
                            VK_ACCESS_TRANSFER_READ_BIT,
                            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            aspect,
                            m,
                            1,
                            barrier_layers(1),
                            l);
                    });

                VkBufferImageCopy region{
                    .bufferOffset = 0,
                    .bufferRowLength = 0,
                    .bufferImageHeight = 0,
                    .imageSubresource = {aspect, mip, base_layer, layers},
                    .imageOffset = {0, 0, 0},
                    .imageExtent = {mip_width, mip_height, mip_depth}};
                context_->vk().vkCmdCopyImageToBuffer(
                    cmd, image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging, 1, &region);

                // …and put each one back where it was, which is what makes
                // read() non-destructive on a partially written image.
                for_each_subresource_(
                    base_layer,
                    layers,
                    mip,
                    [&](std::uint32_t l, std::uint32_t m)
                    {
                        record_image_transition(
                            context_->vk(),
                            cmd,
                            image_,
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            layouts_.get(l, m),
                            VK_ACCESS_TRANSFER_READ_BIT,
                            VK_ACCESS_MEMORY_READ_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                            aspect,
                            m,
                            1,
                            barrier_layers(1),
                            l);
                    });
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
    // main thread replays this through deferred_submit; the upload worker records
    // it into a command buffer from its own pool.
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
            barrier_layers(array_layers_));

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
            barrier_layers(array_layers_));

        record_copy_and_finalize_(cmd, staging, mips);
    }

    // A partial upload into an existing image: `extent` pixels at `offset` of
    // one (layer, mip), from a staging buffer holding exactly that rectangle.
    //
    // The whole point of image.update(): changing the pixels of an image that
    // already exists had no spelling at all, so a video frame, a camera feed, a
    // matplotlib figure or a painted texture meant a new Image every frame.
    //
    // `from` is the subresource's layout, read on the MAIN thread when the job
    // was queued — the worker must not touch the layout state, and hard-coding
    // SHADER_READ_ONLY the way record_reload_commands does would be wrong for an
    // image that has never been written.
    //
    // Only the named subresource is transitioned, so an update to one cube face
    // leaves the other five alone.
    // Offsets and extents are 3D since 0.23: for a 2D image z is 0 and the depth
    // is 1, for a volume they select the Z range the update writes.
    void record_update_commands(
        VkCommandBuffer cmd,
        VkBuffer staging,
        std::uint32_t layer,
        std::uint32_t mip,
        VkOffset3D offset,
        VkExtent3D extent,
        VkImageLayout from)
    {
        const VkImageAspectFlags aspect = aspect_mask_for(vk_format());
        const VkPipelineStageFlags shader_stages = context_->all_shader_stages();
        record_image_transition(
            context_->vk(),
            cmd,
            image_,
            from,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_ACCESS_SHADER_READ_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            shader_stages,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            aspect,
            mip,
            1,
            barrier_layers(1),
            layer);

        VkBufferImageCopy region{
            .bufferOffset = 0,
            // Zero means "tightly packed to imageExtent", which is what the
            // staging buffer is: the binding copies the caller's rectangle into
            // it row by row, so there is no source padding to describe.
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {aspect, mip, layer, 1},
            .imageOffset = offset,
            .imageExtent = extent};
        context_->vk().vkCmdCopyBufferToImage(cmd, staging, image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        record_image_transition(
            context_->vk(),
            cmd,
            image_,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            shader_stages,
            aspect,
            mip,
            1,
            barrier_layers(1),
            layer);
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
            barrier_layers(array_layers_));

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
            barrier_layers(array_layers_));

        std::int32_t mip_width = static_cast<std::int32_t>(width_);
        std::int32_t mip_height = static_cast<std::int32_t>(height_);
        // Depth halves alongside width and height for a volume (it is 1 and
        // stays 1 for everything else), so the loop terminates exactly where
        // full_mip_count says the chain ends.
        std::int32_t mip_depth = static_cast<std::int32_t>(depth_);

        for (std::uint32_t i = 1; i < mip_levels_; ++i)
        {
            const std::int32_t next_width = mip_width > 1 ? mip_width / 2 : 1;
            const std::int32_t next_height = mip_height > 1 ? mip_height / 2 : 1;
            const std::int32_t next_depth = mip_depth > 1 ? mip_depth / 2 : 1;

            // One blit with layerCount = array_layers_ fills level i for every
            // face/layer at once (they share mip dimensions).
            VkImageBlit blit{
                .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, array_layers_},
                .srcOffsets = {{0, 0, 0}, {mip_width, mip_height, mip_depth}},
                .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, array_layers_},
                .dstOffsets = {{0, 0, 0}, {next_width, next_height, next_depth}}};
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
                barrier_layers(array_layers_));

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
                    barrier_layers(array_layers_));
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
                    barrier_layers(array_layers_));
            }

            mip_width = next_width;
            mip_height = next_height;
            mip_depth = next_depth;
        }
    }

    // Creates and fills a staging buffer for this image's mip 0, all layers
    // (one image for a plain 2D texture; `array_layers_` images laid out back to
    // back for an array/cubemap).
    std::expected<std::pair<VkBuffer, VmaAllocation>, Error> create_filled_staging(Context& context, const void* pixels)
    {
        const FormatInfo info = format_info(format_);
        // Layers and depth are never both >1 (create_empty refuses it), so one
        // multiply chain covers the array case and the volume case.
        const VkDeviceSize size = static_cast<VkDeviceSize>(width_) * height_ * info.bytes_per_pixel * array_layers_ *
                                  depth_;
        return create_staging_buffer(context, size, Staging::Upload, pixels);
    }

    // Calls fn(layer, mip) for every subresource of one mip across a layer
    // range. One barrier each rather than one coalesced barrier per run of equal
    // layouts: the callers are blocking setup and readback paths, and a handful
    // of extra barriers there is cheaper than the bookkeeping to merge them.
    template <typename Fn>
    void for_each_subresource_(std::uint32_t base_layer, std::uint32_t layer_count, std::uint32_t mip, Fn&& fn) const
    {
        for (std::uint32_t layer = base_layer; layer < base_layer + layer_count; ++layer)
        {
            fn(layer, mip);
        }
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
                barrier_layers(array_layers_));
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
                    barrier_layers(1),
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
                barrier_layers(array_layers_));
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
                    barrier_layers(1),
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
            // Every queued job, not just the first to be submitted. The predicate
            // used to be "the state is no longer Pending", which one worker
            // submitting the first of several satisfies immediately.
            upload_cv_.wait(lock, [&] { return pending_uploads_ == 0; });
            if (upload_state_.load() == UploadState::Failed)
            {
                return std::unexpected(err_resource(upload_error_));
            }
        }
        return {};
    }

    // Staging upload of mip 0, then either the blit chain filling the rest of
    // the levels or a single transition to SHADER_READ_ONLY.
    //
    // Asynchronous since 0.18.0, and without the upload worker: the caller
    // already handed over decoded bytes (a numpy array), so there is nothing to
    // move off this thread. The submit happens here — which is why every error
    // still surfaces at the create_image call — and only the wait is gone. The
    // image carries the serial, so it is its own future exactly like a
    // load_image one, with no second state machine and no second failure mode.
    std::expected<void, Error> upload_pixels(Context& context, const void* pixels, std::uint32_t mips)
    {
        auto staging = create_filled_staging(context, pixels);
        if (!staging)
        {
            return std::unexpected(staging.error());
        }
        auto [buffer, allocation] = *staging;

        auto serial = deferred_submit(context, [&](VkCommandBuffer cmd) { record_upload_commands(cmd, buffer, mips); });
        if (!serial)
        {
            vmaDestroyBuffer(context.allocator(), buffer, allocation);
            return std::unexpected(serial.error());
        }
        // The GPU reads the staging buffer after this returns, so it retires on
        // the serial rather than here.
        context.defer_destroy([allocator = context.allocator(), buffer, allocation]
                              { vmaDestroyBuffer(allocator, buffer, allocation); });

        mark_has_contents(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        set_upload_submitted(*serial);
        context.note_upload_serial(*serial);
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
            .imageExtent = {width_, height_, depth_}};
        context_->vk().vkCmdCopyBufferToImage(cmd, staging, image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        if (mips > 1)
        {
            record_mip_generation(context_->vk(), cmd, image_, width_, height_, mips, array_layers_, depth_);
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
                barrier_layers(array_layers_));
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
        std::uint32_t layers = 1,
        std::uint32_t depth = 1)
    {
        std::int32_t mip_width = static_cast<std::int32_t>(width);
        std::int32_t mip_height = static_cast<std::int32_t>(height);
        std::int32_t mip_depth = static_cast<std::int32_t>(depth);
        // Barriers on a volume name VK_REMAINING_ARRAY_LAYERS; see barrier_layers.
        const std::uint32_t barrier_span = depth > 1 ? VK_REMAINING_ARRAY_LAYERS : layers;

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
                barrier_span);

            const std::int32_t next_width = mip_width > 1 ? mip_width / 2 : 1;
            const std::int32_t next_height = mip_height > 1 ? mip_height / 2 : 1;
            const std::int32_t next_depth = mip_depth > 1 ? mip_depth / 2 : 1;

            VkImageBlit blit{
                .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, layers},
                .srcOffsets = {{0, 0, 0}, {mip_width, mip_height, mip_depth}},
                .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, layers},
                .dstOffsets = {{0, 0, 0}, {next_width, next_height, next_depth}}};
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
                barrier_span);

            mip_width = next_width;
            mip_height = next_height;
            mip_depth = next_depth;
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
            barrier_span);
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
    std::uint32_t depth_ = 1; // Z extent; >1 makes this a VK_IMAGE_TYPE_3D volume
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
    // How many queued uploads have not been submitted yet. Guarded by
    // upload_mutex_, which is also what the condition variable waits on.
    std::uint32_t pending_uploads_ = 0;
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
    // Read off the source rather than threaded in as a parameter: both images
    // already belong to this Context (the binding layer compares owners), so the
    // fact is here, and record_image_transition takes `vk` for the same reason.
    const VkPipelineStageFlags shader_stages = src.owner()->all_shader_stages();
    const std::uint32_t layers = src.array_layers();
    const std::uint32_t barrier_span = src.barrier_layers(layers);
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
        shader_stages,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        src.aspect(),
        0,
        mips,
        barrier_span);
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
        shader_stages,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        dst.aspect(),
        0,
        mips,
        barrier_span);

    std::vector<VkImageCopy> regions;
    regions.reserve(mips);
    for (std::uint32_t mip = 0; mip < mips; ++mip)
    {
        // Level dimensions floor at 1, which is what the mip chain of a
        // non-square image does on its short axis before the long one. Depth
        // participates for a volume and is 1 everywhere else.
        const std::uint32_t w = (std::ranges::max)(src.width() >> mip, 1u);
        const std::uint32_t h = (std::ranges::max)(src.height() >> mip, 1u);
        const std::uint32_t d = (std::ranges::max)(src.depth() >> mip, 1u);
        regions.push_back(
            VkImageCopy{
                .srcSubresource =
                    {.aspectMask = src.aspect(), .mipLevel = mip, .baseArrayLayer = 0, .layerCount = layers},
                .srcOffset = {0, 0, 0},
                .dstSubresource =
                    {.aspectMask = dst.aspect(), .mipLevel = mip, .baseArrayLayer = 0, .layerCount = layers},
                .dstOffset = {0, 0, 0},
                .extent = {w, h, d}});
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
            shader_stages,
            image->aspect(),
            0,
            mips,
            barrier_span);
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
    const VkPipelineStageFlags shader_stages = src.owner()->all_shader_stages();
    const std::uint32_t layers = (std::ranges::min)(src.array_layers(), dst.array_layers());
    const std::uint32_t barrier_span = src.barrier_layers(layers);

    record_image_transition(
        vk,
        cmd,
        src.vk_image(),
        src_layout,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_TRANSFER_READ_BIT,
        shader_stages,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        src.aspect(),
        0,
        1,
        barrier_span);
    record_image_transition(
        vk,
        cmd,
        dst.vk_image(),
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        0,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        shader_stages,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        dst.aspect(),
        0,
        1,
        barrier_span);

    VkImageBlit region{
        .srcSubresource = {.aspectMask = src.aspect(), .mipLevel = 0, .baseArrayLayer = 0, .layerCount = layers},
        .srcOffsets =
            {{0, 0, 0},
             {static_cast<std::int32_t>(src.width()),
              static_cast<std::int32_t>(src.height()),
              static_cast<std::int32_t>(src.depth())}},
        .dstSubresource = {.aspectMask = dst.aspect(), .mipLevel = 0, .baseArrayLayer = 0, .layerCount = layers},
        .dstOffsets = {
            {0, 0, 0},
            {static_cast<std::int32_t>(dst.width()),
             static_cast<std::int32_t>(dst.height()),
             static_cast<std::int32_t>(dst.depth())}}};
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
            shader_stages,
            image->aspect(),
            0,
            1,
            barrier_span);
    }
}

// Fill every layer of mip 0 with one colour and leave the image sampleable.
// The contents are discarded on entry for the same reason a copy's destination
// is: the clear covers all of them.
inline void record_image_clear(const VolkDeviceTable& vk, VkCommandBuffer cmd, Image& image, std::array<float, 4> color)
{
    const VkPipelineStageFlags shader_stages = image.owner()->all_shader_stages();
    const std::uint32_t barrier_span = image.barrier_layers(image.array_layers());
    record_image_transition(
        vk,
        cmd,
        image.vk_image(),
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        0,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        shader_stages,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        image.aspect(),
        0,
        image.mip_levels(),
        barrier_span);

    const VkClearColorValue value{{color[0], color[1], color[2], color[3]}};
    VkImageSubresourceRange range{
        .aspectMask = image.aspect(),
        .baseMipLevel = 0,
        .levelCount = image.mip_levels(),
        .baseArrayLayer = 0,
        .layerCount = barrier_span};
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
        shader_stages,
        image.aspect(),
        0,
        image.mip_levels(),
        barrier_span);
}
