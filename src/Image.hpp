#pragma once
#include <volk.h>
#include <vk_mem_alloc.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <expected>
#include <format>
#include <memory>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "Context.hpp"
#include "Error.hpp"
#include "Format.hpp"
#include "ImmediateSubmit.hpp"
#include "ResourceTracker.hpp"

// Layout transition helper shared by uploads, mip generation and readback.
// Formerly a private of Texture; RenderTarget grew its own copy, which is
// exactly the drift this ends.
void record_image_transition(
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
    std::uint32_t baseLayer = 0);

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

// Copy the whole of `src` into `dst` — same size, same format, every layer of
// mip 0. Both ends are transitioned around the transfer and left in
// SHADER_READ_ONLY, which is the layout a copy exists to produce. `src_layout`
// is where the source currently is (see cmd.copy_image).
//
// Only mip 0 is copied: the destination's other levels, if it has any, are
// regenerated with cmd.generate_mipmaps. Copying a chain would be N regions for
// a case that has not come up.
// legal_stages: the stage bits the replaying queue family supports (0.29).
// Both helpers name all_shader_stages() as one side of their transitions, and
// on a compute-only family most of those bits are illegal in a barrier.
// One (layer, mip) between a tightly packed buffer and an image, with the
// transitions on both sides — the engine behind cmd.copy_buffer_to_image and
// its mirror, shaped like record_image_copy below: self-contained, narrowed to
// the replaying family, both directions leave the subresource in
// SHADER_READ_ONLY. to_image discards the old level (UNDEFINED — the copy
// overwrites all of it); from_image starts from src_layout.
void record_buffer_image_copy(
    const VolkDeviceTable& vk,
    VkCommandBuffer cmd,
    Image& image,
    VkBuffer buffer,
    VkDeviceSize buffer_offset,
    std::uint32_t layer,
    std::uint32_t mip,
    bool to_image,
    VkImageLayout src_layout,
    VkPipelineStageFlags legal_stages);

void record_image_copy(
    const VolkDeviceTable& vk,
    VkCommandBuffer cmd,
    Image& src,
    Image& dst,
    VkImageLayout src_layout,
    VkPipelineStageFlags legal_stages = ~VkPipelineStageFlags{0});

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
void record_image_blit(
    const VolkDeviceTable& vk,
    VkCommandBuffer cmd,
    Image& src,
    Image& dst,
    VkImageLayout src_layout,
    VkFilter filter);

// Fill every layer of mip 0 with one colour and leave the image sampleable.
// The contents are discarded on entry for the same reason a copy's destination
// is: the clear covers all of them.
void record_image_clear(
    const VolkDeviceTable& vk,
    VkCommandBuffer cmd,
    Image& image,
    std::array<float, 4> color,
    VkPipelineStageFlags legal_stages = ~VkPipelineStageFlags{0});

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
        std::uint32_t depth = 1);

    // Deferred: an in-flight frame may still sample this image.
    ~Image();

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

    // The layout the GPU really leaves a subresource in, with no claim about
    // contents. The graph writes these back after a replay, and it must not
    // answer the contents question too: a pass that only SAMPLES an image
    // moves nothing into it, so marking it as filled would let read() hand
    // back a virgin image's garbage. Defaults cover the whole image.
    void set_layout(
        VkImageLayout layout,
        std::uint32_t base_layer = 0,
        std::uint32_t layer_count = VK_REMAINING_ARRAY_LAYERS,
        std::uint32_t base_mip = 0,
        std::uint32_t mip_count = VK_REMAINING_MIP_LEVELS)
    {
        layouts_.set_range(layout, base_layer, layer_count, base_mip, mip_count);
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
        std::uint32_t mip_count);

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
    bool ready() const;

    // Block until this one image's upload has finished on the GPU.
    // A failed decode surfaces here as ResourceError.
    std::expected<void, Error> wait();

    // Called at submit time for every image a command buffer references.
    // Returns the timeline serials the frame's GPU work must wait for, one per
    // queue (all zero when nothing is pending — RTT attachments and
    // synchronously uploaded images short-circuit here). CPU-blocks only while
    // the worker is still decoding.
    std::expected<QueueSerials, Error> require_resident();

    // The last upload submitted for this image, per queue, or zeros. Read by
    // the upload worker so a queued job can be ordered behind an upload some
    // OTHER thread made: create_image(array) submits inline on the calling
    // thread, so the worker's own chain does not know about it. Two values
    // can be set at once since 0.30: the copy signals the transfer timeline
    // and the mip cascade the graphics one.
    // One (layer, mip) of the image as a view, for a narrowed descriptor
    // (set_image(layer=, mip=), 0.30). Whole-range asks return view() /
    // storage_view() untouched, so the ordinary path allocates nothing and
    // the 99% case is what it always was.
    //
    // A narrowed LAYER is always VIEW_TYPE_2D: naming one face of a cubemap
    // asks for that face as a texture, not for a samplerCube of it. A
    // mip-only narrowing keeps the image's own view type, so mip 2 of a cube
    // is still a samplerCube.
    //
    // Cached and owned here, destroyed (deferred) with the image, exactly as
    // OffscreenTarget caches its attachment views.
    VkImageView subresource_view(std::optional<std::uint32_t> layer, std::optional<std::uint32_t> mip, bool storage);

    // The name= the image was created with, or empty. A debug label for
    // graph.explain() and the validation layer — never a key, the Pass::name_
    // contract.
    const std::string& name() const
    {
        return name_;
    }
    void set_name(std::string name)
    {
        name_ = std::move(name);
    }

    QueueSerials upload_serial() const
    {
        QueueSerials out{};
        for (std::size_t i = 0; i < kQueueCount; ++i)
        {
            out[i] = upload_serial_[i].load();
        }
        return out;
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
    void set_upload_pending();
    // Worker thread. It sets has_contents_ (an atomic) but NOT the layout state:
    // that is a vector the main thread reads without a lock, and since 0.18 it
    // is per-subresource, so a worker-side set_all would both race the reader
    // and be wrong for a partial update — claiming every layer sampleable when
    // one was written. Whoever queues the job records the layout instead, on the
    // main thread, where the rest of that state already lives.
    // Max-merged rather than assigned: the two halves of a mipped upload name
    // two timelines, and create_image(array) followed by update() may too.
    void set_upload_submitted(const QueueSerials& serials);
    void set_upload_failed(std::string message);
    // A queued job that produced no submit and changed nothing — a hot reload
    // the worker refused (wrong size, bad file). Balances set_upload_pending
    // and restores Submitted, because the previous contents still render.
    void abandon_upload();

    // ── Creation ──────────────────────────────────────────────────────────────

    // Every usage the format legally supports, filtered through the driver's
    // format properties. There is no `usage=` parameter on purpose: forgetting
    // STORAGE_BIT (or TRANSFER_SRC, or SAMPLED) is a classic Vulkan paper cut
    // with no upside — the driver knows what the format can do, so ask it.
    static VkImageUsageFlags usage_for(Context& context, Format format);

    // The usage an image of this format and sample count actually gets.
    //
    // Two narrowings on top of "everything the device supports", and the order
    // between them matters: the depth/stencil rule is a hard Vulkan limit, so it
    // runs first and the multisample rule cannot hand SAMPLED back.
    //
    // A combined depth/stencil image drops SAMPLED and STORAGE because its view
    // carries both aspects, and Vulkan forbids sampling through such a view.
    // Keeping the usage would make every DEPTH_STENCIL view illegal at creation,
    // which is a validation error at the target's constructor rather than at the
    // sample that was never going to work.
    //
    // A multisampled image keeps the attachment usage and SAMPLED. SAMPLED is
    // there so a shader can read the individual samples through sampler2DMS and
    // texelFetch — a custom resolve, which is what per-sample edge detection and
    // a TAA resolve are. Until 0.25 it was stripped, and the ceiling entry that
    // said so called the multisampled image transient: it is an ordinary
    // device-local image, so nothing but the usage bit stood in the way.
    //
    // STORAGE stays off: imageStore into a multisampled image needs
    // shaderStorageImageMultisample, which this library does not negotiate.
    // TRANSFER stays off because copying a multisampled image is illegal, which
    // is the same reason read() refuses one.
    static VkImageUsageFlags usage_for_image(Context& context, Format format, VkSampleCountFlagBits samples);

    // Mip generation needs to blit and to linearly filter the format.
    static bool can_generate_mips(Context& context, Format format);

    // Whether this GPU can blit `from` into `to`. A blit is a transfer command
    // and yet it is NOT universally available: BLIT_SRC and BLIT_DST are format
    // features, and a linear filter needs the source to be filterable on top.
    // Same shape as can_generate_mips, which asks the same question of one
    // format — and the same reason for asking it here: the alternative is a
    // validation message about VkFormatFeatureFlags instead of a sentence
    // naming the two formats.
    static bool can_blit(Context& context, Format from, Format to);

    // The chain length Vulkan derives from an extent: floor(log2(max axis)) + 1.
    // Depth participates for a volume — a 1x1x64 image has 7 levels — and
    // defaults to 1 so every 2D caller reads as before.
    static std::uint32_t full_mip_count(std::uint32_t width, std::uint32_t height, std::uint32_t depth = 1)
    {
        return static_cast<std::uint32_t>(std::bit_width((std::ranges::max)({width, height, depth})));
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
        std::uint32_t depth = 1);

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
        std::uint32_t depth = 1);

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
        bool mipmaps = false);

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
        std::uint32_t mip = 0);

    // Records the copy half of the first upload: transition all mips to
    // TRANSFER_DST from UNDEFINED (there are no contents to preserve), copy the
    // staging buffer into mip 0, then transition to SHADER_READ_ONLY — or,
    // when mips > 1, leave every level in TRANSFER_DST for record_upload_mips
    // on the graphics queue. The main thread replays this through
    // deferred_submit; the upload worker records it into a command buffer from
    // its own pool.
    //
    // `legal` is the replaying family's stage mask (0.30): the copy runs on
    // the transfer runtime, and on a transfer-only family a barrier may not
    // name a shader stage, so the retire narrows to BOTTOM_OF_PIPE and the
    // consumer's timeline wait is what makes the write visible.
    void record_upload_commands(VkCommandBuffer cmd, VkBuffer staging, std::uint32_t mips, VkPipelineStageFlags legal);

    // Hot reload: the image already holds contents that in-flight frames may
    // still be sampling. Transition from SHADER_READ_ONLY with a fragment-shader
    // source scope, so the copy waits for those reads — a WAR execution
    // dependency against every frame already submitted on this queue, no CPU
    // sync needed. UNDEFINED (as in the first upload) would instead let the
    // driver discard the live contents mid-frame, which sync validation flags.
    // On a transfer-only family that source scope narrows to nothing, and the
    // worker waits the graphics timeline for those frames instead (0.30).
    void record_reload_commands(VkCommandBuffer cmd, VkBuffer staging, std::uint32_t mips, VkPipelineStageFlags legal);

    // The blit half of a mipped upload (0.30): every level is in TRANSFER_DST
    // after record_upload_commands, and this generates levels 1..N from level
    // 0 and retires the chain to SHADER_READ_ONLY. Graphics only — a blit
    // needs a graphics family and always will — so the worker submits it on
    // the graphics runtime, waiting the copy's transfer serial.
    void record_upload_mips(VkCommandBuffer cmd, std::uint32_t mips);

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
        VkImageLayout from,
        VkPipelineStageFlags legal);

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
        VkAccessFlags src_access);

    // Creates and fills a staging buffer for this image's mip 0, all layers
    // (one image for a plain 2D texture; `array_layers_` images laid out back to
    // back for an array/cubemap).
    std::expected<std::pair<VkBuffer, VmaAllocation>, Error> create_filled_staging(
        Context& context,
        const void* pixels);

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

private:
    // CPU-side half of a wait: block while the worker is still decoding, then
    // surface a failed decode as the error it is.
    std::expected<void, Error> wait_submitted_();

    // Staging upload of mip 0, then either the blit chain filling the rest of
    // the levels or a single transition to SHADER_READ_ONLY.
    //
    // Asynchronous since 0.18.0, and without the upload worker: the caller
    // already handed over decoded bytes (a numpy array), so there is nothing to
    // move off this thread. The submit happens here — which is why every error
    // still surfaces at the create_image call — and only the wait is gone. The
    // image carries the serial, so it is its own future exactly like a
    // load_image one, with no second state machine and no second failure mode.
    std::expected<void, Error> upload_pixels(Context& context, const void* pixels, std::uint32_t mips);

    // Copy staging into mip 0, then transition mip 0 to SHADER_READ_ONLY when
    // there is no chain to generate. Shared by the first upload and hot reload
    // — the image must already be in TRANSFER_DST across all mips when this
    // runs, and it stays there when mips > 1.
    void record_copy_(VkCommandBuffer cmd, VkBuffer staging, std::uint32_t mips, VkPipelineStageFlags legal);

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
        std::uint32_t depth = 1);

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
    std::string name_;
    // Keyed (viewType, base_layer, layer_count, base_mip, mip_count).
    std::map<std::tuple<VkImageViewType, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>, VkImageView>
        subresource_views_;

    std::atomic<UploadState> upload_state_{UploadState::None};
    std::array<std::atomic<std::uint64_t>, kQueueCount> upload_serial_{};
    // How many queued uploads have not been submitted yet. Guarded by
    // upload_mutex_, which is also what the condition variable waits on.
    std::uint32_t pending_uploads_ = 0;
    std::mutex upload_mutex_;
    std::condition_variable upload_cv_;
    std::string upload_error_;
};
