#pragma once
#include <volk.h>
#include <vk_mem_alloc.h>

#include <cstdint>
#include <expected>
#include <map>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#include "Context.hpp"
#include "Error.hpp"
#include "Format.hpp"
#include "Image.hpp"
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

// Turn a user-facing sample count (1/2/4/8/…) into the Vulkan flag bit, rejecting
// anything this GPU can't back with both a colour and a depth attachment. One
// count serves every attachment in a pass, so validating against
// Context::max_samples() (the colour∩depth intersection) is the whole check.
// Same constructor-contract shape as the format guards below: a bad value fails
// loudly with a fix now, not a validation-layer crash at draw time.
inline std::expected<VkSampleCountFlagBits, Error> validate_sample_count(std::uint32_t samples, const Context& context)
{
    if (samples == 1)
    {
        return VK_SAMPLE_COUNT_1_BIT;
    }
    std::uint32_t max = context.max_samples();
    if (samples == 0 || (samples & (samples - 1)) != 0 || samples > max)
    {
        return std::unexpected(err_resource(std::format(
            "samples={} is not a valid MSAA count on this GPU; use a power of two "
            "in 1..{} (query it with bz.Context.max_samples())",
            samples,
            max)));
    }
    return static_cast<VkSampleCountFlagBits>(samples);
}

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

    // ── MSAA ──────────────────────────────────────────────────────────────────
    // A non-multisampled target answers 1-sample / VK_NULL_HANDLE to all of these,
    // so CommandBuffer's resolve wiring vanishes for it (resolveMode stays NONE).
    //
    // When samples() > 1, color_image/color_view return the *multisampled* images
    // that are rendered into, and color_resolve_* return the single-sample images
    // the pass resolves into — the ones that become sampleable/presentable and
    // that final_layout() applies to. Depth resolves the same way (SAMPLE_ZERO)
    // only when the target keeps its depth (offscreen); a swapchain's scratch
    // depth is multisampled but never resolved.
    virtual VkSampleCountFlagBits samples() const
    {
        return VK_SAMPLE_COUNT_1_BIT;
    }
    virtual VkImage color_resolve_image(std::uint32_t) const
    {
        return VK_NULL_HANDLE;
    }
    virtual VkImageView color_resolve_view(std::uint32_t) const
    {
        return VK_NULL_HANDLE;
    }
    virtual VkImage depth_resolve_image() const
    {
        return VK_NULL_HANDLE;
    }
    virtual VkImageView depth_resolve_view() const
    {
        return VK_NULL_HANDLE;
    }

    // ── Subresource selection (render-to-layer / render-to-mip) ────────────────
    // Which array layer(s) and mip of each attachment a pass actually writes.
    // The default is the whole-image, base-subresource case every target used
    // before 0.13: layer 0, mip 0, one of each. A SubresourceTarget overrides
    // these so CommandBuffer's attachment barriers hit exactly the layer/mip the
    // view renders into — the view and the barrier both read from here, so they
    // cannot drift. extent() (mip-scaled) covers renderArea/viewport/scissor.
    struct Subresource
    {
        std::uint32_t base_layer = 0;
        std::uint32_t layer_count = 1;
        std::uint32_t base_mip = 0;
        std::uint32_t mip_count = 1;
    };
    virtual Subresource color_subresource() const
    {
        return {};
    }
    virtual Subresource depth_subresource() const
    {
        return {};
    }

    // The layout the colour attachments must be left in when rendering ends.
    // A swapchain needs PRESENT_SRC_KHR; an offscreen target that will be sampled
    // needs SHADER_READ_ONLY_OPTIMAL. This being a virtual is what removes the
    // hardcoded present transition from CommandBuffer.
    virtual VkImageLayout final_layout() const = 0;

    // Same question for the depth attachment. The swapchain's depth buffer is
    // scratch (stays DEPTH_ATTACHMENT_OPTIMAL, store DONT_CARE); an offscreen
    // depth ends sampleable, which is the whole of what makes `shadow.depth` a
    // texture with zero extra API. end_rendering also derives its store-op from
    // this: a depth that will be consumed must be stored.
    virtual VkImageLayout depth_final_layout() const
    {
        return VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    }

    // Called by CommandBuffer when the end-of-rendering barrier is recorded into
    // a real submit. An OffscreenTarget uses this to learn that its image has
    // left UNDEFINED — the submit paths never see the target (it lives inside the
    // recorded lambdas), so the notification has to come from the recording
    // itself. No-op for targets that don't care.
    virtual void on_rendering_recorded()
    {
    }
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

// A render target backed by Images this object owns, with no swapchain and no
// window involved. This is what makes headless rendering — and therefore the
// test suite — possible. The attachments are ordinary bz.Image objects, which
// is the whole render-to-texture story: `target.color[0]` and `target.depth`
// go straight into set_image() with no extra API.
class OffscreenTarget : public RenderTarget, public std::enable_shared_from_this<OffscreenTarget>
{
public:
    static std::expected<std::shared_ptr<OffscreenTarget>, Error> create(
        Context& context,
        std::uint32_t width,
        std::uint32_t height,
        std::vector<Format> colors,
        std::optional<Format> depth,
        std::uint32_t samples = 1,
        std::uint32_t layers = 1,
        bool cube = false,
        std::uint32_t mip_levels = 1,
        const std::string& name = "")
    {
        if (colors.empty() && !depth)
        {
            return std::unexpected(err_resource(
                "A RenderTarget needs at least one attachment: pass color=..., "
                "depth=..., or both"));
        }
        // Render-to-layer / render-to-mip: the attachments become layered / cube /
        // mipped images and target.layer(i)/.mip(m) slice one subresource to render
        // into. cube fixes 6 square layers (Vulkan face order +X,-X,+Y,-Y,+Z,-Z);
        // the colour attachment gets a CUBE view so target.color[0] samples as a
        // cubemap, the depth attachment stays a plain 2D array (never a cube).
        if (cube)
        {
            if (layers != 1 && layers != 6)
            {
                return std::unexpected(err_resource(std::format(
                    "a cube RenderTarget implies 6 layers; drop layers= or pass layers=6, got {}", layers)));
            }
            if (width != height)
            {
                return std::unexpected(err_resource(std::format(
                    "a cube RenderTarget needs square faces, got {}x{}", width, height)));
            }
            layers = 6;
        }
        if (layers == 0 || mip_levels == 0)
        {
            return std::unexpected(err_resource("layers and mip_levels must be >= 1"));
        }
        // Cap the mip chain to the dimensions, like create_image: a level count past
        // the full chain fails at vkCreateImage (and trips the validation layer), so
        // reject it here with a message that says the ceiling.
        if (width > 0 && height > 0)
        {
            const std::uint32_t max_mips = Image::full_mip_count(width, height);
            if (mip_levels > max_mips)
            {
                return std::unexpected(err_resource(std::format(
                    "mip_levels must be 1..{} for a {}x{} target, got {}", max_mips, width, height, mip_levels)));
            }
        }
        // MSAA is a single-subresource attachment only (Image forbids samples>1 with
        // mips/layers/cube): a multisampled array + per-layer resolve is a separate
        // future feature, so reject the combination here with a clear message
        // instead of at vkCreateImage.
        if (samples > 1 && (layers > 1 || cube || mip_levels > 1))
        {
            return std::unexpected(err_resource(
                "samples>1 cannot combine with layers/cube/mip_levels: a render-to-layer "
                "target is single-sample in this release"));
        }
        for (Format f : colors)
        {
            if (format_info(f).depth)
            {
                return std::unexpected(err_resource(
                    std::format(
                        "{} is a depth format and cannot be a colour attachment; "
                        "pass it as depth= instead",
                        format_name(f))));
            }
        }
        if (depth && !format_info(*depth).depth)
        {
            return std::unexpected(
                err_resource(std::format("{} is not a depth format; use bz.Format.D32F", format_name(*depth))));
        }
        auto vk_samples = validate_sample_count(samples, context);
        if (!vk_samples)
        {
            return std::unexpected(vk_samples.error());
        }
        const bool msaa = *vk_samples != VK_SAMPLE_COUNT_1_BIT;

        auto target = std::shared_ptr<OffscreenTarget>(new OffscreenTarget(context.shared_from_this()));
        target->extent_ = {width, height};
        target->samples_ = *vk_samples;
        target->layers_ = layers;
        target->mip_levels_ = mip_levels;

        // colors_/depth_ are always the single-sample, sampleable attachments —
        // what target.color/target.depth expose and what final_layout() applies to.
        // With MSAA they double as resolve targets and a parallel multisampled
        // image (msaa_colors_/msaa_depth_) is what actually gets rendered into.
        for (std::size_t i = 0; i < colors.size(); ++i)
        {
            auto resolve = Image::create_empty(context, width, height, colors[i], mip_levels, layers, cube);
            if (!resolve)
            {
                return std::unexpected(resolve.error());
            }
            context.set_debug_name(
                VK_OBJECT_TYPE_IMAGE,
                reinterpret_cast<std::uint64_t>((*resolve)->vk_image()),
                name.empty() ? "" : std::format("{} color[{}]", name, i));
            target->colors_.push_back(std::move(*resolve));
            if (msaa)
            {
                auto ms = Image::create_empty(context, width, height, colors[i], 1, 1, false, *vk_samples);
                if (!ms)
                {
                    return std::unexpected(ms.error());
                }
                context.set_debug_name(
                    VK_OBJECT_TYPE_IMAGE,
                    reinterpret_cast<std::uint64_t>((*ms)->vk_image()),
                    name.empty() ? "" : std::format("{} msaa color[{}]", name, i));
                target->msaa_colors_.push_back(std::move(*ms));
            }
        }
        if (depth)
        {
            // Depth is a plain 2D array even for a cube target: it is scratch, never
            // sampled as a cubemap, so it needs no CUBE view (cube=false).
            auto resolve = Image::create_empty(context, width, height, *depth, mip_levels, layers, false);
            if (!resolve)
            {
                return std::unexpected(resolve.error());
            }
            context.set_debug_name(
                VK_OBJECT_TYPE_IMAGE,
                reinterpret_cast<std::uint64_t>((*resolve)->vk_image()),
                name.empty() ? "" : std::format("{} depth", name));
            target->depth_ = std::move(*resolve);
            if (msaa)
            {
                auto ms = Image::create_empty(context, width, height, *depth, 1, 1, false, *vk_samples);
                if (!ms)
                {
                    return std::unexpected(ms.error());
                }
                context.set_debug_name(
                    VK_OBJECT_TYPE_IMAGE,
                    reinterpret_cast<std::uint64_t>((*ms)->vk_image()),
                    name.empty() ? "" : std::format("{} msaa depth", name));
                target->msaa_depth_ = std::move(*ms);
            }
        }

        return target;
    }

    OffscreenTarget(const OffscreenTarget&) = delete;
    OffscreenTarget& operator=(const OffscreenTarget&) = delete;

    std::uint32_t color_count() const override
    {
        return static_cast<std::uint32_t>(colors_.size());
    }
    // With MSAA the multisampled image is the one rendered into; colors_ is its
    // resolve target (returned by color_resolve_* below).
    VkImage color_image(std::uint32_t i) const override
    {
        return msaa_colors_.empty() ? colors_[i]->vk_image() : msaa_colors_[i]->vk_image();
    }
    VkImageView color_view(std::uint32_t i) const override
    {
        return msaa_colors_.empty() ? colors_[i]->view() : msaa_colors_[i]->view();
    }
    VkFormat color_format(std::uint32_t i) const override
    {
        return format_info(colors_[i]->format()).vk;
    }
    VkImage depth_image() const override
    {
        if (msaa_depth_)
        {
            return msaa_depth_->vk_image();
        }
        return depth_ ? depth_->vk_image() : VK_NULL_HANDLE;
    }
    VkImageView depth_view() const override
    {
        if (msaa_depth_)
        {
            return msaa_depth_->view();
        }
        return depth_ ? depth_->view() : VK_NULL_HANDLE;
    }
    VkFormat depth_format() const override
    {
        return depth_ ? format_info(depth_->format()).vk : VK_FORMAT_UNDEFINED;
    }
    VkExtent2D extent() const override
    {
        return extent_;
    }

    VkSampleCountFlagBits samples() const override
    {
        return samples_;
    }
    VkImage color_resolve_image(std::uint32_t i) const override
    {
        return msaa_colors_.empty() ? VK_NULL_HANDLE : colors_[i]->vk_image();
    }
    VkImageView color_resolve_view(std::uint32_t i) const override
    {
        return msaa_colors_.empty() ? VK_NULL_HANDLE : colors_[i]->view();
    }
    VkImage depth_resolve_image() const override
    {
        return msaa_depth_ ? depth_->vk_image() : VK_NULL_HANDLE;
    }
    VkImageView depth_resolve_view() const override
    {
        return msaa_depth_ ? depth_->view() : VK_NULL_HANDLE;
    }

    // Left ready to be sampled, so using the result as a texture needs no extra
    // step — colour and depth both.
    VkImageLayout final_layout() const override
    {
        return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    VkImageLayout depth_final_layout() const override
    {
        return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    // The attachments as Images, for Python and for readback.
    const std::vector<std::shared_ptr<Image>>& colors() const
    {
        return colors_;
    }
    const std::shared_ptr<Image>& depth() const
    {
        return depth_;
    }

    // Copies colour attachment 0 back to host memory; kept as the ergonomic
    // spelling for tests (target.color[0].read() is the general form).
    std::expected<std::vector<std::byte>, Error> read_pixels()
    {
        if (colors_.empty())
        {
            return std::unexpected(
                err_resource("read_pixels() on a depth-only RenderTarget; read target.depth instead"));
        }
        return colors_[0]->read();
    }

    // Runs at execute() time, inside a real submit — the attachments learn they
    // have contents exactly when that becomes true (the 0.4.1 read_pixels fix,
    // now spelled per-Image). Depth included: that is what makes shadow maps
    // readable and sampleable.
    void on_rendering_recorded() override
    {
        for (auto& image : colors_)
        {
            image->mark_has_contents(final_layout());
        }
        if (depth_)
        {
            depth_->mark_has_contents(depth_final_layout());
        }
    }

    ~OffscreenTarget()
    {
        // The per-subresource views are the only Vulkan objects OffscreenTarget owns
        // beyond its Images (which self-destruct, deferred). A SubresourceTarget
        // borrows these — they live exactly as long as the parent, so it never owns
        // Vulkan objects itself and can be created/discarded freely.
        if (!subresource_views_.empty() && context_)
        {
            std::vector<VkImageView> views;
            views.reserve(subresource_views_.size());
            for (auto& [key, view] : subresource_views_)
            {
                views.push_back(view);
            }
            context_->defer_destroy(
                [device = context_->device(), views = std::move(views)]
                {
                    for (VkImageView v : views)
                    {
                        vkDestroyImageView(device, v, nullptr);
                    }
                });
        }
    }

    // ── render-to-layer / render-to-mip ────────────────────────────────────────
    // A single (layer, mip) subresource of a colour/depth attachment, as a plain
    // 2D view — what a SubresourceTarget renders into. Cached and owned here;
    // slicing the same subresource twice returns the same view.
    VkImageView color_subresource_view(std::uint32_t attachment, std::uint32_t layer, std::uint32_t mip)
    {
        return subresource_view_(
            static_cast<int>(attachment), colors_[attachment], VK_IMAGE_ASPECT_COLOR_BIT, layer, mip);
    }
    VkImageView depth_subresource_view(std::uint32_t layer, std::uint32_t mip)
    {
        if (!depth_)
        {
            return VK_NULL_HANDLE;
        }
        return subresource_view_(-1, depth_, VK_IMAGE_ASPECT_DEPTH_BIT, layer, mip);
    }

    // Bounds-checked slices. Returned as a RenderTargetBase (the view is a
    // RenderTarget), so it passes straight into cmd.rendering(...). The other axis
    // is pinned to 0: one axis per slice this release (.layer(i).mip(m) deferred).
    std::expected<std::shared_ptr<RenderTarget>, Error> layer(std::uint32_t i);
    std::expected<std::shared_ptr<RenderTarget>, Error> mip(std::uint32_t m);

private:
    explicit OffscreenTarget(std::shared_ptr<Context> context)
        : context_(std::move(context))
    {
    }

    // Shared body of the two subresource-view accessors: cache lookup keyed by
    // (attachment; -1 == depth, layer, mip), on miss create a 2D single-mip
    // single-layer view and store it.
    VkImageView subresource_view_(
        int attachment, const std::shared_ptr<Image>& image, VkImageAspectFlags aspect, std::uint32_t layer, std::uint32_t mip)
    {
        auto key = std::tuple{attachment, layer, mip};
        if (auto it = subresource_views_.find(key); it != subresource_views_.end())
        {
            return it->second;
        }
        VkImageViewCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = image->vk_image(),
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = format_info(image->format()).vk,
            .components = {},
            .subresourceRange = {aspect, mip, 1, layer, 1}};
        VkImageView view = VK_NULL_HANDLE;
        // Bounds are checked in layer()/mip() before we get here; a create failure
        // is a genuine driver error, so surface a null and let the caller's
        // validation-as-assert catch the bad attachment rather than crashing.
        if (vkCreateImageView(context_->device(), &info, nullptr, &view) != VK_SUCCESS)
        {
            return VK_NULL_HANDLE;
        }
        subresource_views_.emplace(key, view);
        return view;
    }

    std::shared_ptr<Context> context_;
    VkExtent2D extent_{};

    // Sampleable single-sample attachments (Python's target.color/target.depth).
    // With MSAA these are the resolve targets; without it they're rendered into
    // directly.
    std::vector<std::shared_ptr<Image>> colors_;
    std::shared_ptr<Image> depth_;

    // The multisampled images actually rendered into. Empty / null unless
    // samples_ > 1; colors_/depth_ then serve as their resolve targets.
    std::vector<std::shared_ptr<Image>> msaa_colors_;
    std::shared_ptr<Image> msaa_depth_;
    VkSampleCountFlagBits samples_ = VK_SAMPLE_COUNT_1_BIT;

    // Layer / mip counts every attachment shares. layers_ == 6 for a cube.
    std::uint32_t layers_ = 1;
    std::uint32_t mip_levels_ = 1;

    // Lazily created per-subresource render views, keyed (attachment; -1 == depth,
    // layer, mip). Owned here, destroyed (deferred) in the destructor.
    std::map<std::tuple<int, std::uint32_t, std::uint32_t>, VkImageView> subresource_views_;
};

// A render target that is one (layer, mip) subresource of an OffscreenTarget —
// what target.layer(i) / target.mip(m) hand back. It owns no Vulkan objects: the
// per-subresource views live in the parent's cache, the attachment Images are the
// parent's. begin_rendering needs no change to render into it — the attachment
// views come back single-subresource (color_view/depth_view), the barriers read
// the {layer,1,mip,1} range (color_subresource/depth_subresource), and extent()
// is mip-scaled so renderArea/viewport/scissor shrink to the mip automatically.
class SubresourceTarget : public RenderTarget
{
public:
    SubresourceTarget(std::shared_ptr<OffscreenTarget> parent, std::uint32_t layer, std::uint32_t mip)
        : parent_(std::move(parent)), layer_(layer), mip_(mip)
    {
    }

    std::uint32_t color_count() const override
    {
        return parent_->color_count();
    }
    VkImage color_image(std::uint32_t i) const override
    {
        return parent_->color_image(i);
    }
    VkImageView color_view(std::uint32_t i) const override
    {
        return parent_->color_subresource_view(i, layer_, mip_);
    }
    VkFormat color_format(std::uint32_t i) const override
    {
        return parent_->color_format(i);
    }
    VkImage depth_image() const override
    {
        return parent_->depth_image();
    }
    VkImageView depth_view() const override
    {
        return parent_->depth_subresource_view(layer_, mip_);
    }
    VkFormat depth_format() const override
    {
        return parent_->depth_format();
    }

    // The whole point of "no CommandBuffer edits for renderArea": a .mip(m) target
    // reports the mip's dimensions, so the pass covers exactly that mip.
    VkExtent2D extent() const override
    {
        VkExtent2D e = parent_->extent();
        std::uint32_t w = e.width >> mip_;
        std::uint32_t h = e.height >> mip_;
        return {w ? w : 1u, h ? h : 1u};
    }

    // samples()/color_resolve_*/depth_resolve_* keep the base defaults (1 /
    // VK_NULL_HANDLE): layered/mipped targets are single-sample, so CommandBuffer's
    // resolve wiring stays off.

    VkImageLayout final_layout() const override
    {
        return parent_->final_layout();
    }
    VkImageLayout depth_final_layout() const override
    {
        return parent_->depth_final_layout();
    }

    Subresource color_subresource() const override
    {
        return {layer_, 1, mip_, 1};
    }
    Subresource depth_subresource() const override
    {
        return {layer_, 1, mip_, 1};
    }

    // Forwards to the parent, which marks the whole attachment Image sampleable.
    // Correct once every layer/mip a caller intends to sample has been rendered
    // (Image holds one layout for the whole image — sampling a partially rendered
    // layered target is undefined and validation will flag it).
    void on_rendering_recorded() override
    {
        parent_->on_rendering_recorded();
    }

private:
    std::shared_ptr<OffscreenTarget> parent_;
    std::uint32_t layer_;
    std::uint32_t mip_;
};

inline std::expected<std::shared_ptr<RenderTarget>, Error> OffscreenTarget::layer(std::uint32_t i)
{
    if (i >= layers_)
    {
        return std::unexpected(err_resource(
            std::format("layer {} is out of range; this target has {} layer(s)", i, layers_)));
    }
    return std::make_shared<SubresourceTarget>(shared_from_this(), i, 0);
}

inline std::expected<std::shared_ptr<RenderTarget>, Error> OffscreenTarget::mip(std::uint32_t m)
{
    if (m >= mip_levels_)
    {
        return std::unexpected(err_resource(
            std::format("mip {} is out of range; this target has {} mip level(s)", m, mip_levels_)));
    }
    return std::make_shared<SubresourceTarget>(shared_from_this(), 0, m);
}
