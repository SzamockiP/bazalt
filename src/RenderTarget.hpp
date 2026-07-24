#pragma once
#include <volk.h>
#include <vk_mem_alloc.h>

#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
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
        const std::string& name = "")
    {
        if (colors.empty() && !depth)
        {
            return std::unexpected(err_resource(
                "A RenderTarget needs at least one attachment: pass color=..., "
                "depth=..., or both"));
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

        // colors_/depth_ are always the single-sample, sampleable attachments —
        // what target.color/target.depth expose and what final_layout() applies to.
        // With MSAA they double as resolve targets and a parallel multisampled
        // image (msaa_colors_/msaa_depth_) is what actually gets rendered into.
        for (std::size_t i = 0; i < colors.size(); ++i)
        {
            auto resolve = Image::create_empty(context, width, height, colors[i]);
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
            auto resolve = Image::create_empty(context, width, height, *depth);
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

private:
    explicit OffscreenTarget(std::shared_ptr<Context> context)
        : context_(std::move(context))
    {
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
};
