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
        std::optional<Format> depth)
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

        auto target = std::shared_ptr<OffscreenTarget>(new OffscreenTarget(context.shared_from_this()));
        target->extent_ = {width, height};

        for (Format f : colors)
        {
            auto image = Image::create_empty(context, width, height, f);
            if (!image)
            {
                return std::unexpected(image.error());
            }
            target->colors_.push_back(std::move(*image));
        }
        if (depth)
        {
            auto image = Image::create_empty(context, width, height, *depth);
            if (!image)
            {
                return std::unexpected(image.error());
            }
            target->depth_ = std::move(*image);
        }

        return target;
    }

    OffscreenTarget(const OffscreenTarget&) = delete;
    OffscreenTarget& operator=(const OffscreenTarget&) = delete;

    std::uint32_t color_count() const override
    {
        return static_cast<std::uint32_t>(colors_.size());
    }
    VkImage color_image(std::uint32_t i) const override
    {
        return colors_[i]->vk_image();
    }
    VkImageView color_view(std::uint32_t i) const override
    {
        return colors_[i]->view();
    }
    VkFormat color_format(std::uint32_t i) const override
    {
        return format_info(colors_[i]->format()).vk;
    }
    VkImage depth_image() const override
    {
        return depth_ ? depth_->vk_image() : VK_NULL_HANDLE;
    }
    VkImageView depth_view() const override
    {
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

    std::vector<std::shared_ptr<Image>> colors_;
    std::shared_ptr<Image> depth_;
};
