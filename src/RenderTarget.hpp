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
        return std::unexpected(err_unsupported(
            std::format(
                "samples={} is not a valid MSAA count on this GPU. Use a power of two "
                "in 1..{} (query it with bz.Context.max_samples())",
                samples,
                max)));
    }
    return static_cast<VkSampleCountFlagBits>(samples);
}

// Transition every subresource of `image` that is not already in `layout` up to
// it, then collapse the layout state. A no-op — no barrier recorded at all —
// when the image is already uniform, which is every image that was written
// whole. See RenderTarget::record_even_out for why this exists.
inline void even_out_image(
    const VolkDeviceTable& vk,
    VkCommandBuffer cmd,
    Image& image,
    VkImageLayout layout,
    VkImageAspectFlags aspect)
{
    if (image.uniform_layout().has_value())
    {
        return;
    }
    for (std::uint32_t layer = 0; layer < image.array_layers(); ++layer)
    {
        for (std::uint32_t mip = 0; mip < image.mip_levels(); ++mip)
        {
            const VkImageLayout from = image.layout_of(layer, mip);
            if (from == layout)
            {
                continue;
            }
            record_image_transition(
                vk,
                cmd,
                image.vk_image(),
                from,
                layout,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                aspect,
                mip,
                1,
                image.barrier_layers(1),
                layer);
        }
    }
    image.mark_has_contents(layout);
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

    // Which Context this object belongs to. Multi-context (0.15) made "a
    // resource from the other Context" a reachable mistake, and its symptom
    // without a check is a driver crash or a validation message from Vulkan
    // rather than from bazalt; the binding layer compares owners at record time.
    virtual const Context* owner() const = 0;

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

    // Multiview: a non-zero mask renders every set bit's layer in ONE pass, the
    // shader keying per-view work off gl_ViewIndex. 0 (the default) is the ordinary
    // single-layer path. A MultiviewTarget (RenderTarget.all_layers()) returns
    // (1<<N)-1; its color/depth views span all N layers and its subresource covers
    // them, so the attachment barriers transition the whole array.
    virtual std::uint32_t view_mask() const
    {
        return 0;
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
    //
    // DEPTH_ATTACHMENT_OPTIMAL covers the depth aspect alone, and Vulkan
    // forbids it outright for an image that also carries stencil — so the
    // default reads the format rather than naming one layout. A window with
    // stencil=True lands here, and it is the reason this is not a constant.
    virtual VkImageLayout depth_final_layout() const
    {
        return has_stencil(depth_format()) ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                                           : VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    }

    // Called by CommandBuffer when the end-of-rendering barrier is recorded into
    // a real submit. An OffscreenTarget uses this to learn that its image has
    // left UNDEFINED — the submit paths never see the target (it lives inside the
    // recorded lambdas), so the notification has to come from the recording
    // itself. No-op for targets that don't care.
    virtual void on_rendering_recorded()
    {
    }

    // Bring the subresources this pass did NOT write up to the final layout too,
    // so the whole image ends where the RenderTarget contract promises.
    //
    // "The layout the result must end in" has always been a promise about the
    // IMAGE, and until 0.18 a subresource pass honoured it for the part it drew
    // and left the rest wherever it was. `target.mip(1)` therefore produced an
    // image that was SHADER_READ_ONLY at level 1 and UNDEFINED at level 0, and
    // the sampler saw one view over both — a validation error at the sample,
    // a long way from the pass that caused it. The old note called this "render
    // every layer and every mip before you sample", which reads as advice and
    // was really a missing barrier.
    //
    // Costs nothing on a whole-image pass: the layout state is uniform there, so
    // there is nothing to even out and this records no barrier at all. Runs
    // AFTER on_rendering_recorded, which is what makes the state true.
    //
    // Transitioning an untouched subresource out of UNDEFINED discards contents
    // that were undefined anyway, so it loses nothing that existed.
    virtual void record_even_out(const VolkDeviceTable& /*vk*/, VkCommandBuffer /*cmd*/)
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

    // The recording Context's device dispatch table (Context::vk()). Every
    // vkCmd* inside a recorded lambda goes through this — one field instead of
    // capturing a table pointer in each of the ~35 lambdas, and it keeps the
    // rule that a deferred lambda holds nothing that (indirectly) holds the
    // Context. A raw pointer into a Context that outlives its own recordings.
    const VolkDeviceTable* vk = nullptr;
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
                return std::unexpected(err_resource(
                    std::format(
                        "a cube RenderTarget implies 6 layers. Drop layers= or pass layers=6. Got layers={}", layers)));
            }
            if (width != height)
            {
                return std::unexpected(
                    err_resource(std::format("a cube RenderTarget needs square faces, got {}x{}", width, height)));
            }
            layers = 6;
        }
        if (layers == 0 || mip_levels == 0)
        {
            return std::unexpected(err_resource(
                std::format(
                    "layers and mip_levels must be at least 1. Got layers={}, mip_levels={}", layers, mip_levels)));
        }
        // Cap the mip chain to the dimensions, like create_image: a level count past
        // the full chain fails at vkCreateImage (and trips the validation layer), so
        // reject it here with a message that says the ceiling.
        if (width > 0 && height > 0)
        {
            const std::uint32_t max_mips = Image::full_mip_count(width, height);
            if (mip_levels > max_mips)
            {
                return std::unexpected(err_resource(
                    std::format(
                        "mip_levels must be 1..{} for a {}x{} target, got {}", max_mips, width, height, mip_levels)));
            }
        }
        // MSAA composes with layers/cube (the multisampled attachment is layered and
        // resolves per layer), but not with mips: a multisampled image has no mip
        // chain. Reject that combination here with a clear message instead of at
        // vkCreateImage.
        if (samples > 1 && mip_levels > 1)
        {
            return std::unexpected(err_resource(
                "samples>1 cannot combine with mip_levels: a multisampled image has no "
                "mip chain (MSAA + layers/cube is fine)"));
        }
        for (Format f : colors)
        {
            if (format_info(f).depth)
            {
                return std::unexpected(err_resource(
                    std::format(
                        "{} is a depth format and cannot be a colour attachment. "
                        "Pass it as depth= instead",
                        format_name(f))));
            }
        }
        if (depth && !format_info(*depth).depth)
        {
            return std::unexpected(err_resource(
                std::format(
                    "{} is not a depth format. Use bz.Format.D32F, or bz.Format.DEPTH_STENCIL "
                    "when the pass needs a stencil buffer",
                    format_name(*depth))));
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
                // The multisampled image matches the resolve target's layer count
                // (never cube: it's a plain layered attachment, resolved per layer).
                auto ms = Image::create_empty(context, width, height, colors[i], 1, layers, false, *vk_samples);
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
                auto ms = Image::create_empty(context, width, height, *depth, 1, layers, false, *vk_samples);
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

    // A target on images the caller already owns, instead of attachments this
    // class allocates. Everything create() takes as a knob — extent, layers, cube,
    // mip_levels — is read off the images instead, because the images already
    // answer those questions and a second answer could disagree with them.
    //
    // It earns being a separate entry point rather than optional width/height on
    // create() for the same reason blit_image is not copy_image(scale=True): the
    // two do different jobs. "Allocate attachments for me" and "render into these"
    // share only what happens afterwards.
    //
    // What this makes reachable: a graphics ping-pong between two textures, drawing
    // over a texture a compute pass baked, and drawing into an image brought from
    // another Context. All three were impossible while a target insisted on owning
    // its attachments.
    //
    // Ownership needs no work: the images are held by shared_ptr exactly as the
    // allocated ones are, and the destructor already destroys only its own views.
    // Note that the target does WRITE to a borrowed image's layout tracking
    // (mark_rendered / record_even_out) — that is the point, since final_layout()
    // leaves the result sampleable.
    //
    // samples>1 works here exactly as it does on create(): bazalt allocates the
    // multisampled attachments and the images handed in become their resolve
    // targets, which is what they already are on the allocating path. The
    // alternative — the caller creating a multisampled image and handing THAT in —
    // would need matching resolve images passed alongside it and would put a second
    // MSAA idiom in the API, one where bazalt owns the multisampled image and one
    // where the caller does.
    static std::expected<std::shared_ptr<OffscreenTarget>, Error> create_from_images(
        Context& context,
        std::vector<std::shared_ptr<Image>> colors,
        std::shared_ptr<Image> depth,
        std::uint32_t samples = 1,
        const std::string& name = "")
    {
        if (colors.empty() && !depth)
        {
            return std::unexpected(err_resource(
                "A RenderTarget needs at least one attachment: pass color=[...], "
                "depth=..., or both"));
        }
        for (const auto& image : colors)
        {
            if (!image)
            {
                return std::unexpected(err_resource("color contains a null image"));
            }
            if (format_info(image->format()).depth)
            {
                return std::unexpected(err_resource(
                    std::format(
                        "a {} image is a depth attachment and cannot go in color=. Pass it as depth=",
                        format_name(image->format()))));
            }
        }
        if (depth && !format_info(depth->format()).depth)
        {
            return std::unexpected(err_resource(
                std::format(
                    "depth= needs a depth format, got {}. Create the image with bz.Format.D32F, "
                    "or bz.Format.DEPTH_STENCIL when the pass needs a stencil buffer",
                    format_name(depth->format()))));
        }

        // One extent, one layer count and one mip count for the whole target: the
        // render area, the viewport and every subresource view come from a single
        // set of numbers, so attachments that disagree have no correct answer.
        // Reported as a mismatch rather than silently intersected — a target half
        // the size of the texture handed in is never what was meant.
        const Image* first = colors.empty() ? depth.get() : colors[0].get();
        for (const auto& image : colors)
        {
            if (auto e = require_matching_attachment(*first, *image, "color"); !e)
            {
                return std::unexpected(e.error());
            }
        }
        if (depth)
        {
            if (auto e = require_matching_attachment(*first, *depth, "depth"); !e)
            {
                return std::unexpected(e.error());
            }
        }

        auto vk_samples = validate_sample_count(samples, context);
        if (!vk_samples)
        {
            return std::unexpected(vk_samples.error());
        }
        const bool msaa = *vk_samples != VK_SAMPLE_COUNT_1_BIT;
        // Same rule as create(): a multisampled image has no mip chain, so a
        // multisampled pass cannot resolve into a chosen level of one.
        if (msaa && first->mip_levels() > 1)
        {
            return std::unexpected(err_resource(
                std::format(
                    "samples>1 cannot combine with a mipped attachment: a multisampled image has "
                    "no mip chain, and this image has {} levels",
                    first->mip_levels())));
        }
        // A 3D color attachment is rendered one Z slice at a time through
        // target.layer(z), and the slice view is a 2D view of a 3D image — which
        // only a driver with imageView2DOn3DImage allows (full Vulkan always
        // does; MoltenVK does not). Gated at creation, because every use of this
        // target goes through that view.
        if (first->is_3d())
        {
            if (!context.supports(Feature::IMAGE_VIEW_2D_ON_3D))
            {
                return std::unexpected(err_unsupported(
                    "rendering into a 3D image needs the IMAGE_VIEW_2D_ON_3D feature, which this "
                    "driver does not offer. Ask ctx.supports(bz.Feature.IMAGE_VIEW_2D_ON_3D), and "
                    "fill the volume with a compute shader (image3D + imageStore) where it "
                    "answers False."));
            }
            // Depth testing against one slice of a volume is not a case any use
            // of a 3D target has; a depth-format 3D image is a validation
            // minefield, and the guard is one sentence (SCOPE, DESIGN.md 0.23).
            if (depth)
            {
                return std::unexpected(err_resource(
                    "a 3D color attachment cannot combine with a depth attachment. Render the "
                    "slices without depth, or use a layered 2D target."));
            }
            if (msaa)
            {
                return std::unexpected(err_resource("samples>1 cannot combine with a 3D attachment"));
            }
        }

        auto target = std::shared_ptr<OffscreenTarget>(new OffscreenTarget(context.shared_from_this()));
        target->extent_ = {first->width(), first->height()};
        target->samples_ = *vk_samples;
        target->layers_ = first->array_layers();
        target->slices_ = first->depth();
        target->mip_levels_ = first->mip_levels();
        target->colors_ = std::move(colors);
        target->depth_ = std::move(depth);

        // The borrowed images are the resolve targets; the multisampled
        // attachments beside them are allocated here and owned by the target,
        // exactly as on the allocating path. Layer count comes from the image, and
        // never cube: a multisampled attachment is a plain layered image that
        // resolves per layer.
        if (msaa)
        {
            for (std::size_t i = 0; i < target->colors_.size(); ++i)
            {
                auto ms = Image::create_empty(
                    context,
                    target->extent_.width,
                    target->extent_.height,
                    target->colors_[i]->format(),
                    1,
                    target->layers_,
                    false,
                    *vk_samples);
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
            if (target->depth_)
            {
                auto ms = Image::create_empty(
                    context,
                    target->extent_.width,
                    target->extent_.height,
                    target->depth_->format(),
                    1,
                    target->layers_,
                    false,
                    *vk_samples);
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

        if (!name.empty())
        {
            // Names the target's use of the image, and accumulates on the object the
            // same way a shared sampler's name does: an image may be an attachment
            // here and a texture somewhere else, and neither caller can predict the
            // other.
            for (std::size_t i = 0; i < target->colors_.size(); ++i)
            {
                context.set_debug_name(
                    VK_OBJECT_TYPE_IMAGE,
                    reinterpret_cast<std::uint64_t>(target->colors_[i]->vk_image()),
                    std::format("{} color[{}]", name, i));
            }
            if (target->depth_)
            {
                context.set_debug_name(
                    VK_OBJECT_TYPE_IMAGE,
                    reinterpret_cast<std::uint64_t>(target->depth_->vk_image()),
                    std::format("{} depth", name));
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
        return colors_[i]->vk_format();
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
        return depth_ ? depth_->vk_format() : VK_FORMAT_UNDEFINED;
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
    // A depth-only attachment is left sampleable, which is the whole of what
    // makes `shadow.depth` a texture. A combined depth/stencil one is not: its
    // view carries two aspects and no sampler can read it, so it stays in the
    // attachment layout and a second pass loads it with no round trip.
    VkImageLayout depth_final_layout() const override
    {
        if (depth_ && has_stencil(depth_->vk_format()))
        {
            return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        }
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
                err_resource("read_pixels() does not work on a depth-only RenderTarget. Read target.depth instead"));
        }
        return colors_[0]->read();
    }

    // Runs at execute() time, inside a real submit — the attachments learn they
    // have contents exactly when that becomes true (the 0.4.1 read_pixels fix,
    // now spelled per-Image). Depth included: that is what makes shadow maps
    // readable and sampleable.
    void on_rendering_recorded() override
    {
        mark_rendered(color_subresource(), depth_subresource());
    }

    // What a pass actually wrote, named by the target that ran it.
    //
    // Until 0.18 this marked the WHOLE image whatever the pass covered, so
    // `target.mip(1)` claimed mip 0 was in the final layout too — and the next
    // barrier over the whole image handed the driver an oldLayout that was true
    // of one level and a lie about the rest. That is why the old note said to
    // render every layer and every mip before sampling. A SubresourceTarget and
    // a MultiviewTarget call this with THEIR subresource, so the statement is
    // always made by whoever knows what was drawn.
    void record_even_out(const VolkDeviceTable& vk, VkCommandBuffer cmd) override
    {
        for (auto& image : colors_)
        {
            even_out_image(vk, cmd, *image, final_layout(), VK_IMAGE_ASPECT_COLOR_BIT);
        }
        if (depth_)
        {
            even_out_image(vk, cmd, *depth_, depth_final_layout(), aspect_mask_for(depth_->vk_format()));
        }
    }

    void mark_rendered(const Subresource& color_sr, const Subresource& depth_sr)
    {
        for (auto& image : colors_)
        {
            image->mark_subresource_contents(
                final_layout(), color_sr.base_layer, color_sr.layer_count, color_sr.base_mip, color_sr.mip_count);
        }
        if (depth_)
        {
            depth_->mark_subresource_contents(
                depth_final_layout(), depth_sr.base_layer, depth_sr.layer_count, depth_sr.base_mip, depth_sr.mip_count);
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
                [vk = &context_->vk(), device = context_->device(), views = std::move(views)]
                {
                    for (VkImageView v : views)
                    {
                        vk->vkDestroyImageView(device, v, nullptr);
                    }
                });
        }
    }

    // ── render-to-layer / render-to-mip ────────────────────────────────────────
    // A single (layer, mip) subresource of an attachment, as a plain 2D view —
    // what a SubresourceTarget renders into. Cached and owned here; slicing the
    // same subresource twice returns the same view. The *_view accessors return
    // the image actually rendered into (the multisampled one under MSAA); the
    // *_resolve_view accessors return the single-sample resolve subresource (null
    // without MSAA), so a layered MSAA target resolves each layer on its own.
    VkImageView color_subresource_view(std::uint32_t attachment, std::uint32_t layer, std::uint32_t mip)
    {
        const auto& image = msaa_colors_.empty() ? colors_[attachment] : msaa_colors_[attachment];
        return view_(image, VK_IMAGE_ASPECT_COLOR_BIT, layer, 1, mip);
    }
    VkImageView color_resolve_subresource_view(std::uint32_t attachment, std::uint32_t layer, std::uint32_t mip)
    {
        if (msaa_colors_.empty())
        {
            return VK_NULL_HANDLE;
        }
        return view_(colors_[attachment], VK_IMAGE_ASPECT_COLOR_BIT, layer, 1, mip);
    }
    VkImageView depth_subresource_view(std::uint32_t layer, std::uint32_t mip)
    {
        const auto& image = msaa_depth_ ? msaa_depth_ : depth_;
        if (!image)
        {
            return VK_NULL_HANDLE;
        }
        return view_(image, image->aspect(), layer, 1, mip);
    }
    VkImageView depth_resolve_subresource_view(std::uint32_t layer, std::uint32_t mip)
    {
        if (!msaa_depth_)
        {
            return VK_NULL_HANDLE;
        }
        return view_(depth_, depth_->aspect(), layer, 1, mip);
    }

    // Multiview attachment views: a 2D_ARRAY view over ALL layers at mip 0, what a
    // MultiviewTarget renders every layer through in one pass. The *_resolve_ ones
    // are the single-sample resolve targets (null without MSAA) — a multiview MSAA
    // pass resolves every view into the matching resolve layer.
    VkImageView color_array_view(std::uint32_t attachment)
    {
        const auto& image = msaa_colors_.empty() ? colors_[attachment] : msaa_colors_[attachment];
        return view_(image, VK_IMAGE_ASPECT_COLOR_BIT, 0, layers_, 0);
    }
    VkImageView color_resolve_array_view(std::uint32_t attachment)
    {
        if (msaa_colors_.empty())
        {
            return VK_NULL_HANDLE;
        }
        return view_(colors_[attachment], VK_IMAGE_ASPECT_COLOR_BIT, 0, layers_, 0);
    }
    VkImageView depth_array_view()
    {
        const auto& image = msaa_depth_ ? msaa_depth_ : depth_;
        if (!image)
        {
            return VK_NULL_HANDLE;
        }
        return view_(image, image->aspect(), 0, layers_, 0);
    }
    VkImageView depth_resolve_array_view()
    {
        if (!msaa_depth_)
        {
            return VK_NULL_HANDLE;
        }
        return view_(depth_, depth_->aspect(), 0, layers_, 0);
    }
    std::uint32_t array_layers() const
    {
        return layers_;
    }

    // Bounds-checked slices. Returned as a RenderTargetBase (the view is a
    // RenderTarget), so it passes straight into cmd.rendering(...). `layer(i, mip)`
    // is the general form (both axes); `mip(m)` is sugar for layer 0. A layered
    // AND mipped target (e.g. a mipped cube for prefiltered reflections) needs the
    // combined form.
    std::expected<std::shared_ptr<RenderTarget>, Error> layer(std::uint32_t i, std::uint32_t mip = 0);

    // Multiview: render into EVERY layer in one pass (the shader keys per-view work
    // off gl_ViewIndex) instead of a pass per layer. Needs a layered target and the
    // multiview GPU feature; composes with MSAA (resolves each view per layer).
    std::expected<std::shared_ptr<RenderTarget>, Error> all_layers();

    // True when the color attachment is a 3D image. Such a target is rendered
    // one Z slice at a time through layer(z): the slice view is legal, the 3D
    // main view as an attachment is not — which is why begin_rendering on the
    // whole target is refused at the binding layer.
    bool is_3d() const
    {
        return slices_ > 1;
    }
    std::uint32_t slices() const
    {
        return slices_;
    }

    const Context* owner() const override
    {
        return context_.get();
    }

private:
    explicit OffscreenTarget(std::shared_ptr<Context> context)
        : context_(std::move(context))
    {
    }

    // Every attachment of one target shares its extent, layer count and mip count.
    // Checked against the first attachment rather than pairwise, which is the same
    // result in fewer comparisons and gives a message naming a concrete reference.
    static std::expected<void, Error> require_matching_attachment(
        const Image& reference,
        const Image& image,
        const char* role)
    {
        if (image.width() != reference.width() || image.height() != reference.height())
        {
            return std::unexpected(err_resource(
                std::format(
                    "every attachment of a RenderTarget must be the same size: {} is {}x{}, "
                    "the first attachment is {}x{}",
                    role,
                    image.width(),
                    image.height(),
                    reference.width(),
                    reference.height())));
        }
        if (image.array_layers() != reference.array_layers())
        {
            return std::unexpected(err_resource(
                std::format(
                    "every attachment must have the same layer count: {} has {}, the first has {}",
                    role,
                    image.array_layers(),
                    reference.array_layers())));
        }
        // Also refuses mixing a 3D attachment with 2D ones: their Z extents
        // differ. "Deep", not "depth", because the depth ATTACHMENT is the
        // other meaning of the word in this message's own signature.
        if (image.depth() != reference.depth())
        {
            return std::unexpected(err_resource(
                std::format(
                    "every attachment must be equally deep: {} is {} deep, the first attachment "
                    "is {} (a 3D attachment cannot mix with 2D ones)",
                    role,
                    image.depth(),
                    reference.depth())));
        }
        if (image.mip_levels() != reference.mip_levels())
        {
            return std::unexpected(err_resource(
                std::format(
                    "every attachment must have the same mip count: {} has {}, the first has {}",
                    role,
                    image.mip_levels(),
                    reference.mip_levels())));
        }
        return {};
    }

    // Shared body of the view accessors: cache lookup keyed by (VkImage, baseLayer,
    // layerCount, mip) — the image handle disambiguates render vs resolve vs depth —
    // on miss create a single-mip view (2D for one layer, 2D_ARRAY for a multiview
    // span) and store it.
    VkImageView view_(
        const std::shared_ptr<Image>& image,
        VkImageAspectFlags aspect,
        std::uint32_t base_layer,
        std::uint32_t layer_count,
        std::uint32_t mip)
    {
        auto key = std::tuple{reinterpret_cast<std::uint64_t>(image->vk_image()), base_layer, layer_count, mip};
        if (auto it = subresource_views_.find(key); it != subresource_views_.end())
        {
            return it->second;
        }
        VkImageViewCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = image->vk_image(),
            .viewType = layer_count > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D,
            .format = image->vk_format(),
            .components = {},
            .subresourceRange = {aspect, mip, 1, base_layer, layer_count}};
        VkImageView view = VK_NULL_HANDLE;
        // Bounds are checked in layer()/mip() before we get here; a create failure
        // is a genuine driver error, so surface a null and let the caller's
        // validation-as-assert catch the bad attachment rather than crashing.
        if (context_->vk().vkCreateImageView(context_->device(), &info, nullptr, &view) != VK_SUCCESS)
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
    // Z slices of a 3D attachment (1 for every 2D target). A volume has one
    // array layer, so layers_ and slices_ are never both > 1; layer(i) means
    // "slice i" exactly when slices_ > 1.
    std::uint32_t slices_ = 1;

    // Lazily created views, keyed (VkImage handle, base layer, layer count, mip).
    // Owned here, destroyed (deferred) in the destructor.
    std::map<std::tuple<std::uint64_t, std::uint32_t, std::uint32_t, std::uint32_t>, VkImageView> subresource_views_;
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
    const Context* owner() const override
    {
        return parent_->owner();
    }

    SubresourceTarget(std::shared_ptr<OffscreenTarget> parent, std::uint32_t layer, std::uint32_t mip)
        : parent_(std::move(parent)),
          layer_(layer),
          mip_(mip)
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

    // MSAA composes with layers: the multisampled attachment and its single-sample
    // resolve target are both sliced to this subresource, so CommandBuffer resolves
    // exactly this layer. Without MSAA these return 1 / VK_NULL_HANDLE (from the
    // parent) and the resolve wiring stays off.
    VkSampleCountFlagBits samples() const override
    {
        return parent_->samples();
    }
    VkImage color_resolve_image(std::uint32_t i) const override
    {
        return parent_->color_resolve_image(i);
    }
    VkImageView color_resolve_view(std::uint32_t i) const override
    {
        return parent_->color_resolve_subresource_view(i, layer_, mip_);
    }
    VkImage depth_resolve_image() const override
    {
        return parent_->depth_resolve_image();
    }
    VkImageView depth_resolve_view() const override
    {
        return parent_->depth_resolve_subresource_view(layer_, mip_);
    }

    VkImageLayout final_layout() const override
    {
        return parent_->final_layout();
    }
    VkImageLayout depth_final_layout() const override
    {
        return parent_->depth_final_layout();
    }

    // For a 3D parent the view axis and the barrier axis diverge, on purpose —
    // the one exception to the 0.13 "view and barrier come from one (layer,
    // mip)" rule. The slice index feeds only the VIEW (baseArrayLayer selects
    // the Z slice of a 2D_ARRAY_COMPATIBLE volume); Vulkan tracks the layout of
    // a 3D image per mip with exactly one array layer, so the barrier and the
    // marking must name layer 0 or they would index past the layout state. The
    // count is spelled VK_REMAINING_ARRAY_LAYERS for a volume — the layers warn
    // about the narrower form — and mark_subresource_contents clamps it back.
    // Consequence: rendering one slice marks the whole mip — correct, because
    // that IS the granularity a volume's layout has.
    Subresource color_subresource() const override
    {
        return parent_->is_3d() ? Subresource{0, VK_REMAINING_ARRAY_LAYERS, mip_, 1} : Subresource{layer_, 1, mip_, 1};
    }
    Subresource depth_subresource() const override
    {
        return parent_->is_3d() ? Subresource{0, VK_REMAINING_ARRAY_LAYERS, mip_, 1} : Subresource{layer_, 1, mip_, 1};
    }

    // Tells the parent that exactly THIS layer and mip are now in the final
    // layout — not the whole image, which is what it used to say and what made
    // "render every layer before you sample" a rule rather than an optimization.
    // The Image collapses back to one layout as soon as the last subresource
    // catches up, so a fully rendered target still costs one barrier.
    void on_rendering_recorded() override
    {
        parent_->mark_rendered(color_subresource(), depth_subresource());
    }

    void record_even_out(const VolkDeviceTable& vk, VkCommandBuffer cmd) override
    {
        parent_->record_even_out(vk, cmd);
    }

private:
    std::shared_ptr<OffscreenTarget> parent_;
    std::uint32_t layer_;
    std::uint32_t mip_;
};

// Renders into EVERY layer of an OffscreenTarget in one pass via multiview — what
// target.all_layers() hands back. The attachment views span all layers (2D_ARRAY),
// view_mask() lights one bit per layer, and the barriers cover the whole array; the
// shader selects per-layer work with gl_ViewIndex. Composes with MSAA (each view
// resolves into its own layer), and because it renders every layer, the whole-image
// sampleable mark is exactly correct — no partial-render caveat.
class MultiviewTarget : public RenderTarget
{
public:
    const Context* owner() const override
    {
        return parent_->owner();
    }

    explicit MultiviewTarget(std::shared_ptr<OffscreenTarget> parent)
        : parent_(std::move(parent))
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
        return parent_->color_array_view(i);
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
        return parent_->depth_array_view();
    }
    VkFormat depth_format() const override
    {
        return parent_->depth_format();
    }
    VkExtent2D extent() const override
    {
        return parent_->extent();
    }

    std::uint32_t view_mask() const override
    {
        const std::uint32_t n = parent_->array_layers();
        return n >= 32 ? 0xFFFFFFFFu : ((1u << n) - 1u);
    }

    // MSAA composes with multiview: one pass renders every layer of the
    // multisampled attachment and resolves each view into the matching resolve
    // layer. The array views span all layers, so the resolve is per-view. Without
    // MSAA these come back 1 / VK_NULL_HANDLE from the parent (resolve wiring off).
    VkSampleCountFlagBits samples() const override
    {
        return parent_->samples();
    }
    VkImage color_resolve_image(std::uint32_t i) const override
    {
        return parent_->color_resolve_image(i);
    }
    VkImageView color_resolve_view(std::uint32_t i) const override
    {
        return parent_->color_resolve_array_view(i);
    }
    VkImage depth_resolve_image() const override
    {
        return parent_->depth_resolve_image();
    }
    VkImageView depth_resolve_view() const override
    {
        return parent_->depth_resolve_array_view();
    }

    Subresource color_subresource() const override
    {
        return {0, parent_->array_layers(), 0, 1};
    }
    Subresource depth_subresource() const override
    {
        return {0, parent_->array_layers(), 0, 1};
    }

    VkImageLayout final_layout() const override
    {
        return parent_->final_layout();
    }
    VkImageLayout depth_final_layout() const override
    {
        return parent_->depth_final_layout();
    }

    // Multiview writes every layer in one pass, so its subresource already spans
    // the whole array and the parent marks all of it.
    void on_rendering_recorded() override
    {
        parent_->mark_rendered(color_subresource(), depth_subresource());
    }

    void record_even_out(const VolkDeviceTable& vk, VkCommandBuffer cmd) override
    {
        parent_->record_even_out(vk, cmd);
    }

private:
    std::shared_ptr<OffscreenTarget> parent_;
};

inline std::expected<std::shared_ptr<RenderTarget>, Error> OffscreenTarget::layer(std::uint32_t i, std::uint32_t mip)
{
    if (mip >= mip_levels_)
    {
        return std::unexpected(
            err_resource(std::format("mip {} is out of range. This target has {} mip level(s)", mip, mip_levels_)));
    }
    // On a 3D target the layer axis IS the slice axis, and unlike array layers
    // it shrinks with the mip: level 1 of a depth-4 volume has 2 slices. A
    // volume has one array layer, so one bound covers both kinds of target.
    const std::uint32_t extent = slices_ > 1 ? (std::ranges::max)(slices_ >> mip, 1u) : layers_;
    if (i >= extent)
    {
        return std::unexpected(err_resource(
            std::format(
                "layer {} is out of range. Mip {} of this target has {} {}",
                i,
                mip,
                extent,
                slices_ > 1 ? "slice(s)" : "layer(s)")));
    }
    return std::make_shared<SubresourceTarget>(shared_from_this(), i, mip);
}

inline std::expected<std::shared_ptr<RenderTarget>, Error> OffscreenTarget::all_layers()
{
    // Multiview routes one draw into array layers, and a volume has exactly
    // one: its slices are not layers, so there is nothing for a view mask to
    // light up. Render slice by slice through layer(z) instead.
    if (slices_ > 1)
    {
        return std::unexpected(err_resource(
            "all_layers() has no meaning on a 3D target: a volume has one array layer. "
            "Render one slice at a time with target.layer(z)."));
    }
    if (!context_->supports(Feature::MULTIVIEW))
    {
        return std::unexpected(err_unsupported(
            "all_layers() needs the multiview GPU feature, which this device does not support. "
            "Ask ctx.supports(bz.Feature.MULTIVIEW) first, and render one layer at a time if it "
            "answers False."));
    }
    if (layers_ <= 1)
    {
        return std::unexpected(
            err_resource("all_layers() needs a layered target (layers>1 or cube). This target has 1 layer"));
    }
    return std::make_shared<MultiviewTarget>(shared_from_this());
}
