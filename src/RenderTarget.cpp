#include "RenderTarget.hpp"

#include <algorithm>
#include <format>

std::expected<VkSampleCountFlagBits, Error> validate_sample_count(std::uint32_t samples, const Context& context)
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

void even_out_image(
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

std::expected<std::shared_ptr<OffscreenTarget>, Error> OffscreenTarget::create(
    Context& context,
    std::uint32_t width,
    std::uint32_t height,
    std::vector<Format> colors,
    std::optional<Format> depth,
    std::uint32_t samples,
    std::uint32_t layers,
    bool cube,
    std::uint32_t mip_levels,
    const std::string& name,
    bool keep_samples)
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
            std::format("layers and mip_levels must be at least 1. Got layers={}, mip_levels={}", layers, mip_levels)));
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

    // keep_samples without MSAA has nothing to keep. Silently false rather than
    // an error: it is a hint about a multisampled attachment that does not
    // exist, so there is no wrong result to warn about.
    if (keep_samples && !msaa)
    {
        keep_samples = false;
    }

    auto target = std::shared_ptr<OffscreenTarget>(new OffscreenTarget(context.shared_from_this()));
    target->extent_ = {.width = width, .height = height};
    target->samples_ = *vk_samples;
    target->layers_ = layers;
    target->mip_levels_ = mip_levels;
    target->keep_samples_ = keep_samples;

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

std::expected<std::shared_ptr<OffscreenTarget>, Error> OffscreenTarget::create_from_images(
    Context& context,
    std::vector<std::shared_ptr<Image>> colors,
    std::shared_ptr<Image> depth,
    std::uint32_t samples,
    const std::string& name,
    bool keep_samples)
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
    target->extent_ = {.width = first->width(), .height = first->height()};
    target->samples_ = *vk_samples;
    target->layers_ = first->array_layers();
    target->slices_ = first->depth();
    target->mip_levels_ = first->mip_levels();
    target->colors_ = std::move(colors);
    target->depth_ = std::move(depth);
    target->keep_samples_ = keep_samples && msaa;

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

VkImage OffscreenTarget::depth_image() const
{
    if (msaa_depth_)
    {
        return msaa_depth_->vk_image();
    }
    return depth_ ? depth_->vk_image() : VK_NULL_HANDLE;
}

VkImageView OffscreenTarget::depth_view() const
{
    if (msaa_depth_)
    {
        return msaa_depth_->view();
    }
    return depth_ ? depth_->view() : VK_NULL_HANDLE;
}

VkImageLayout OffscreenTarget::depth_final_layout() const
{
    if (depth_ && has_stencil(depth_->vk_format()))
    {
        return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }
    return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

const std::vector<std::shared_ptr<Image>>& OffscreenTarget::multisampled_colors() const
{
    static const std::vector<std::shared_ptr<Image>> none;
    return keep_samples_ ? msaa_colors_ : none;
}

const std::shared_ptr<Image>& OffscreenTarget::multisampled_depth() const
{
    static const std::shared_ptr<Image> none;
    return keep_samples_ ? msaa_depth_ : none;
}

std::expected<std::vector<std::byte>, Error> OffscreenTarget::read_pixels()
{
    if (colors_.empty())
    {
        return std::unexpected(
            err_resource("read_pixels() does not work on a depth-only RenderTarget. Read target.depth instead"));
    }
    return colors_[0]->read();
}

void OffscreenTarget::record_even_out(const VolkDeviceTable& vk, VkCommandBuffer cmd)
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

void OffscreenTarget::mark_rendered(const Subresource& color_sr, const Subresource& depth_sr)
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
    if (!keep_samples_)
    {
        return;
    }
    // The multisampled attachments end SHADER_READ_ONLY_OPTIMAL, which is what
    // end_rendering retires them to and not what final_layout() says: a
    // swapchain's final layout is PRESENT_SRC_KHR, and no multisampled image is
    // ever presented. The tracker reads these, so a mark that disagreed with
    // the barrier would hand the next use a stale oldLayout.
    for (auto& image : msaa_colors_)
    {
        image->mark_subresource_contents(
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            color_sr.base_layer,
            color_sr.layer_count,
            color_sr.base_mip,
            color_sr.mip_count);
    }
    // Only when the pass actually retired it — the swapchain's scratch depth
    // keeps its attachment layout, and end_rendering skips the whole depth
    // block there.
    if (msaa_depth_ && depth_ && depth_final_layout() != attachment_depth_layout_())
    {
        msaa_depth_->mark_subresource_contents(
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            depth_sr.base_layer,
            depth_sr.layer_count,
            depth_sr.base_mip,
            depth_sr.mip_count);
    }
}

OffscreenTarget::~OffscreenTarget()
{
    // The per-subresource views are the only Vulkan objects OffscreenTarget owns
    // beyond its Images (which self-destruct, deferred). A SubresourceTarget
    // borrows these — they live exactly as long as the parent, so it never owns
    // Vulkan objects itself and can be created/discarded freely.
    //
    // Wrapped, because this allocates twice (the handle list, then the closure
    // the deletion queue stores) and a destructor may not throw. Under memory
    // exhaustion the views leak, which is strictly better than std::terminate
    // during teardown — and by then the process has a larger problem.
    try
    {
        retire_subresource_views_();
    }
    catch (...)
    {
        // Catch-all rather than bad_alloc alone, and the difference is not
        // paranoia: a destructor that can throw ANY type makes every closure
        // holding a shared_ptr to this target throw on destruction too, which
        // is how clang-tidy reported it — three recorded lambdas, none of which
        // had a throwing line of its own.
        //
        // Nothing to retry with: recording the destroy needs the memory that
        // just ran out. The views then live until vkDestroyDevice takes them,
        // which is the same end they were headed for.
        subresource_views_.clear();
    }
}

void OffscreenTarget::retire_subresource_views_()
{
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

VkImageView OffscreenTarget::color_subresource_view(std::uint32_t attachment, std::uint32_t layer, std::uint32_t mip)
{
    const auto& image = msaa_colors_.empty() ? colors_[attachment] : msaa_colors_[attachment];
    return view_(image, VK_IMAGE_ASPECT_COLOR_BIT, layer, 1, mip);
}

VkImageView OffscreenTarget::color_resolve_subresource_view(
    std::uint32_t attachment,
    std::uint32_t layer,
    std::uint32_t mip)
{
    if (msaa_colors_.empty())
    {
        return VK_NULL_HANDLE;
    }
    return view_(colors_[attachment], VK_IMAGE_ASPECT_COLOR_BIT, layer, 1, mip);
}

VkImageView OffscreenTarget::depth_subresource_view(std::uint32_t layer, std::uint32_t mip)
{
    const auto& image = msaa_depth_ ? msaa_depth_ : depth_;
    if (!image)
    {
        return VK_NULL_HANDLE;
    }
    return view_(image, image->aspect(), layer, 1, mip);
}

VkImageView OffscreenTarget::depth_resolve_subresource_view(std::uint32_t layer, std::uint32_t mip)
{
    if (!msaa_depth_)
    {
        return VK_NULL_HANDLE;
    }
    return view_(depth_, depth_->aspect(), layer, 1, mip);
}

VkImageView OffscreenTarget::color_array_view(std::uint32_t attachment)
{
    const auto& image = msaa_colors_.empty() ? colors_[attachment] : msaa_colors_[attachment];
    return view_(image, VK_IMAGE_ASPECT_COLOR_BIT, 0, layers_, 0);
}

VkImageView OffscreenTarget::color_resolve_array_view(std::uint32_t attachment)
{
    if (msaa_colors_.empty())
    {
        return VK_NULL_HANDLE;
    }
    return view_(colors_[attachment], VK_IMAGE_ASPECT_COLOR_BIT, 0, layers_, 0);
}

VkImageView OffscreenTarget::depth_array_view()
{
    const auto& image = msaa_depth_ ? msaa_depth_ : depth_;
    if (!image)
    {
        return VK_NULL_HANDLE;
    }
    return view_(image, image->aspect(), 0, layers_, 0);
}

VkImageView OffscreenTarget::depth_resolve_array_view()
{
    if (!msaa_depth_)
    {
        return VK_NULL_HANDLE;
    }
    return view_(depth_, depth_->aspect(), 0, layers_, 0);
}

std::expected<std::shared_ptr<RenderTarget>, Error> OffscreenTarget::layer(std::uint32_t i, std::uint32_t mip)
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

std::expected<std::shared_ptr<RenderTarget>, Error> OffscreenTarget::all_layers()
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

std::expected<void, Error> OffscreenTarget::require_matching_attachment(
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

VkImageView OffscreenTarget::view_(
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

VkExtent2D SubresourceTarget::extent() const
{
    VkExtent2D e = parent_->extent();
    std::uint32_t w = e.width >> mip_;
    std::uint32_t h = e.height >> mip_;
    return {w ? w : 1u, h ? h : 1u};
}
