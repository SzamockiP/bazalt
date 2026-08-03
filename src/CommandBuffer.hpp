#pragma once
#include <volk.h>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>
#include <array>
#include <expected>
#include <functional>
#include <string>
#include "Context.hpp"
#include "RenderTarget.hpp"
#include "Pipeline.hpp"
#include "Buffer.hpp"
#include "DescriptorSet.hpp"
#include "ResourceTracker.hpp"

// Records commands once and replays them every submit.
//
// The recorded lambdas take a FrameContext rather than a SwapchainRenderer&.
// That single change is what lets the same command buffer be replayed against a
// window, an offscreen image, or (later) a compute-only submit: this file no
// longer knows that swapchains exist.
class CommandBuffer
{
public:
    // Takes a Context, not a renderer: command buffers are a device resource and
    // have nothing to do with presentation. This is what lets a headless Context
    // with no renderer at all record commands.
    static std::expected<std::shared_ptr<CommandBuffer>, Error> create(
        Context& context,
        std::optional<bool> auto_barriers = std::nullopt)
    {
        auto ctx = context.shared_from_this();
        auto cmd = std::shared_ptr<CommandBuffer>(new CommandBuffer(ctx));
        // Per-command-buffer override of the Context-wide mode, so one hot
        // path can go manual without flipping the whole application.
        cmd->auto_barriers_ = auto_barriers.value_or(ctx->auto_barriers());
        // Told once, here and not in begin(): the mask describes the device, not
        // the recording, so tracker_.reset() must leave it alone.
        cmd->tracker_.set_all_shader_stages(ctx->all_shader_stages());
        cmd->command_buffers_.resize(ctx->frames_in_flight(), VK_NULL_HANDLE);

        VkCommandBufferAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr,
            .commandPool = ctx->command_pool(),
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = ctx->frames_in_flight()};

        if (auto e = check(
                ctx->vk().vkAllocateCommandBuffers(ctx->device(), &allocInfo, cmd->command_buffers_.data()),
                "allocate command buffers",
                ErrorCode::Resource))
        {
            return std::unexpected(*e);
        }

        return cmd;
    }

    // Deferred: a per-frame VkCommandBuffer may still be executing when the
    // Python object is dropped.
    ~CommandBuffer()
    {
        if (context_)
        {
            if (timer_pool_ != VK_NULL_HANDLE)
            {
                context_->defer_destroy([vk = &context_->vk(), device = context_->device(), pool = timer_pool_]
                                        { vk->vkDestroyQueryPool(device, pool, nullptr); });
            }
            if (occlusion_pool_ != VK_NULL_HANDLE)
            {
                context_->defer_destroy([vk = &context_->vk(), device = context_->device(), pool = occlusion_pool_]
                                        { vk->vkDestroyQueryPool(device, pool, nullptr); });
            }
            context_->defer_destroy(
                [vk = &context_->vk(),
                 device = context_->device(),
                 pool = context_->command_pool(),
                 buffers = std::move(command_buffers_)]
                { vk->vkFreeCommandBuffers(device, pool, static_cast<uint32_t>(buffers.size()), buffers.data()); });
        }
    }

    CommandBuffer(const CommandBuffer&) = delete;
    CommandBuffer& operator=(const CommandBuffer&) = delete;

    // Which Context this command buffer records for; see Buffer::owner().
    const Context* owner() const
    {
        return context_.get();
    }

    CommandBuffer& begin()
    {
        commands_.clear();
        used_sets_.clear();
        used_buffers_.clear();
        // A reused command buffer must forget its previous recording entirely,
        // or it would emit barriers against uses that no longer exist.
        tracker_.reset();
        bound_graphics_sets_.clear();
        bound_compute_sets_.clear();
        bound_graphics_pipeline_.reset();
        bound_compute_pipeline_.reset();
        bound_last_pipeline_.reset();
        tracked_writes_ = false;
        in_rendering_ = false;
        rendering_insert_pos_ = 0;
        // Timers are re-declared each recording; the query pool itself is kept
        // and reset (vkCmdResetQueryPool) at the top of every replay. Bumping
        // the generation invalidates handles from the previous recording.
        timer_count_ = 0;
        // Same story for occlusion queries and for label nesting: both are
        // properties of one recording.
        occlusion_count_ = 0;
        open_labels_ = 0;
        ++recording_generation_;
        return *this;
    }

    // The target is explicit. It used to default to "the swapchain" implicitly,
    // which made presentation a special case dressed up as the default and left
    // no way to name anything else. Naming what you draw into costs one token
    // and buys one rule that holds everywhere.
    // One clear per colour attachment. Empty → black; a single entry clears every
    // attachment (the common case); N entries clear attachment i with entry i
    // (per-attachment clears for MRT). The binding accepts both [r,g,b,a] and
    // [[r,g,b,a], …] and normalises to this.
    //
    // nullopt means PRESERVE: load what the attachment already holds instead of
    // clearing it, which is what puts a second pass on one target (opaque then
    // transparent, or a UI over a scene). It has to be a distinct state from an
    // empty vector, because empty already means "clear to black".
    //
    // Colour and depth preserve together. Splitting them would be two knobs on
    // the verb for one question, and the multi-pass case wants both.
    //
    // clear_depth is the value, not a second preserve switch: 1.0 is the far
    // plane and the default, 0.0 is what a reversed-depth buffer starts from
    // (which is the only way depth_test(compare=GREATER) can ever pass). It is
    // ignored when the pass preserves.
    CommandBuffer& begin_rendering(
        std::shared_ptr<RenderTarget> target,
        const std::optional<std::vector<std::array<float, 4>>>& clear_colors,
        float clear_depth = 1.0f,
        std::uint32_t clear_stencil = 0)
    {
        commands_.push_back(
            [clear_colors, clear_depth, clear_stencil, target](VkCommandBuffer cmd, const FrameContext& frame)
            {
                RenderTarget* rt = target.get();
                const bool preserve = !clear_colors.has_value();

                // A depth attachment that carries a stencil aspect is one image
                // in one layout: DEPTH_ATTACHMENT_OPTIMAL covers the depth
                // aspect only, so a combined format needs the combined layout,
                // and every barrier, view and attachment info below reads both
                // from here.
                const VkImageAspectFlags depth_aspect = aspect_mask_for(rt->depth_format());
                const bool stencil = has_stencil(rt->depth_format());
                const VkImageLayout depth_layout = stencil ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                                                           : VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;

                // Which layer/mip each attachment barrier must transition. Defaults
                // to {layer 0, mip 0, one of each}; a SubresourceTarget narrows it to
                // the single subresource its view renders into (render-to-layer/mip).
                const RenderTarget::Subresource color_sr = rt->color_subresource();
                const RenderTarget::Subresource depth_sr = rt->depth_subresource();

                // Every colour attachment enters COLOR_ATTACHMENT_OPTIMAL. UNDEFINED
                // as the source: contents are cleared each pass anyway.
                //
                // Except when preserving, where UNDEFINED discards exactly what
                // is about to be loaded. The source is then the layout the
                // previous pass retired to (end_rendering below), and the source
                // stage covers both ways the attachment can have got there:
                // written by an earlier pass, or sampled since.
                const VkImageLayout color_old_layout = preserve ? rt->final_layout() : VK_IMAGE_LAYOUT_UNDEFINED;
                const VkAccessFlags color_src_access =
                    preserve ? (VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT) : 0;
                const VkPipelineStageFlags color_src_stage =
                    preserve ? (VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)
                             : VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

                for (uint32_t i = 0; i < rt->color_count(); ++i)
                {
                    VkImageMemoryBarrier barrier{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        .pNext = nullptr,
                        .srcAccessMask = color_src_access,
                        // LOAD_OP_LOAD reads the attachment, so preserving needs
                        // the read bit as well as the write.
                        .dstAccessMask = static_cast<VkAccessFlags>(
                            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            (preserve ? VK_ACCESS_COLOR_ATTACHMENT_READ_BIT : 0)),
                        .oldLayout = color_old_layout,
                        .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = rt->color_image(i),
                        .subresourceRange = {
                            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .baseMipLevel = color_sr.base_mip,
                            .levelCount = color_sr.mip_count,
                            .baseArrayLayer = color_sr.base_layer,
                            .layerCount = color_sr.layer_count}};

                    frame.vk->vkCmdPipelineBarrier(
                        cmd,
                        color_src_stage,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        0,
                        0,
                        nullptr,
                        0,
                        nullptr,
                        1,
                        &barrier);

                    // With MSAA the single-sample resolve target is a second
                    // attachment written this pass — it needs the same transition.
                    if (rt->color_resolve_image(i) != VK_NULL_HANDLE)
                    {
                        barrier.image = rt->color_resolve_image(i);
                        frame.vk->vkCmdPipelineBarrier(
                            cmd,
                            color_src_stage,
                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                            0,
                            0,
                            nullptr,
                            0,
                            nullptr,
                            1,
                            &barrier);
                    }
                }

                // Depth follows colour: preserving takes it from the layout the
                // previous pass left it in, and a swapchain's scratch depth never
                // leaves DEPTH_ATTACHMENT_OPTIMAL, which is exactly what
                // depth_final_layout() reports for it.
                const VkImageLayout depth_old_layout = preserve ? rt->depth_final_layout() : VK_IMAGE_LAYOUT_UNDEFINED;
                const VkAccessFlags depth_src_access =
                    preserve ? (VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT) : 0;
                const VkPipelineStageFlags depth_src_stage =
                    preserve ? (VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)
                             : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

                if (rt->depth_image() != VK_NULL_HANDLE)
                {
                    VkImageMemoryBarrier depthBarrier{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        .pNext = nullptr,
                        .srcAccessMask = depth_src_access,
                        .dstAccessMask = static_cast<VkAccessFlags>(
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                            (preserve ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT : 0)),
                        .oldLayout = depth_old_layout,
                        .newLayout = depth_layout,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = rt->depth_image(),
                        .subresourceRange = {
                            .aspectMask = depth_aspect,
                            .baseMipLevel = depth_sr.base_mip,
                            .levelCount = depth_sr.mip_count,
                            .baseArrayLayer = depth_sr.base_layer,
                            .layerCount = depth_sr.layer_count}};

                    frame.vk->vkCmdPipelineBarrier(
                        cmd,
                        depth_src_stage,
                        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                        0,
                        0,
                        nullptr,
                        0,
                        nullptr,
                        1,
                        &depthBarrier);

                    // MSAA depth resolves into a single-sample image (offscreen
                    // only — a swapchain's scratch depth has no resolve target).
                    if (rt->depth_resolve_image() != VK_NULL_HANDLE)
                    {
                        depthBarrier.image = rt->depth_resolve_image();
                        frame.vk->vkCmdPipelineBarrier(
                            cmd,
                            depth_src_stage,
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                            0,
                            0,
                            nullptr,
                            0,
                            nullptr,
                            1,
                            &depthBarrier);
                    }
                }

                std::vector<VkRenderingAttachmentInfo> colorAttachments;
                colorAttachments.reserve(rt->color_count());
                for (uint32_t i = 0; i < rt->color_count(); ++i)
                {
                    const std::array<float, 4> cc = preserve || clear_colors->empty()
                                                        ? std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}
                                                        : (i < clear_colors->size() ? (*clear_colors)[i]
                                                                                    : (*clear_colors)[0]);
                    // MSAA: render into the multisampled view, resolve (averaging
                    // the samples) into the single-sample target. The multisampled
                    // image is discarded afterwards unless the target asked to keep
                    // it — see the store-op below.
                    const bool resolve = rt->color_resolve_view(i) != VK_NULL_HANDLE;
                    colorAttachments.push_back(
                        {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                         .pNext = nullptr,
                         .imageView = rt->color_view(i),
                         .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                         .resolveMode = resolve ? VK_RESOLVE_MODE_AVERAGE_BIT : VK_RESOLVE_MODE_NONE,
                         .resolveImageView = resolve ? rt->color_resolve_view(i) : VK_NULL_HANDLE,
                         .resolveImageLayout = resolve ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                                                       : VK_IMAGE_LAYOUT_UNDEFINED,
                         .loadOp = preserve ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR,
                         // DONT_CARE on the multisampled attachment is what makes
                         // MSAA cheap: on a tiler the samples never leave tile
                         // memory, and the resolve is the only thing written out.
                         // A custom resolve needs them written out, so the target
                         // says so once (keep_samples=True) and pays for it there.
                         // Deriving this from "did anyone bind the multisampled
                         // image" is not available — that happens in another
                         // recording, or in another frame.
                         .storeOp = (resolve && !rt->keep_samples()) ? VK_ATTACHMENT_STORE_OP_DONT_CARE
                                                                     : VK_ATTACHMENT_STORE_OP_STORE,
                         .clearValue = {.color = {{cc[0], cc[1], cc[2], cc[3]}}}});
                }

                // Depth resolve uses SAMPLE_ZERO (averaging depth is meaningless and
                // not guaranteed; taking sample 0 always is). Only offscreen targets
                // resolve depth — the swapchain's scratch depth has no resolve view.
                const bool depthResolve = rt->depth_resolve_view() != VK_NULL_HANDLE;
                VkRenderingAttachmentInfo depthAttachment{
                    .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                    .pNext = nullptr,
                    .imageView = rt->depth_view(),
                    .imageLayout = depth_layout,
                    .resolveMode = depthResolve ? VK_RESOLVE_MODE_SAMPLE_ZERO_BIT : VK_RESOLVE_MODE_NONE,
                    .resolveImageView = depthResolve ? rt->depth_resolve_view() : VK_NULL_HANDLE,
                    .resolveImageLayout = depthResolve ? depth_layout : VK_IMAGE_LAYOUT_UNDEFINED,
                    .loadOp = preserve ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR,
                    // Always stored. It used to be DONT_CARE unless the depth
                    // would be consumed (shadow maps), which is the cheaper
                    // choice right up until a second pass preserves it: DONT_CARE
                    // makes the depth undefined the moment the first pass ends, so
                    // opaque-then-transparent on one target would z-test against
                    // garbage. The cost is depth bandwidth on tiled GPUs, and the
                    // upgrade path is deriving the store-op from whether a later
                    // pass in the same recording loads.
                    .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                    .clearValue = {.depthStencil = {clear_depth, clear_stencil}}};

                // The stencil aspect of the same image, named separately because
                // dynamic rendering takes two attachment pointers. It follows the
                // depth attachment in everything except which half of the clear
                // value it reads — one image, one layout, one load-op, so a pass
                // cannot preserve depth while clearing stencil.
                VkRenderingAttachmentInfo stencilAttachment = depthAttachment;

                VkRenderingInfo renderingInfo{
                    .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .renderArea = {{0, 0}, rt->extent()},
                    // Multiview renders every set layer in one pass (viewMask != 0);
                    // layerCount is then ignored. 0 keeps the ordinary single-layer path.
                    .layerCount = 1,
                    .viewMask = rt->view_mask(),
                    .colorAttachmentCount = static_cast<uint32_t>(colorAttachments.size()),
                    .pColorAttachments = colorAttachments.empty() ? nullptr : colorAttachments.data(),
                    .pDepthAttachment = rt->depth_view() != VK_NULL_HANDLE ? &depthAttachment : nullptr,
                    .pStencilAttachment = (stencil && rt->depth_view() != VK_NULL_HANDLE) ? &stencilAttachment
                                                                                          : nullptr};

                frame.vk->vkCmdBeginRendering(cmd, &renderingInfo);

                // Emitted automatically: set_viewport()/set_scissor() took no arguments
                // and silently read the swapchain, which is magic â€” just less legible
                // than doing it here. set_viewport(x, y, w, h) remains for the cases
                // that genuinely want something other than the whole target.
                VkViewport viewport{
                    .x = 0.0f,
                    .y = 0.0f,
                    .width = static_cast<float>(rt->extent().width),
                    .height = static_cast<float>(rt->extent().height),
                    .minDepth = 0.0f,
                    .maxDepth = 1.0f};
                frame.vk->vkCmdSetViewport(cmd, 0, 1, &viewport);

                VkRect2D scissor{.offset = {0, 0}, .extent = rt->extent()};
                frame.vk->vkCmdSetScissor(cmd, 0, 1, &scissor);
            });
        // vkCmdPipelineBarrier is illegal inside a dynamic rendering scope, so
        // auto barriers discovered between begin and end are hoisted to just
        // before this lambda (record_barrier_ reads these two fields).
        in_rendering_ = true;
        rendering_insert_pos_ = commands_.size() - 1;
        return *this;
    }

    CommandBuffer& end_rendering(std::shared_ptr<RenderTarget> target)
    {
        commands_.push_back(
            [target](VkCommandBuffer cmd, const FrameContext& frame)
            {
                frame.vk->vkCmdEndRendering(cmd);

                const RenderTarget::Subresource color_sr = target->color_subresource();
                const RenderTarget::Subresource depth_sr = target->depth_subresource();

                // Every colour attachment retires to the target's final layout.
                // (Was VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, unconditionally, on colour 0
                // only â€” that one constant is why nothing but a swapchain could ever
                // be drawn into.)
                for (uint32_t i = 0; i < target->color_count(); ++i)
                {
                    // With MSAA it is the resolve image that must reach the final
                    // layout, because that is the one that gets presented.
                    VkImage final_image = target->color_resolve_image(i) != VK_NULL_HANDLE
                                              ? target->color_resolve_image(i)
                                              : target->color_image(i);
                    VkImageMemoryBarrier barrier{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        .pNext = nullptr,
                        .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                        .dstAccessMask = 0,
                        .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        .newLayout = target->final_layout(),
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = final_image,
                        .subresourceRange = {
                            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .baseMipLevel = color_sr.base_mip,
                            .levelCount = color_sr.mip_count,
                            .baseArrayLayer = color_sr.base_layer,
                            .layerCount = color_sr.layer_count}};

                    frame.vk->vkCmdPipelineBarrier(
                        cmd,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                        0,
                        0,
                        nullptr,
                        0,
                        nullptr,
                        1,
                        &barrier);

                    // A kept multisampled image retires too, since 0.25, because it
                    // is then readable: target.multisampled_color[i] goes into a
                    // sampler2DMS for a custom resolve. It goes to
                    // SHADER_READ_ONLY_OPTIMAL rather than to final_layout(), and
                    // that difference is the point — a swapchain's final layout is
                    // PRESENT_SRC_KHR, and a multisampled image is never the thing
                    // that gets presented.
                    if (target->keep_samples() && target->color_resolve_image(i) != VK_NULL_HANDLE)
                    {
                        barrier.image = target->color_image(i);
                        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                        frame.vk->vkCmdPipelineBarrier(
                            cmd,
                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                            0,
                            0,
                            nullptr,
                            0,
                            nullptr,
                            1,
                            &barrier);
                    }
                }

                // Depth retires to its own final layout when it will be consumed
                // (offscreen: SHADER_READ_ONLY, which is what makes `target.depth`
                // sampleable). The swapchain's depth stays put â€” no barrier.
                const VkImageAspectFlags depth_aspect = aspect_mask_for(target->depth_format());
                const VkImageLayout depth_layout = has_stencil(target->depth_format())
                                                       ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                                                       : VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
                if (target->depth_image() != VK_NULL_HANDLE && target->depth_final_layout() != depth_layout)
                {
                    // Same as colour: the resolved single-sample depth is what gets
                    // sampled, so it is the one that must reach the final layout.
                    VkImage final_depth = target->depth_resolve_image() != VK_NULL_HANDLE
                                              ? target->depth_resolve_image()
                                              : target->depth_image();
                    VkImageMemoryBarrier depthBarrier{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        .pNext = nullptr,
                        .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                        .oldLayout = depth_layout,
                        .newLayout = target->depth_final_layout(),
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = final_depth,
                        .subresourceRange = {
                            .aspectMask = depth_aspect,
                            .baseMipLevel = depth_sr.base_mip,
                            .levelCount = depth_sr.mip_count,
                            .baseArrayLayer = depth_sr.base_layer,
                            .layerCount = depth_sr.layer_count}};

                    frame.vk->vkCmdPipelineBarrier(
                        cmd,
                        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                        0,
                        0,
                        nullptr,
                        0,
                        nullptr,
                        1,
                        &depthBarrier);

                    // The multisampled depth follows the multisampled colour, for
                    // the reason the colour comment gives. Symmetric on purpose:
                    // "the colour samples are readable and the depth samples are
                    // not" would be a second rule to remember, and it would show up
                    // as a validation error rather than as a message.
                    if (target->keep_samples() && target->depth_resolve_image() != VK_NULL_HANDLE)
                    {
                        depthBarrier.image = target->depth_image();
                        depthBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                        frame.vk->vkCmdPipelineBarrier(
                            cmd,
                            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                            0,
                            0,
                            nullptr,
                            0,
                            nullptr,
                            1,
                            &depthBarrier);
                    }
                }

                // Runs at execute() time, inside a real submit â€” so the target learns
                // its images have left UNDEFINED exactly when that becomes true, and a
                // recorded-but-never-submitted command buffer marks nothing.
                target->on_rendering_recorded();
                // …then bring anything this pass did NOT write up to the same
                // final layout, so the promise "the result ends in this layout"
                // covers the whole image and not just the drawn part. Records
                // nothing when the pass wrote the image whole, which is the
                // usual case. Must follow on_rendering_recorded: that is what
                // makes the per-subresource state true.
                target->record_even_out(*frame.vk, cmd);
            });
        in_rendering_ = false;
        return *this;
    }

    // Explicit override for split-screen and similar. The no-argument version is
    // gone: begin_rendering already covers the whole-target case.
    CommandBuffer& set_viewport(float x, float y, float width, float height)
    {
        commands_.push_back(
            [x, y, width, height](VkCommandBuffer cmd, const FrameContext& frame)
            {
                VkViewport viewport{
                    .x = x, .y = y, .width = width, .height = height, .minDepth = 0.0f, .maxDepth = 1.0f};
                frame.vk->vkCmdSetViewport(cmd, 0, 1, &viewport);
            });
        return *this;
    }

    CommandBuffer& set_scissor(std::int32_t x, std::int32_t y, std::uint32_t width, std::uint32_t height)
    {
        commands_.push_back(
            [x, y, width, height](VkCommandBuffer cmd, const FrameContext& frame)
            {
                VkRect2D scissor{.offset = {x, y}, .extent = {width, height}};
                frame.vk->vkCmdSetScissor(cmd, 0, 1, &scissor);
            });
        return *this;
    }

    CommandBuffer& bind_pipeline(std::shared_ptr<Pipeline> pipeline)
    {
        // Remembered as record-time state, not only pushed as a replay lambda: the
        // tracker needs the bound pipeline to ask its shaders what they write, and
        // before 0.19 a draw had no way to find out which pipeline it belonged to.
        // Keyed on the bind point exactly as bind_descriptor_set already is, so a
        // compute and a graphics pipeline can be bound at once.
        //
        // No lifetime problem: the replay lambda captures the same shared_ptr, and a
        // CommandBuffer is not owned by the Context, so this is not the "a deferred
        // lambda holds nothing that indirectly holds the Context" case.
        if (pipeline)
        {
            if (pipeline->bind_point() == VK_PIPELINE_BIND_POINT_COMPUTE)
            {
                bound_compute_pipeline_ = pipeline;
                bound_last_pipeline_ = pipeline;
            }
            else
            {
                bound_graphics_pipeline_ = pipeline;
                bound_last_pipeline_ = pipeline;
            }
        }
        commands_.push_back([pipeline](VkCommandBuffer cmd, const FrameContext& frame)
                            { frame.vk->vkCmdBindPipeline(cmd, pipeline->bind_point(), pipeline->get()); });
        return *this;
    }

    // binding= selects which of the pipeline's vertex bindings this buffer
    // feeds: 0 is vertex_format (per vertex), 1 is instance_format (per
    // instance). A kwarg on the existing verb rather than a second method —
    // binding one buffer and binding the other are the same operation.
    CommandBuffer& bind_vertex_buffer(std::shared_ptr<Buffer> buffer, std::uint32_t binding = 0)
    {
        // The read truly happens at draw, but a barrier placed before the bind
        // is still before the draw — sound, and simpler than deferring it.
        track_use_(buffer, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT, false);
        record_buffer_use_(buffer);
        commands_.push_back(
            [buffer, binding](VkCommandBuffer cmd, const FrameContext& frame)
            {
                VkBuffer vertexBuffers[] = {buffer->get()};
                VkDeviceSize offsets[] = {0};
                frame.vk->vkCmdBindVertexBuffers(cmd, binding, 1, vertexBuffers, offsets);
            });
        return *this;
    }

    CommandBuffer& bind_index_buffer(std::shared_ptr<Buffer> buffer)
    {
        track_use_(buffer, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_ACCESS_INDEX_READ_BIT, false);
        record_buffer_use_(buffer);
        commands_.push_back(
            [buffer](VkCommandBuffer cmd, const FrameContext& frame)
            {
                // Derived from the buffer rather than hardcoded to UINT32: create_buffer
                // accepts UINT16 indices, which used to be read back at half count.
                frame.vk->vkCmdBindIndexBuffer(cmd, buffer->get(), 0, buffer->index_type());
            });
        return *this;
    }

    // instances= is a kwarg on both draw verbs rather than a third verb: the
    // instance count is one argument of a draw, and draw_indexed_instanced was a
    // second name for a call that already existed.
    CommandBuffer& draw(uint32_t vertexCount, uint32_t instances = 1)
    {
        track_draw_();
        commands_.push_back([vertexCount, instances](VkCommandBuffer cmd, const FrameContext& frame)
                            { frame.vk->vkCmdDraw(cmd, vertexCount, instances, 0, 0); });
        return *this;
    }

    CommandBuffer& draw_indexed(
        uint32_t indexCount,
        uint32_t firstIndex = 0,
        int32_t vertexOffset = 0,
        uint32_t instances = 1)
    {
        track_draw_();
        commands_.push_back(
            [indexCount, firstIndex, vertexOffset, instances](VkCommandBuffer cmd, const FrameContext& frame)
            { frame.vk->vkCmdDrawIndexed(cmd, indexCount, instances, firstIndex, vertexOffset, 0); });
        return *this;
    }

    CommandBuffer& dispatch(uint32_t groupCountX, uint32_t groupCountY = 1, uint32_t groupCountZ = 1)
    {
        track_dispatch_();
        commands_.push_back([groupCountX, groupCountY, groupCountZ](VkCommandBuffer cmd, const FrameContext& frame)
                            { frame.vk->vkCmdDispatch(cmd, groupCountX, groupCountY, groupCountZ); });
        return *this;
    }

    // ── Indirect draw and dispatch (0.19) ─────────────────────────────────────
    //
    // The arguments come out of a buffer the GPU can write, so a compute pass
    // decides what gets drawn and the CPU never learns the answer. That is the
    // whole feature: culling, LOD selection and particle compaction stop needing a
    // readback between the pass that decides and the draw that obeys.
    //
    // Three verbs rather than a buffer= kwarg on draw/draw_indexed/dispatch,
    // because such a kwarg would invalidate vertex_count and instances in the same
    // signature — the shape 0.15 rejected for the cross-Context transfer. They
    // return expected because they have real preconditions, unlike draw().
    //
    // bazalt declares no struct type for the arguments. The layout is
    // VkDrawIndirectCommand and numpy already writes it:
    //
    //     np.array([[vertex_count, instance_count, first_vertex, first_instance]],
    //              dtype=np.uint32)
    //
    // A dtype that exists only to be converted fails the scope test's second
    // question, and a std430 struct in GLSL is byte-identical to the above.
    // count_buffer moves the last CPU-side number onto the GPU: with one, `count`
    // becomes the MAXIMUM and the 4 bytes at count_offset say how many of those
    // commands to issue. A compute pass then decides the number of draws, not only
    // their contents.
    //
    // A kwarg on the verb rather than a fourth verb: where the count comes from is
    // one argument of a draw that already exists, and a draw_indirect_count name
    // would need a copy of every future argument to draw_indirect — the reasoning
    // that made draw_indexed_instanced disappear in 0.17.
    std::expected<void, Error> draw_indirect(
        std::shared_ptr<Buffer> buffer,
        VkDeviceSize offset = 0,
        std::uint32_t count = 1,
        std::shared_ptr<Buffer> count_buffer = nullptr,
        VkDeviceSize count_offset = 0,
        std::uint32_t stride = 0)
    {
        // 16 bytes: vertexCount, instanceCount, firstVertex, firstInstance.
        // stride=0 means "packed", which is the array a compute shader usually
        // writes. A larger stride lets the arguments be INTERLEAVED with per-draw
        // data — a material index, a bounding sphere — so one buffer carries both
        // instead of two buffers that have to stay index-aligned with each other.
        const VkDeviceSize draw_stride = stride ? stride : sizeof(VkDrawIndirectCommand);
        if (auto e =
                check_indirect_(buffer, offset, count, draw_stride, sizeof(VkDrawIndirectCommand), "draw_indirect");
            !e)
        {
            return std::unexpected(e.error());
        }
        if (auto e = check_count_buffer_(count_buffer, count_offset, "draw_indirect"); !e)
        {
            return std::unexpected(e.error());
        }
        track_indirect_(buffer);
        if (count_buffer)
        {
            track_indirect_(count_buffer);
        }
        track_draw_();
        commands_.push_back(
            [buffer = std::move(buffer),
             offset,
             count,
             count_buffer = std::move(count_buffer),
             count_offset,
             draw_stride](VkCommandBuffer cmd, const FrameContext& frame)
            {
                if (count_buffer)
                {
                    frame.vk->vkCmdDrawIndirectCount(
                        cmd,
                        buffer->get(),
                        offset,
                        count_buffer->get(),
                        count_offset,
                        count,
                        static_cast<std::uint32_t>(draw_stride));
                    return;
                }
                frame.vk->vkCmdDrawIndirect(cmd, buffer->get(), offset, count, static_cast<std::uint32_t>(draw_stride));
            });
        return {};
    }

    std::expected<void, Error> draw_indexed_indirect(
        std::shared_ptr<Buffer> buffer,
        VkDeviceSize offset = 0,
        std::uint32_t count = 1,
        std::shared_ptr<Buffer> count_buffer = nullptr,
        VkDeviceSize count_offset = 0,
        std::uint32_t stride = 0)
    {
        // 20 bytes, and note vertexOffset is SIGNED: indexCount, instanceCount,
        // firstIndex, vertexOffset (int32), firstInstance.
        const VkDeviceSize draw_stride = stride ? stride : sizeof(VkDrawIndexedIndirectCommand);
        if (auto e = check_indirect_(
                buffer, offset, count, draw_stride, sizeof(VkDrawIndexedIndirectCommand), "draw_indexed_indirect");
            !e)
        {
            return std::unexpected(e.error());
        }
        if (auto e = check_count_buffer_(count_buffer, count_offset, "draw_indexed_indirect"); !e)
        {
            return std::unexpected(e.error());
        }
        track_indirect_(buffer);
        if (count_buffer)
        {
            track_indirect_(count_buffer);
        }
        track_draw_();
        commands_.push_back(
            [buffer = std::move(buffer),
             offset,
             count,
             count_buffer = std::move(count_buffer),
             count_offset,
             draw_stride](VkCommandBuffer cmd, const FrameContext& frame)
            {
                if (count_buffer)
                {
                    frame.vk->vkCmdDrawIndexedIndirectCount(
                        cmd,
                        buffer->get(),
                        offset,
                        count_buffer->get(),
                        count_offset,
                        count,
                        static_cast<std::uint32_t>(draw_stride));
                    return;
                }
                frame.vk->vkCmdDrawIndexedIndirect(
                    cmd, buffer->get(), offset, count, static_cast<std::uint32_t>(draw_stride));
            });
        return {};
    }

    // No count: vkCmdDispatchIndirect takes exactly one VkDispatchIndirectCommand
    // (12 bytes, x/y/z group counts), so there is no multi-dispatch to gate.
    // Deliberately NOT refused inside a rendering scope, because plain dispatch()
    // is not either — one rule for both.
    std::expected<void, Error> dispatch_indirect(std::shared_ptr<Buffer> buffer, VkDeviceSize offset = 0)
    {
        // No stride argument here, and not an omission: dispatch_indirect issues
        // exactly one command, so there is nothing for a stride to step over.
        if (auto e = check_indirect_(
                buffer,
                offset,
                1,
                sizeof(VkDispatchIndirectCommand),
                sizeof(VkDispatchIndirectCommand),
                "dispatch_indirect");
            !e)
        {
            return std::unexpected(e.error());
        }
        track_indirect_(buffer);
        track_dispatch_();
        commands_.push_back([buffer = std::move(buffer), offset](VkCommandBuffer cmd, const FrameContext& frame)
                            { frame.vk->vkCmdDispatchIndirect(cmd, buffer->get(), offset); });
        return {};
    }

    // Manual-mode barrier (also legal, if redundant, in auto mode). Refused
    // inside a rendering scope: vkCmdPipelineBarrier is invalid there, and in
    // manual mode nothing is hoisted by magic — that would be a second,
    // implicit way of doing the explicit thing.
    std::expected<void, Error> barrier(std::shared_ptr<Buffer> buffer, Access src, Access dst)
    {
        if (!buffer)
        {
            return std::unexpected(err_resource("barrier: buffer is null"));
        }
        if (in_rendering_)
        {
            return std::unexpected(err_state(
                "cmd.barrier() is not allowed inside a rendering scope. "
                "Record it before begin_rendering"));
        }
        const StageAccess s = to_vk(src, context_->all_shader_stages());
        const StageAccess d = to_vk(dst, context_->all_shader_stages());
        Buffer* buf = buffer.get();
        record_barrier_(std::move(buffer), {s.stages, d.stages, s.access, d.access});
        // Keep the auto-tracker in sync, for the reason the image overload below
        // does it: the caller just expressed this dependency, so the next
        // automatic use of the buffer must not emit the first-use floor on top of
        // it. No-op in manual mode (the tracker is never consulted).
        if (auto_barriers_)
        {
            tracker_.note_buffer_access(buf, d.stages, d.access);
        }
        return {};
    }

    // The image counterpart: transition an image between shader accesses by
    // hand, across every mip and layer. The one cross-submit case the automatic
    // tracker can't reach — a compute shader bakes a storage image (GENERAL) in
    // one submit and later frames sample it (SHADER_READ_ONLY) — becomes
    // `cmd.barrier(image, Access.SHADER_WRITE, Access.SHADER_READ)` once, after
    // the dispatch, so the asset is generated once instead of every frame. The
    // layout is inferred from the access (WRITE->GENERAL, READ->SHADER_READ_ONLY);
    // only those two shader accesses name an image layout. In auto mode this also
    // updates the tracker, so mixing it with automatic uses of the same image in
    // one recording is safe — no stale-oldLayout double transition.
    std::expected<void, Error> barrier(std::shared_ptr<Image> image, Access src, Access dst)
    {
        if (!image)
        {
            return std::unexpected(err_resource("barrier: image is null"));
        }
        if (in_rendering_)
        {
            return std::unexpected(err_state(
                "cmd.barrier() is not allowed inside a rendering scope. "
                "Record it before begin_rendering"));
        }
        const auto old_layout = image_layout_for(src);
        const auto new_layout = image_layout_for(dst);
        if (!old_layout || !new_layout)
        {
            return std::unexpected(err_resource(
                "cmd.barrier(image, ...) takes Access.SHADER_WRITE or Access.SHADER_READ. "
                "The other accesses apply to buffers only."));
        }
        const StageAccess s = to_vk(src, context_->all_shader_stages());
        const StageAccess d = to_vk(dst, context_->all_shader_stages());
        Image* img = image.get();
        record_image_barrier_(std::move(image), {*old_layout, *new_layout, s.stages, d.stages, s.access, d.access});
        // Keep the auto-tracker in sync: a later automatic use of this image in
        // the same recording must see the post-barrier layout, not re-transition
        // from a stale one. No-op in manual mode (the tracker is never consulted).
        if (auto_barriers_)
        {
            tracker_.note_image_layout(img, *new_layout, d.stages, d.access);
        }
        return {};
    }

    // Fills mip levels 1..N-1 of a mipped image by blitting mip 0 down the chain
    // (every array layer / cube face at once), leaving every level sampleable in
    // SHADER_READ_ONLY. The pair to create_image(..., mip_levels=N): write mip 0
    // (upload, compute, or a render pass), then generate the rest here.
    //
    // `src` names mip 0's CURRENT layout via the same access vocabulary as
    // cmd.barrier: SHADER_READ (SHADER_READ_ONLY — an uploaded or already-baked
    // image, the default) or SHADER_WRITE (GENERAL — mip 0 fresh from a compute
    // imageStore). Its scope doubles as the barrier waiting on that producer.
    // Refused inside a rendering scope (blits and barriers are illegal there).
    std::expected<void, Error> generate_mipmaps(std::shared_ptr<Image> image, Access src = Access::SHADER_READ)
    {
        if (!image)
        {
            return std::unexpected(err_resource("generate_mipmaps: image is null"));
        }
        if (in_rendering_)
        {
            return std::unexpected(err_state(
                "cmd.generate_mipmaps() is not allowed inside a rendering scope. "
                "Record it before begin_rendering"));
        }
        if (image->mip_levels() <= 1)
        {
            return std::unexpected(err_resource(
                "generate_mipmaps: image has a single mip level. Create it with "
                "mip_levels>1 (empty) or mipmaps=True (from pixels/files)"));
        }
        if (!Image::can_generate_mips(*context_, image->format()))
        {
            return std::unexpected(err_unsupported(
                "generate_mipmaps: this format cannot be blitted and linearly "
                "filtered on this device, so a mip chain can't be generated"));
        }
        const auto src_layout = image_layout_for(src);
        if (!src_layout)
        {
            return std::unexpected(err_resource(
                "generate_mipmaps: src must be Access.SHADER_READ (mip 0 in "
                "SHADER_READ_ONLY) or Access.SHADER_WRITE (mip 0 in GENERAL)"));
        }
        const StageAccess s = to_vk(src, context_->all_shader_stages());
        Image* img = image.get();
        commands_.push_back(
            [image = std::move(image), layout = *src_layout, s](VkCommandBuffer cmd, const FrameContext& frame)
            { image->record_generate_mipmaps(cmd, layout, s.stages, s.access); });
        // The image now rests in SHADER_READ_ONLY across every level; keep the
        // tracker in sync so a later automatic sample emits no extra transition.
        if (auto_barriers_)
        {
            tracker_.note_image_layout(
                img,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                context_->all_shader_stages(),
                VK_ACCESS_SHADER_READ_BIT);
        }
        return {};
    }

    // Copy one image into another of the same size and format. The history
    // buffer every temporal effect needs: keep last frame's result to blend
    // against this one (motion blur, TAA, a feedback trail), or ping-pong two
    // storage images across dispatches.
    //
    // Both images are left in SHADER_READ_ONLY, because reading the copy is the
    // only reason to make one. `src` names the source's CURRENT layout in
    // cmd.barrier's vocabulary, exactly as generate_mipmaps does: SHADER_READ for
    // an image that is sampled (the default) or SHADER_WRITE for one a compute
    // dispatch just wrote. The destination is discarded, since the copy
    // overwrites all of it.
    //
    // Refused inside a rendering scope, like every other transfer verb.
    std::expected<void, Error> copy_image(
        std::shared_ptr<Image> src,
        std::shared_ptr<Image> dst,
        Access src_access = Access::SHADER_READ)
    {
        if (!src || !dst)
        {
            return std::unexpected(err_resource("copy_image: image is null"));
        }
        if (in_rendering_)
        {
            return std::unexpected(err_state(
                "cmd.copy_image() is not allowed inside a rendering scope. "
                "Record it before begin_rendering"));
        }
        if (src->width() != dst->width() || src->height() != dst->height() || src->depth() != dst->depth() ||
            src->format() != dst->format() || src->array_layers() != dst->array_layers())
        {
            return std::unexpected(err_resource(
                std::format(
                    "copy_image: source and destination must match in size, format and layer "
                    "count. Got {}x{}x{} {} ({} layers) into {}x{}x{} {} ({} layers). A resize "
                    "or a format change is a render pass, not a copy.",
                    src->width(),
                    src->height(),
                    src->depth(),
                    format_name(src->format()),
                    src->array_layers(),
                    dst->width(),
                    dst->height(),
                    dst->depth(),
                    format_name(dst->format()),
                    dst->array_layers())));
        }
        if (src->samples() != 1 || dst->samples() != 1)
        {
            return std::unexpected(err_resource(
                "copy_image: a multisampled image cannot be copied. Render into it and "
                "read the resolved attachment"));
        }
        const auto src_layout = image_layout_for(src_access);
        if (!src_layout)
        {
            return std::unexpected(err_resource(
                "copy_image: src must be Access.SHADER_READ (the source is sampled, "
                "SHADER_READ_ONLY) or Access.SHADER_WRITE (a compute shader just wrote "
                "it, GENERAL)"));
        }
        Image* src_ptr = src.get();
        Image* dst_ptr = dst.get();
        commands_.push_back([src = std::move(src), dst = std::move(dst), layout = *src_layout](
                                VkCommandBuffer cmd, const FrameContext& frame)
                            { record_image_copy(*frame.vk, cmd, *src, *dst, layout); });
        // BOTH ends, not just the destination: the copy leaves the source
        // sampleable too, and an Image that still believed it was in GENERAL
        // would hand a stale oldLayout to the next read() — a validation error
        // with no obvious author.
        src_ptr->mark_has_contents(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        dst_ptr->mark_has_contents(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        if (auto_barriers_)
        {
            // Both ends leave the copy sampleable, and the tracker has to learn
            // it or the next automatic use transitions from a stale layout — the
            // same rule the manual image barrier follows.
            tracker_.note_image_layout(
                src_ptr,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                context_->all_shader_stages(),
                VK_ACCESS_SHADER_READ_BIT);
            tracker_.note_image_layout(
                dst_ptr,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                context_->all_shader_stages(),
                VK_ACCESS_SHADER_READ_BIT);
        }
        return {};
    }

    // A copy that RESIZES: the two images need not share an extent, and `filter`
    // says how the pixels are sampled on the way.
    //
    // copy_image demands identical size and format; generate_mipmaps scales but
    // only inside one image. Downsampling for bloom, upscaling a compute result
    // and making a thumbnail all sat in that gap, and each one was a full
    // graphics pass with a fullscreen shader to do what the transfer queue does
    // in one command.
    //
    // Same `src_access` vocabulary as copy_image and generate_mipmaps: the
    // tracker treats an image's layout at the start of a replay as UNDEFINED, so
    // the caller names where the source actually is.
    std::expected<void, Error> blit_image(
        std::shared_ptr<Image> src,
        std::shared_ptr<Image> dst,
        Access src_access = Access::SHADER_READ,
        VkFilter filter = VK_FILTER_LINEAR)
    {
        if (!src || !dst)
        {
            return std::unexpected(err_resource("blit_image: image is null"));
        }
        if (src.get() == dst.get())
        {
            return std::unexpected(err_resource(
                "blit_image: source and destination are the same image. A blit within one "
                "image is what generate_mipmaps does"));
        }
        if (in_rendering_)
        {
            return std::unexpected(err_state(
                "cmd.blit_image() is not allowed inside a rendering scope. "
                "Record it before begin_rendering"));
        }
        if (src->samples() != 1 || dst->samples() != 1)
        {
            return std::unexpected(err_resource(
                "blit_image: a multisampled image cannot be blitted. Render into it and "
                "blit the resolved attachment"));
        }
        // Vulkan requires both ends of a blit to be the same image type, so a
        // volume scales into a volume — resampling a volume into a 2D image is a
        // shader's job (sample the slice you want).
        if (src->is_3d() != dst->is_3d())
        {
            return std::unexpected(err_resource(
                "blit_image: a 3D image can only be blitted into another 3D image. To "
                "flatten a volume, sample it in a shader."));
        }
        // A blit filters, and filtering is a format capability rather than a
        // given. Checking here names the format; letting it through produces a
        // validation error about VkFormatFeatureFlags instead.
        if (!Image::can_blit(*context_, src->format(), dst->format()))
        {
            return std::unexpected(err_unsupported(
                std::format(
                    "blit_image: this GPU cannot blit {} into {}. Both formats need "
                    "BLIT_SRC/BLIT_DST support, and a linear filter needs the source to be "
                    "filterable — use copy_image for a same-size copy, or a render pass.",
                    format_name(src->format()),
                    format_name(dst->format()))));
        }
        const auto src_layout = image_layout_for(src_access);
        if (!src_layout)
        {
            return std::unexpected(err_resource(
                "blit_image: src must be Access.SHADER_READ (the source is sampled, "
                "SHADER_READ_ONLY) or Access.SHADER_WRITE (a compute shader just wrote "
                "it, GENERAL)"));
        }
        Image* src_ptr = src.get();
        Image* dst_ptr = dst.get();
        commands_.push_back([src = std::move(src), dst = std::move(dst), layout = *src_layout, filter](
                                VkCommandBuffer cmd, const FrameContext& frame)
                            { record_image_blit(*frame.vk, cmd, *src, *dst, layout, filter); });
        // Both ends, for the reason copy_image spells out above.
        src_ptr->mark_has_contents(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        dst_ptr->mark_has_contents(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        if (auto_barriers_)
        {
            tracker_.note_image_layout(
                src_ptr,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                context_->all_shader_stages(),
                VK_ACCESS_SHADER_READ_BIT);
            tracker_.note_image_layout(
                dst_ptr,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                context_->all_shader_stages(),
                VK_ACCESS_SHADER_READ_BIT);
        }
        return {};
    }

    // Copy bytes from one buffer into another, GPU-side.
    //
    // There was no way to move buffer contents without a round trip through the
    // host or a compute shader written to do nothing but assign. A compute
    // ping-pong and "keep last frame's values" are both this one command.
    std::expected<void, Error> copy_buffer(
        std::shared_ptr<Buffer> src,
        std::shared_ptr<Buffer> dst,
        VkDeviceSize src_offset = 0,
        VkDeviceSize dst_offset = 0,
        // 0 means "the rest of the source", which is the whole buffer by
        // default. VK_WHOLE_SIZE is not legal in a copy region, so the real
        // length is computed below rather than passed through.
        VkDeviceSize size = 0)
    {
        if (!src || !dst)
        {
            return std::unexpected(err_resource("copy_buffer: buffer is null"));
        }
        if (in_rendering_)
        {
            return std::unexpected(err_state(
                "cmd.copy_buffer() is not allowed inside a rendering scope. "
                "Record it before begin_rendering"));
        }
        const VkDeviceSize length = size != 0 ? size : (src->size() > src_offset ? src->size() - src_offset : 0);
        if (length == 0)
        {
            return std::unexpected(err_resource("copy_buffer: nothing to copy (size is 0)"));
        }
        if (!fits_within(src_offset, length, src->size()) || !fits_within(dst_offset, length, dst->size()))
        {
            return std::unexpected(err_resource(
                std::format(
                    "copy_buffer: the region does not fit. It is {} bytes at offset {} of a {}-byte source "
                    "into offset {} of a {}-byte destination",
                    length,
                    src_offset,
                    src->size(),
                    dst_offset,
                    dst->size())));
        }

        track_use_(src, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT, false);
        track_use_(dst, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, true);
        record_buffer_use_(src);
        record_buffer_use_(dst);
        commands_.push_back(
            [src = std::move(src), dst = std::move(dst), src_offset, dst_offset, length](
                VkCommandBuffer cmd, const FrameContext& frame)
            {
                VkBufferCopy region{.srcOffset = src_offset, .dstOffset = dst_offset, .size = length};
                // Resolved at execute, never captured: a DynamicBuffer has one
                // handle per frame in flight.
                frame.vk->vkCmdCopyBuffer(cmd, src->get(), dst->get(), 1, &region);
            });
        return {};
    }

    // Fill a buffer with a repeated 32-bit value, GPU-side. Zeroing is the
    // reason it exists: a counter an atomic increments, or an accumulation
    // buffer, has to start each frame at a known value, and the only way to say
    // that was a dispatch whose whole body was an assignment.
    //
    // 32-bit because vkCmdFillBuffer is: the offset and the size must both be
    // multiples of 4, and the value is one dword repeated.
    std::expected<void, Error> fill_buffer(
        std::shared_ptr<Buffer> buffer,
        std::uint32_t value = 0,
        VkDeviceSize offset = 0,
        VkDeviceSize size = 0)
    {
        if (!buffer)
        {
            return std::unexpected(err_resource("fill_buffer: buffer is null"));
        }
        if (in_rendering_)
        {
            return std::unexpected(err_state(
                "cmd.fill_buffer() is not allowed inside a rendering scope. "
                "Record it before begin_rendering"));
        }
        if (offset % 4 != 0 || (size != 0 && size % 4 != 0))
        {
            return std::unexpected(err_resource(
                std::format(
                    "fill_buffer: offset and size must be multiples of 4 (the value is one "
                    "32-bit word repeated). Got offset={}, size={}",
                    offset,
                    size)));
        }
        const VkDeviceSize length = size != 0 ? size : (buffer->size() > offset ? buffer->size() - offset : 0);
        if (length == 0 || !fits_within(offset, length, buffer->size()))
        {
            return std::unexpected(err_resource(
                std::format(
                    "fill_buffer: the region does not fit. It is {} bytes at offset {} of a {}-byte buffer",
                    length,
                    offset,
                    buffer->size())));
        }

        track_use_(buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, true);
        record_buffer_use_(buffer);
        commands_.push_back(
            [buffer = std::move(buffer), value, offset, length](VkCommandBuffer cmd, const FrameContext& frame)
            { frame.vk->vkCmdFillBuffer(cmd, buffer->get(), offset, length, value); });
        return {};
    }

    // Fill an image with one colour, with no pipeline and no pass. Resetting an
    // accumulation or history buffer, or clearing a storage image a compute
    // shader only writes part of. A depth image is refused: clearing depth is
    // what a rendering pass does, and it needs the depth clear value.
    std::expected<void, Error> clear_image(std::shared_ptr<Image> image, std::array<float, 4> color)
    {
        if (!image)
        {
            return std::unexpected(err_resource("clear_image: image is null"));
        }
        if (in_rendering_)
        {
            return std::unexpected(err_state(
                "cmd.clear_image() is not allowed inside a rendering scope. "
                "Record it before begin_rendering"));
        }
        if (format_info(image->format()).depth)
        {
            return std::unexpected(err_resource(
                "clear_image: a depth image is cleared by the pass that renders into it "
                "(cmd.rendering(target, clear_depth=...))"));
        }
        Image* img = image.get();
        commands_.push_back([image = std::move(image), color](VkCommandBuffer cmd, const FrameContext& frame)
                            { record_image_clear(*frame.vk, cmd, *image, color); });
        img->mark_has_contents(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        if (auto_barriers_)
        {
            tracker_.note_image_layout(
                img,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                context_->all_shader_stages(),
                VK_ACCESS_SHADER_READ_BIT);
        }
        return {};
    }

    // ── GPU timers ──────────────────────────────────────────────────────────
    //
    // A GPU timer is a pair of query slots — exactly a Vulkan timestamp query.
    // The Python-facing handle (struct Timer, in bindings/Common.hpp) owns one pair:
    // cmd.timer() records the opening timestamp and hands back the handle, which
    // is stopped explicitly (t.stop()) or by a `with`, and read back off itself
    // (t.ms). The handle IS the identity — no name, no key — so multiple, nested
    // and overlapping timers all just work.
    //
    // Unlike renderer.gpu_time_ms this needs no window and no frame loop:
    // the headless submit blocks, so the readback is ready as soon as
    // ctx.submit() returns (profiling a dispatch is the use case).
    //
    // Self-gating: the query pool is created only when a timer is actually used,
    // so an app that never calls timer() pays nothing, no Context flag required.
    // Best-effort: a device without timestamp support reports None, never errors.

    // Records the opening timestamp and returns the timer's index (its two query
    // slots are 2*index / 2*index+1). Paired with stop_timer.
    std::size_t start_timer()
    {
        const std::size_t index = timer_count_++;
        record_timer_write_(static_cast<std::uint32_t>(2 * index), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
        return index;
    }

    void stop_timer(std::size_t index)
    {
        record_timer_write_(static_cast<std::uint32_t>(2 * index + 1), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    }

    // Which recording a timer belongs to. begin() bumps this, so a handle read
    // after the command buffer was re-recorded reports None (its slots now hold
    // a different timer's data) instead of a misleading number.
    std::uint64_t recording_generation() const
    {
        return recording_generation_;
    }

    struct TimerReading
    {
        QueryStatus status = QueryStatus::NotReady;
        double ms = 0.0;
    };

    struct OcclusionReading
    {
        QueryStatus status = QueryStatus::NotReady;
        std::uint64_t samples = 0;
    };

    // The measured time of one timer in milliseconds, with the reason attached
    // when there is none.
    TimerReading read_timer(std::size_t index, std::uint64_t generation) const
    {
        // First, because a stale handle's index may be out of range for the
        // recording that replaced it — that is still the stale handle's fault.
        if (generation != recording_generation_)
        {
            return {QueryStatus::Superseded};
        }
        // timer_supported_ is only filled in at the first execute, so an unset
        // one means "nobody has asked the device yet", which is NotReady rather
        // than a verdict.
        if (timer_supported_.has_value() && !*timer_supported_)
        {
            return {QueryStatus::Unsupported};
        }
        if (timer_pool_ == VK_NULL_HANDLE || index >= timer_count_)
        {
            return {QueryStatus::NotReady};
        }
        std::uint64_t ts[2] = {0, 0};
        if (context_->vk().vkGetQueryPoolResults(
                context_->device(),
                timer_pool_,
                static_cast<std::uint32_t>(2 * index),
                2,
                sizeof(ts),
                ts,
                sizeof(std::uint64_t),
                VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
        {
            return {QueryStatus::NotReady}; // VK_NOT_READY: submit not finished
        }
        const std::uint64_t mask = timer_valid_bits_ >= 64 ? ~std::uint64_t{0}
                                                           : ((std::uint64_t{1} << timer_valid_bits_) - 1);
        const std::uint64_t delta = (ts[1] - ts[0]) & mask;
        return {QueryStatus::Ok, static_cast<double>(delta) * static_cast<double>(timer_period_) / 1.0e6};
    }

    // ── Debug labels ────────────────────────────────────────────────────────
    //
    // A named scope in a capture. bazalt has named its OBJECTS since 0.8, which
    // answers "which image is that", but a RenderDoc capture was still a flat
    // list of draws with nothing saying where the shadow pass ended and the
    // composite began.
    //
    // Silent no-op without VK_EXT_debug_utils, exactly like set_debug_name, and
    // for the same reason: vk-bootstrap only requests the extension when a debug
    // callback is set. So a release run pays nothing and simply shows no labels.
    //
    // The entry points stay on volk's globals rather than moving to ctx.vk(),
    // which is the documented rule for debug utils: it is an INSTANCE extension,
    // so vkGetInstanceProcAddr is the sanctioned route and vkGetDeviceProcAddr
    // may legally return null. They are loader trampolines dispatching on the
    // VkCommandBuffer, so one pointer is right for every Context.
    CommandBuffer& begin_label(const std::string& name)
    {
        commands_.push_back(
            [name](VkCommandBuffer cmd, const FrameContext&)
            {
                if (vkCmdBeginDebugUtilsLabelEXT == nullptr)
                {
                    return;
                }
                VkDebugUtilsLabelEXT label{
                    .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
                    .pNext = nullptr,
                    .pLabelName = name.c_str(),
                    .color = {0.0f, 0.0f, 0.0f, 0.0f}};
                vkCmdBeginDebugUtilsLabelEXT(cmd, &label);
            });
        ++open_labels_;
        return *this;
    }

    CommandBuffer& end_label()
    {
        // Unbalanced ends are dropped rather than recorded: ending a label that
        // was never begun is undefined behaviour in Vulkan, and the `with` form
        // that the binding exposes cannot produce one. This guards the explicit
        // verbs only.
        if (open_labels_ == 0)
        {
            return *this;
        }
        --open_labels_;
        commands_.push_back(
            [](VkCommandBuffer cmd, const FrameContext&)
            {
                if (vkCmdEndDebugUtilsLabelEXT != nullptr)
                {
                    vkCmdEndDebugUtilsLabelEXT(cmd);
                }
            });
        return *this;
    }

    // ── Occlusion queries ───────────────────────────────────────────────────
    //
    // How many fragments of the draws inside the scope passed the depth and
    // stencil tests. The handle IS the identity, exactly as for timers (0.9), so
    // several queries in one recording need no names and no keys.
    //
    // Vulkan requires an occlusion query to begin and end inside the SAME render
    // pass, which is why this refuses outside a rendering scope: the alternative
    // is a validation error at submit naming neither the call nor the reason.
    std::expected<std::size_t, Error> start_occlusion_query()
    {
        if (!in_rendering_)
        {
            return std::unexpected(err_state(
                "cmd.occlusion_query() must be used inside a rendering scope: Vulkan requires an occlusion "
                "query to begin and end within one render pass. Move it inside `with cmd.rendering(target):`."));
        }
        const std::size_t index = occlusion_count_++;
        commands_.push_back(
            [this, index](VkCommandBuffer cmd, const FrameContext& frame)
            {
                if (occlusion_pool_ != VK_NULL_HANDLE)
                {
                    // PRECISE exactly where the device promises it (0.25). Without
                    // occlusionQueryPrecise the spec allows any non-zero value for
                    // "something passed", so the flag is the difference between
                    // `query.samples` being a COUNT and being a yes/no wearing an
                    // integer. That is what a Feature row is for: the capability is
                    // negotiated, and ctx.supports(Feature.PRECISE_OCCLUSION) says
                    // which of the two answers this GPU is giving.
                    const VkQueryControlFlags flags = precise_occlusion_ ? VK_QUERY_CONTROL_PRECISE_BIT
                                                                         : static_cast<VkQueryControlFlags>(0);
                    frame.vk->vkCmdBeginQuery(cmd, occlusion_pool_, static_cast<std::uint32_t>(index), flags);
                }
            });
        return index;
    }

    void stop_occlusion_query(std::size_t index)
    {
        commands_.push_back(
            [this, index](VkCommandBuffer cmd, const FrameContext& frame)
            {
                if (occlusion_pool_ != VK_NULL_HANDLE)
                {
                    frame.vk->vkCmdEndQuery(cmd, occlusion_pool_, static_cast<std::uint32_t>(index));
                }
            });
    }

    // The sample count of one query, with the same reasons attached. There is no
    // Unsupported here: an imprecise occlusion query is core Vulkan behind no
    // feature, so the only answers are the number, a stale handle, and "not
    // yet".
    OcclusionReading read_occlusion_query(std::size_t index, std::uint64_t generation) const
    {
        if (generation != recording_generation_)
        {
            return {QueryStatus::Superseded};
        }
        if (occlusion_pool_ == VK_NULL_HANDLE || index >= occlusion_count_)
        {
            return {QueryStatus::NotReady};
        }
        std::uint64_t samples = 0;
        if (context_->vk().vkGetQueryPoolResults(
                context_->device(),
                occlusion_pool_,
                static_cast<std::uint32_t>(index),
                1,
                sizeof(samples),
                &samples,
                sizeof(std::uint64_t),
                VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
        {
            return {QueryStatus::NotReady}; // VK_NOT_READY: submit not finished
        }
        return {QueryStatus::Ok, samples};
    }

    // The short form, for the pipeline that is already bound (0.25). Uses the
    // LAST pipeline bound whatever its bind point, because push constants belong
    // to a pipeline layout rather than to a bind point, and a recording that
    // pushes for a pipeline it has not bound is already confused.
    std::expected<void, Error> push_constants(uint32_t offset, uint32_t size, const void* data)
    {
        if (!bound_last_pipeline_)
        {
            return std::unexpected(err_state(
                "push_constants(offset, data) needs a pipeline bound first, because that is where "
                "it reads the layout and the stage mask from. Call cmd.bind_pipeline(pipeline) "
                "before it, or name the pipeline: push_constants(pipeline, offset, data)."));
        }
        push_constants(bound_last_pipeline_, offset, size, data);
        return {};
    }

    // No stage argument: the Pipeline already knows which stages its push constant
    // range covers, so passing a mismatched one was a validation error for no gain.
    CommandBuffer& push_constants(std::shared_ptr<Pipeline> pipeline, uint32_t offset, uint32_t size, const void* data)
    {
        std::vector<uint8_t> buffer(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + size);
        commands_.push_back(
            [pipeline, offset, size, buffer](VkCommandBuffer cmd, const FrameContext& frame)
            {
                frame.vk->vkCmdPushConstants(
                    cmd, pipeline->layout(), pipeline->push_constant_stages(), offset, size, buffer.data());
            });
        return *this;
    }

    // The short form: bind the set where it was allocated to go, on the pipeline
    // that is already bound (0.25, ergonomics #3). Both arguments the long form
    // takes are known — the set records its index and bind point, and
    // bind_pipeline records the pipeline — so repeating them is a chance to
    // disagree with the truth, not information.
    //
    // The long form stays for a recording split across functions, where the
    // pipeline was bound somewhere this code cannot see, and for binding a set
    // against a DIFFERENT pipeline with a compatible layout.
    std::expected<void, Error> bind_descriptor_set(std::shared_ptr<DescriptorSet> descSet)
    {
        const bool compute = descSet->bind_point() == VK_PIPELINE_BIND_POINT_COMPUTE;
        std::shared_ptr<Pipeline> pipeline = compute ? bound_compute_pipeline_ : bound_graphics_pipeline_;
        if (!pipeline)
        {
            return std::unexpected(err_state(
                std::format(
                    "bind_descriptor_set(set) needs a {} pipeline bound first, because that is where "
                    "it reads the layout from. Call cmd.bind_pipeline(pipeline) before it, or name "
                    "the pipeline: bind_descriptor_set(set, pipeline, set=N).",
                    compute ? "compute" : "graphics")));
        }
        const std::uint32_t set_index = descSet->set_index();
        bind_descriptor_set(std::move(descSet), std::move(pipeline), set_index);
        return {};
    }

    CommandBuffer& bind_descriptor_set(
        std::shared_ptr<DescriptorSet> descSet,
        std::shared_ptr<Pipeline> pipeline,
        uint32_t setIndex)
    {
        // Remembered so submit paths can walk the images this recording
        // references and wait for their (async) uploads — residency is a
        // per-command-buffer question, not a global one, or a loading screen
        // would serialize behind its own cargo.
        used_sets_.push_back(descSet);
        // Record-time bookkeeping for the tracker: the next dispatch/draw walks
        // the sets bound at its bind point. Rebinding an index replaces it.
        if (pipeline->bind_point() == VK_PIPELINE_BIND_POINT_COMPUTE)
        {
            bound_compute_sets_[setIndex] = descSet;
        }
        else
        {
            bound_graphics_sets_[setIndex] = descSet;
        }
        commands_.push_back(
            [descSet, pipeline, setIndex](VkCommandBuffer cmd, const FrameContext& frame)
            {
                VkDescriptorSet set = descSet->get(frame.frame_index);
                frame.vk->vkCmdBindDescriptorSets(
                    cmd, pipeline->bind_point(), pipeline->layout(), setIndex, 1, &set, 0, nullptr);
            });
        return *this;
    }

    const std::vector<std::shared_ptr<DescriptorSet>>& used_sets() const
    {
        return used_sets_;
    }

    // The buffers this recording binds or copies, for the same reason
    // used_sets exists: a STATIC buffer's fill is a submit of its own since
    // 0.18.0, and the submit path waits on it. Recorded by record_buffer_use_,
    // which is deliberately NOT part of track_use_ — that one returns early
    // with auto_barriers=False, and residency is not a barrier question.
    const std::vector<std::shared_ptr<Buffer>>& used_buffers() const
    {
        return used_buffers_;
    }

    VkCommandBuffer get(std::uint32_t frame_index) const
    {
        return command_buffers_[frame_index];
    }

    // One CommandBuffer owns one VkCommandBuffer per ring slot, so replaying it
    // twice inside one logical frame resets and re-records a buffer the first
    // replay very likely still has in flight. Unreachable before 0.14 (one
    // window meant one replay per frame); now it is the obvious way to try to
    // drive two windows, so it gets a sentence instead of a pending-state VUID
    // — which a build without the validation layers wouldn't print at all.
    std::expected<void, Error> claim_for_frame(std::uint64_t serial)
    {
        if (recorded_serial_ == serial)
        {
            return std::unexpected(err_state(
                "This CommandBuffer was already submitted in the current frame. Each "
                "window needs its own CommandBuffer — one holds a single command "
                "buffer per frame slot, so replaying it twice would overwrite work "
                "still in flight."));
        }
        recorded_serial_ = serial;
        return {};
    }

    void execute(VkCommandBuffer vkCmd, const FrameContext& frame)
    {
        // Timer query pool: created/grown here (the scope count is known once
        // recording is done) and reset before any command runs — timestamps
        // must be reset before they are written, and vkCmdResetQueryPool is
        // illegal inside a render pass, so the top of execute is the one safe
        // spot. The timestamp-write lambdas read timer_pool_ at execute, so a
        // grow that recreates the pool is picked up without re-recording.
        if (timer_count_ > 0)
        {
            ensure_timer_pool_(2 * timer_count_);
            if (timer_pool_ != VK_NULL_HANDLE)
            {
                frame.vk->vkCmdResetQueryPool(vkCmd, timer_pool_, 0, timer_capacity_);
            }
        }
        // Occlusion queries reset in the same place and for the same reason: the
        // reset is illegal inside a render pass, and an occlusion query can only
        // BEGIN inside one, so the top of execute is the only spot that serves
        // both halves.
        if (occlusion_count_ > 0)
        {
            ensure_occlusion_pool_(occlusion_count_);
            if (occlusion_pool_ != VK_NULL_HANDLE)
            {
                frame.vk->vkCmdResetQueryPool(vkCmd, occlusion_pool_, 0, occlusion_capacity_);
            }
        }

        // Replay wrap-around. In-recording barriers order uses within one
        // replay, but the same recording ran last frame and may still be in
        // flight — its trailing reads/writes race with this replay's first
        // write. One conservative memory barrier at the top covers that.
        // Emitted only when the recording writes a tracked buffer at all:
        // read-only recordings race with nothing.
        if (auto_barriers_ && tracked_writes_)
        {
            VkMemoryBarrier barrier{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                // INDIRECT_COMMAND_READ is in here for the same reason
                // VERTEX_ATTRIBUTE_READ is (0.19): the canonical indirect chain has
                // compute writing the draw arguments, so frame N+1's command
                // processor reads exactly what frame N's dispatch is still writing.
                // Missing it is invisible without sync validation.
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_UNIFORM_READ_BIT |
                                 VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT |
                                 VK_ACCESS_INDIRECT_COMMAND_READ_BIT};
            // Not constexpr any more: the shader-stage half of this mask depends on
            // which stages the device enabled (see Context::all_shader_stages).
            const VkPipelineStageFlags stages = context_->all_shader_stages() | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
                                                VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
            frame.vk->vkCmdPipelineBarrier(vkCmd, stages, stages, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        }
        for (auto& cmd_func : commands_)
        {
            cmd_func(vkCmd, frame);
        }
    }

private:
    CommandBuffer(std::shared_ptr<Context> context)
        : context_(context)
    {
    }

    // Records a buffer-barrier lambda. Inside a rendering scope it is hoisted
    // to just before the begin_rendering lambda (vkCmdPipelineBarrier is
    // illegal inside dynamic rendering); deferred recording makes the insert
    // a cheap vector operation on data that only exists at record time.
    void record_barrier_(std::shared_ptr<Buffer> buffer, ResourceTracker::Barrier b)
    {
        hoist_or_push_(
            [buffer = std::move(buffer), b](VkCommandBuffer cmd, const FrameContext& frame)
            {
                VkBufferMemoryBarrier barrier{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = b.src_access,
                    .dstAccessMask = b.dst_access,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    // Resolved at execute time, never captured: a DynamicBuffer
                    // has one handle per frame in flight.
                    .buffer = buffer->get(),
                    .offset = 0,
                    .size = VK_WHOLE_SIZE};
                frame.vk->vkCmdPipelineBarrier(cmd, b.src_stages, b.dst_stages, 0, 0, nullptr, 1, &barrier, 0, nullptr);
            });
    }

    // The image counterpart: an image-memory barrier that also carries the
    // layout transition the tracker computed. Same hoisting as buffers — a
    // vkCmdPipelineBarrier is illegal inside dynamic rendering, so a transition
    // discovered mid-pass (a compute-written image about to be sampled) lands
    // just before begin_rendering.
    void record_image_barrier_(std::shared_ptr<Image> image, ResourceTracker::ImageBarrier b)
    {
        hoist_or_push_(
            [image = std::move(image), b](VkCommandBuffer cmd, const FrameContext& frame)
            {
                VkImageMemoryBarrier barrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = b.src_access,
                    .dstAccessMask = b.dst_access,
                    .oldLayout = b.old_layout,
                    .newLayout = b.new_layout,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = image->vk_image(),
                    .subresourceRange = {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = image->mip_levels(),
                        .baseArrayLayer = 0,
                        // All layers transition together: the tracker holds one
                        // layout per image, and a cube/array is used as a whole.
                        // A volume spells that VK_REMAINING_ARRAY_LAYERS.
                        .layerCount = image->barrier_layers(image->array_layers())}};
                frame.vk->vkCmdPipelineBarrier(cmd, b.src_stages, b.dst_stages, 0, 0, nullptr, 0, nullptr, 1, &barrier);
            });
    }

    // Push a recorded barrier, or hoist it before begin_rendering when inside a
    // rendering scope. Shared by the buffer and image paths.
    void hoist_or_push_(std::function<void(VkCommandBuffer, const FrameContext&)> lambda)
    {
        if (in_rendering_)
        {
            commands_.insert(commands_.begin() + rendering_insert_pos_, std::move(lambda));
            ++rendering_insert_pos_;
        }
        else
        {
            commands_.push_back(std::move(lambda));
        }
    }

    // A recorded timestamp write. Captures `this` (safe: the lambda only runs
    // inside this->execute) and reads timer_pool_ at execute, so it no-ops when
    // timestamps are unsupported and follows the pool across a grow.
    void record_timer_write_(std::uint32_t slot, VkPipelineStageFlagBits stage)
    {
        commands_.push_back(
            [this, slot, stage](VkCommandBuffer cmd, const FrameContext& frame)
            {
                if (timer_pool_ != VK_NULL_HANDLE)
                {
                    frame.vk->vkCmdWriteTimestamp(cmd, stage, timer_pool_, slot);
                }
            });
    }

    // Best-effort query pool sized for `needed` slots. Queries timestamp
    // support once; on an unsupported device timer_pool_ stays null and every
    // timer becomes a silent no-op (timer_ms returns None). Grows by recreating
    // (deferred destroy of the old pool) — rare, only when a later recording
    // declares more scopes than any before it.
    void ensure_timer_pool_(std::size_t needed)
    {
        if (timer_pool_ != VK_NULL_HANDLE && timer_capacity_ >= needed)
        {
            return;
        }
        if (!timer_supported_.has_value())
        {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(context_->physical_device(), &props);
            std::uint32_t family_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(context_->physical_device(), &family_count, nullptr);
            std::vector<VkQueueFamilyProperties> families(family_count);
            vkGetPhysicalDeviceQueueFamilyProperties(context_->physical_device(), &family_count, families.data());
            const std::uint32_t gf = context_->graphics_queue_family();
            const bool ok = props.limits.timestampPeriod > 0.0f && gf < family_count &&
                            families[gf].timestampValidBits != 0;
            timer_supported_ = ok;
            if (ok)
            {
                timer_period_ = props.limits.timestampPeriod;
                timer_valid_bits_ = families[gf].timestampValidBits;
            }
        }
        if (!*timer_supported_)
        {
            return;
        }

        if (timer_pool_ != VK_NULL_HANDLE)
        {
            context_->defer_destroy([vk = &context_->vk(), device = context_->device(), pool = timer_pool_]
                                    { vk->vkDestroyQueryPool(device, pool, nullptr); });
            timer_pool_ = VK_NULL_HANDLE;
        }
        VkQueryPoolCreateInfo poolInfo{
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = static_cast<std::uint32_t>(needed),
            .pipelineStatistics = 0};
        if (context_->vk().vkCreateQueryPool(context_->device(), &poolInfo, nullptr, &timer_pool_) != VK_SUCCESS)
        {
            timer_pool_ = VK_NULL_HANDLE;
            return;
        }
        timer_capacity_ = static_cast<std::uint32_t>(needed);
    }

    // The occlusion counterpart. Simpler than the timer pool: occlusion queries
    // are core Vulkan with no feature bit and no per-queue-family validity to
    // check, so there is nothing to probe — only the allocation can fail, and a
    // failure leaves the pool null and every query reporting None.
    void ensure_occlusion_pool_(std::size_t needed)
    {
        if (occlusion_pool_ != VK_NULL_HANDLE && occlusion_capacity_ >= needed)
        {
            return;
        }
        if (occlusion_pool_ != VK_NULL_HANDLE)
        {
            context_->defer_destroy([vk = &context_->vk(), device = context_->device(), pool = occlusion_pool_]
                                    { vk->vkDestroyQueryPool(device, pool, nullptr); });
            occlusion_pool_ = VK_NULL_HANDLE;
        }
        VkQueryPoolCreateInfo poolInfo{
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queryType = VK_QUERY_TYPE_OCCLUSION,
            .queryCount = static_cast<std::uint32_t>(needed),
            .pipelineStatistics = 0};
        if (context_->vk().vkCreateQueryPool(context_->device(), &poolInfo, nullptr, &occlusion_pool_) != VK_SUCCESS)
        {
            occlusion_pool_ = VK_NULL_HANDLE;
            return;
        }
        occlusion_capacity_ = static_cast<std::uint32_t>(needed);
    }

    // Residency bookkeeping, kept apart from track_use_ on purpose: that one
    // returns early with auto_barriers=False, and waiting for a buffer's fill
    // to land is not a barrier the caller can take over. Buffers with no
    // pending upload are skipped, so a recording of DYNAMIC buffers stores
    // nothing.
    void record_buffer_use_(const std::shared_ptr<Buffer>& buffer)
    {
        if (buffer && buffer->upload_serial() != 0)
        {
            used_buffers_.push_back(buffer);
        }
    }

    // Everything the three indirect verbs check, in one place so they cannot
    // disagree about which buffer is legal or what the message says.
    std::expected<void, Error> check_indirect_(
        const std::shared_ptr<Buffer>& buffer,
        VkDeviceSize offset,
        std::uint32_t count,
        VkDeviceSize stride,
        VkDeviceSize argument_size,
        const char* what)
    {
        // A stride smaller than the struct would make consecutive commands
        // overlap, and one that is not a multiple of 4 puts the next command's
        // 32-bit words on an unaligned address
        // (VUID-vkCmdDrawIndirect-drawCount-00476).
        if (stride < argument_size || stride % 4 != 0)
        {
            return std::unexpected(err_resource(
                std::format(
                    "{}: stride must be at least {} bytes (the size of one argument struct) and a "
                    "multiple of 4, got {}. stride= exists to leave room for per-draw data BETWEEN "
                    "the argument structs, so it can only be larger.",
                    what,
                    argument_size,
                    stride)));
        }
        if (!buffer)
        {
            return std::unexpected(err_resource(std::format("{}: buffer is null", what)));
        }
        // Only BufferType::STORAGE carries VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, so
        // this names the fix instead of leaving the layers to report a usage flag.
        if (buffer->buffer_type() != BufferType::STORAGE)
        {
            return std::unexpected(err_resource(
                std::format(
                    "{}: the arguments must live in a BufferType.STORAGE buffer, which is the "
                    "one that carries the indirect usage flag. A compute shader writing the "
                    "draw arguments needs it to be a storage buffer anyway.",
                    what)));
        }
        // The spec requires a 4-byte-aligned offset, and every argument struct is a
        // run of 32-bit words, so an unaligned one is always a mistake.
        if (offset % 4 != 0)
        {
            return std::unexpected(
                err_resource(std::format("{}: offset must be a multiple of 4, got {}", what, offset)));
        }
        if (count == 0)
        {
            return std::unexpected(err_resource(
                std::format(
                    "{}: count must be at least 1. To draw nothing, write 0 into the "
                    "instanceCount of the argument struct — that is the GPU-side way to say it.",
                    what)));
        }
        // The last command needs only its own struct, not a whole stride: the
        // padding stride= leaves is BETWEEN commands, so a buffer sized exactly
        // for the data is legal and must not be refused. With the default packed
        // stride the two are the same number.
        const VkDeviceSize needed = static_cast<VkDeviceSize>(count - 1) * stride + argument_size;
        if (!fits_within(offset, needed, buffer->size()))
        {
            return std::unexpected(err_resource(
                std::format(
                    "{}: {} argument struct(s) of {} bytes at a stride of {} from offset {} need {} "
                    "bytes, but the buffer is {}",
                    what,
                    count,
                    argument_size,
                    stride,
                    offset,
                    needed,
                    buffer->size())));
        }
        // count>1 is multiDrawIndirect, which is NOT free: it is a feature bit, and
        // this is the release that finally gives Feature.MULTI_DRAW_INDIRECT an API
        // to be reachable through. Checked at record time rather than in the
        // binding, so the C++ API is as safe as the Python one.
        if (count > 1 && !context_->supports(Feature::MULTI_DRAW_INDIRECT))
        {
            return std::unexpected(err_unsupported(
                std::format(
                    "{}: count>1 requires the MULTI_DRAW_INDIRECT feature. Create the Context "
                    "with features=[bz.Feature.MULTI_DRAW_INDIRECT] (or optional=[...]), or "
                    "issue one call per draw.",
                    what)));
        }
        return {};
    }

    // The count buffer gets the same three checks the argument buffer gets — it is
    // read by the same command processor at the same stage, and 4 bytes is a
    // VkDeviceSize past the end just as readily as 16 are. The feature is the one
    // thing that differs: drawIndirectCount is optional, and without it the entry
    // point is a null pointer in the dispatch table rather than a diagnostic.
    std::expected<void, Error> check_count_buffer_(
        const std::shared_ptr<Buffer>& count_buffer,
        VkDeviceSize count_offset,
        const char* what)
    {
        if (!count_buffer)
        {
            return {};
        }
        if (!context_->supports(Feature::DRAW_INDIRECT_COUNT))
        {
            return std::unexpected(err_unsupported(
                std::format(
                    "{}: count_buffer requires the DRAW_INDIRECT_COUNT feature. Create the "
                    "Context with optional=[bz.Feature.DRAW_INDIRECT_COUNT], or write 0 into "
                    "the instanceCount of the commands you do not want.",
                    what)));
        }
        if (count_buffer->buffer_type() != BufferType::STORAGE)
        {
            return std::unexpected(err_resource(
                std::format(
                    "{}: the count must live in a BufferType.STORAGE buffer, which is the one "
                    "that carries the indirect usage flag. A compute shader writing the count "
                    "needs it to be a storage buffer anyway.",
                    what)));
        }
        if (count_offset % 4 != 0)
        {
            return std::unexpected(
                err_resource(std::format("{}: count_offset must be a multiple of 4, got {}", what, count_offset)));
        }
        if (!fits_within(count_offset, sizeof(std::uint32_t), count_buffer->size()))
        {
            return std::unexpected(err_resource(
                std::format(
                    "{}: the count is 4 bytes at offset {}, but the count buffer is {}",
                    what,
                    count_offset,
                    count_buffer->size())));
        }
        return {};
    }

    // The command processor reads the arguments at DRAW_INDIRECT, which is earlier
    // than any shader stage — so a compute pass that wrote them in this recording
    // needs the barrier this produces, and hoist_or_push_ lifts it out of a
    // rendering scope on its own.
    void track_indirect_(const std::shared_ptr<Buffer>& buffer)
    {
        track_use_(buffer, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT, false);
        record_buffer_use_(buffer);
    }

    void track_use_(
        const std::shared_ptr<Buffer>& buffer,
        VkPipelineStageFlags stages,
        VkAccessFlags access,
        bool writes)
    {
        if (!auto_barriers_)
        {
            return;
        }
        if (writes)
        {
            tracked_writes_ = true;
        }
        // Only a STORAGE buffer carries VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, so it
        // is the only type a shader can write. The tracker uses that to narrow
        // its first-use floor rather than to switch it off — every type can be
        // written by cmd.copy_buffer and cmd.fill_buffer.
        const bool shader_writable = buffer->buffer_type() == BufferType::STORAGE;
        if (auto b = tracker_.use(buffer.get(), stages, access, writes, shader_writable))
        {
            record_barrier_(buffer, *b);
        }
    }

    void track_image_use_(
        const std::shared_ptr<Image>& image,
        VkImageLayout layout,
        VkPipelineStageFlags stages,
        VkAccessFlags access,
        bool writes)
    {
        if (!auto_barriers_)
        {
            return;
        }
        if (writes)
        {
            tracked_writes_ = true;
        }
        if (auto b = tracker_.use_image(image.get(), layout, stages, access, writes))
        {
            record_image_barrier_(image, *b);
        }
    }

    // Does the pipeline bound at this bind point write (set, binding)?
    //
    // Asks each of its shaders' reflection at record time, so a hot-reload
    // replace() is picked up without invalidating anything. Fail-open on purpose:
    // with no pipeline bound there is nothing to ask, and the answer has to be the
    // conservative one — a draw with no pipeline is a bug the layers name precisely,
    // and guessing "not written" there would silently drop a real barrier.
    bool pipeline_writes_(const std::shared_ptr<Pipeline>& pipeline, std::uint32_t set, std::uint32_t binding) const
    {
        if (!pipeline)
        {
            return true;
        }
        return std::ranges::any_of(
            pipeline->shaders(),
            [&](const std::shared_ptr<ShaderModule>& s) { return s && s->reflection().writes(set, binding); });
    }

    // The shared body of track_draw_ and track_dispatch_. Before 0.19 these were
    // two functions that disagreed about the same question: the graphics one called
    // every storage buffer a READ (so a fragment SSBO write was invisible) and
    // handled storage images not at all, while the compute one called both
    // READ+WRITE unconditionally. Now they differ only in their stage mask, which is
    // the only thing that was ever really different about them.
    //
    // What reflection changes in each direction:
    //
    //  * Graphics gains the writes it never saw. A fragment imageStore was worse
    //    than untracked — DescriptorSet::set_storage_image already recorded the
    //    image as resting in GENERAL, so the layout the descriptor promised and the
    //    layout the image was in disagreed unless the user wrote a barrier by hand.
    //  * Compute LOSES barriers it did not need. `use(..., writes=true)` wipes read
    //    state, so two dispatches that only read the same input SSBO used to get a
    //    WAW barrier between them, and tracked_writes_ also switched on the
    //    per-replay memory barrier. A `readonly buffer` shared down a chain of
    //    passes is the common case, not an exotic one.
    //
    // Safe in both directions because of the fail-open invariant in
    // SpirvReflect.hpp: a barrier only ever disappears on a positive proof that no
    // store, atomic or imageWrite touches that binding.
    void track_descriptor_uses_(
        const std::unordered_map<uint32_t, std::shared_ptr<DescriptorSet>>& sets,
        const std::shared_ptr<Pipeline>& pipeline,
        VkPipelineStageFlags stages)
    {
        if (!auto_barriers_)
        {
            return;
        }
        for (const auto& [set_index, set] : sets)
        {
            for (const auto& bb : set->buffers())
            {
                if (bb.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                {
                    const bool writes = pipeline_writes_(pipeline, set_index, bb.binding);
                    track_use_(
                        bb.buffer,
                        stages,
                        writes ? (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT) : VK_ACCESS_SHADER_READ_BIT,
                        writes);
                }
                else if (bb.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                {
                    // A uniform buffer is read-only by definition, so there is
                    // nothing for reflection to decide.
                    track_use_(bb.buffer, stages, VK_ACCESS_UNIFORM_READ_BIT, false);
                }
            }
            for (const auto& bi : set->images())
            {
                if (bi.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                {
                    // GENERAL either way: it is the only layout a storage image is
                    // read or written in, so an unwritten one still needs the
                    // transition. Only the access mask and the WAW ordering narrow.
                    const bool writes = pipeline_writes_(pipeline, set_index, bi.binding);
                    track_image_use_(
                        bi.image,
                        VK_IMAGE_LAYOUT_GENERAL,
                        stages,
                        writes ? (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT) : VK_ACCESS_SHADER_READ_BIT,
                        writes);
                }
                else if (bi.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER && tracker_.tracks(bi.image.get()))
                {
                    // A sampled image only needs a barrier if the tracker has
                    // already seen it — i.e. something wrote it earlier in this
                    // recording and it is still in GENERAL. An uploaded texture the
                    // tracker never saw rests in SHADER_READ_ONLY and is left
                    // untouched, so ordinary texturing pays nothing.
                    track_image_use_(
                        bi.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, stages, VK_ACCESS_SHADER_READ_BIT, false);
                }
            }
        }
    }

    // The stage mask is the Context's, not a VERTEX|FRAGMENT literal: with
    // tessellation or geometry enabled a graphics pipeline has stages those two bits
    // do not name, and a barrier that omits a stage does not cover the read it was
    // recorded for. Legal by construction — the mask only ever holds bits whose
    // feature the device enabled.
    void track_draw_()
    {
        track_descriptor_uses_(bound_graphics_sets_, bound_graphics_pipeline_, context_->all_shader_stages());
    }

    void track_dispatch_()
    {
        track_descriptor_uses_(bound_compute_sets_, bound_compute_pipeline_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    std::shared_ptr<Context> context_;
    std::vector<VkCommandBuffer> command_buffers_;
    std::vector<std::function<void(VkCommandBuffer, const FrameContext&)>> commands_;
    std::vector<std::shared_ptr<DescriptorSet>> used_sets_;
    std::vector<std::shared_ptr<Buffer>> used_buffers_;

    // Frame serial of the last replay; UINT64_MAX = never replayed. A sentinel
    // rather than 0, because the headless ctx.submit legitimately runs its first
    // submit at serial 0 (it advances the ring after submitting, not before).
    std::uint64_t recorded_serial_ = UINT64_MAX;

    // ── record-time state (reset by begin(), never touched at execute) ──
    bool auto_barriers_ = true;
    ResourceTracker tracker_;
    bool tracked_writes_ = false;
    bool in_rendering_ = false;
    std::size_t rendering_insert_pos_ = 0;
    std::unordered_map<uint32_t, std::shared_ptr<DescriptorSet>> bound_graphics_sets_;
    std::unordered_map<uint32_t, std::shared_ptr<DescriptorSet>> bound_compute_sets_;
    // Record-time state, so the tracker can read the bound shaders' reflection.
    // Cleared in begin() with the sets, because both describe one recording.
    std::shared_ptr<Pipeline> bound_graphics_pipeline_;
    std::shared_ptr<Pipeline> bound_compute_pipeline_;
    // Whichever of the two was bound last. Push constants belong to a layout, not
    // to a bind point, so "the pipeline that is already known" is this one.
    std::shared_ptr<Pipeline> bound_last_pipeline_;

    // ── GPU timers (query pool survives begin(); results read after submit) ──
    VkQueryPool timer_pool_ = VK_NULL_HANDLE;
    std::uint32_t timer_capacity_ = 0;
    float timer_period_ = 0.0f;
    std::uint32_t timer_valid_bits_ = 0;
    std::optional<bool> timer_supported_;    // queried once, lazily
    std::size_t timer_count_ = 0;            // timers declared this recording
    std::uint64_t recording_generation_ = 0; // bumped by begin(); stale-handle guard

    // ── Occlusion queries (same lifetime rules as the timer pool above) ──
    VkQueryPool occlusion_pool_ = VK_NULL_HANDLE;
    std::uint32_t occlusion_capacity_ = 0;
    // Read once per recording rather than per query: the feature set is fixed for
    // the Context's whole life, so asking again per draw would be the same answer
    // through a hash lookup.
    bool precise_occlusion_ = context_->supports(Feature::PRECISE_OCCLUSION);
    std::size_t occlusion_count_ = 0;

    // Depth of the label nesting declared this recording, so end_label() can
    // refuse to close one that was never opened.
    std::size_t open_labels_ = 0;
};
