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
        cmd->command_buffers_.resize(ctx->frames_in_flight(), VK_NULL_HANDLE);

        VkCommandBufferAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr,
            .commandPool = ctx->command_pool(),
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = ctx->frames_in_flight()};

        if (auto e = check(
                vkAllocateCommandBuffers(ctx->device(), &allocInfo, cmd->command_buffers_.data()),
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
                context_->defer_destroy([device = context_->device(), pool = timer_pool_]
                                        { vkDestroyQueryPool(device, pool, nullptr); });
            }
            context_->defer_destroy(
                [device = context_->device(), pool = context_->command_pool(), buffers = std::move(command_buffers_)]
                { vkFreeCommandBuffers(device, pool, static_cast<uint32_t>(buffers.size()), buffers.data()); });
        }
    }

    CommandBuffer(const CommandBuffer&) = delete;
    CommandBuffer& operator=(const CommandBuffer&) = delete;

    CommandBuffer& begin()
    {
        commands_.clear();
        used_sets_.clear();
        // A reused command buffer must forget its previous recording entirely,
        // or it would emit barriers against uses that no longer exist.
        tracker_.reset();
        bound_graphics_sets_.clear();
        bound_compute_sets_.clear();
        tracked_writes_ = false;
        in_rendering_ = false;
        rendering_insert_pos_ = 0;
        // Timers are re-declared each recording; the query pool itself is kept
        // and reset (vkCmdResetQueryPool) at the top of every replay. Bumping
        // the generation invalidates handles from the previous recording.
        timer_count_ = 0;
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
    CommandBuffer& begin_rendering(
        std::shared_ptr<RenderTarget> target, const std::vector<std::array<float, 4>>& clear_colors)
    {
        commands_.push_back(
            [clear_colors, target](VkCommandBuffer cmd, const FrameContext& frame)
            {
                RenderTarget* rt = target.get();

                // Every colour attachment enters COLOR_ATTACHMENT_OPTIMAL. UNDEFINED
                // as the source: contents are cleared each pass anyway.
                for (uint32_t i = 0; i < rt->color_count(); ++i)
                {
                    VkImageMemoryBarrier barrier{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        .pNext = nullptr,
                        .srcAccessMask = 0,
                        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                        .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = rt->color_image(i),
                        .subresourceRange = {
                            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .baseMipLevel = 0,
                            .levelCount = 1,
                            .baseArrayLayer = 0,
                            .layerCount = 1}};

                    vkCmdPipelineBarrier(
                        cmd,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
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
                        vkCmdPipelineBarrier(
                            cmd,
                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
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

                if (rt->depth_image() != VK_NULL_HANDLE)
                {
                    VkImageMemoryBarrier depthBarrier{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        .pNext = nullptr,
                        .srcAccessMask = 0,
                        .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                        .newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = rt->depth_image(),
                        .subresourceRange = {
                            .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                            .baseMipLevel = 0,
                            .levelCount = 1,
                            .baseArrayLayer = 0,
                            .layerCount = 1}};

                    vkCmdPipelineBarrier(
                        cmd,
                        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
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
                        vkCmdPipelineBarrier(
                            cmd,
                            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
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
                    const std::array<float, 4> cc =
                        clear_colors.empty()
                            ? std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}
                            : (i < clear_colors.size() ? clear_colors[i] : clear_colors[0]);
                    // MSAA: render into the multisampled view, resolve (averaging
                    // the samples) into the single-sample target. The multisampled
                    // image is transient — only the resolve is kept (DONT_CARE).
                    const bool resolve = rt->color_resolve_view(i) != VK_NULL_HANDLE;
                    colorAttachments.push_back(
                        {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                         .pNext = nullptr,
                         .imageView = rt->color_view(i),
                         .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                         .resolveMode = resolve ? VK_RESOLVE_MODE_AVERAGE_BIT : VK_RESOLVE_MODE_NONE,
                         .resolveImageView = resolve ? rt->color_resolve_view(i) : VK_NULL_HANDLE,
                         .resolveImageLayout =
                             resolve ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
                         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                         .storeOp = resolve ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE,
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
                    .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                    .resolveMode = depthResolve ? VK_RESOLVE_MODE_SAMPLE_ZERO_BIT : VK_RESOLVE_MODE_NONE,
                    .resolveImageView = depthResolve ? rt->depth_resolve_view() : VK_NULL_HANDLE,
                    .resolveImageLayout =
                        depthResolve ? VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
                    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                    // A depth that will be consumed (shadow maps) must be stored;
                    // the swapchain's scratch depth keeps DONT_CARE.
                    .storeOp = rt->depth_final_layout() == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL
                                   ? VK_ATTACHMENT_STORE_OP_DONT_CARE
                                   : VK_ATTACHMENT_STORE_OP_STORE,
                    .clearValue = {.depthStencil = {1.0f, 0}}};

                VkRenderingInfo renderingInfo{
                    .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .renderArea = {{0, 0}, rt->extent()},
                    .layerCount = 1,
                    .viewMask = 0,
                    .colorAttachmentCount = static_cast<uint32_t>(colorAttachments.size()),
                    .pColorAttachments = colorAttachments.empty() ? nullptr : colorAttachments.data(),
                    .pDepthAttachment = rt->depth_view() != VK_NULL_HANDLE ? &depthAttachment : nullptr,
                    .pStencilAttachment = nullptr};

                vkCmdBeginRendering(cmd, &renderingInfo);

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
                vkCmdSetViewport(cmd, 0, 1, &viewport);

                VkRect2D scissor{.offset = {0, 0}, .extent = rt->extent()};
                vkCmdSetScissor(cmd, 0, 1, &scissor);
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
                vkCmdEndRendering(cmd);

                // Every colour attachment retires to the target's final layout.
                // (Was VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, unconditionally, on colour 0
                // only â€” that one constant is why nothing but a swapchain could ever
                // be drawn into.)
                for (uint32_t i = 0; i < target->color_count(); ++i)
                {
                    // With MSAA it is the resolve image that must reach the final
                    // layout (present / sampleable); the multisampled image is
                    // transient and discarded.
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
                            .baseMipLevel = 0,
                            .levelCount = 1,
                            .baseArrayLayer = 0,
                            .layerCount = 1}};

                    vkCmdPipelineBarrier(
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
                }

                // Depth retires to its own final layout when it will be consumed
                // (offscreen: SHADER_READ_ONLY, which is what makes `target.depth`
                // sampleable). The swapchain's depth stays put â€” no barrier.
                if (target->depth_image() != VK_NULL_HANDLE &&
                    target->depth_final_layout() != VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL)
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
                        .oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                        .newLayout = target->depth_final_layout(),
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = final_depth,
                        .subresourceRange = {
                            .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                            .baseMipLevel = 0,
                            .levelCount = 1,
                            .baseArrayLayer = 0,
                            .layerCount = 1}};

                    vkCmdPipelineBarrier(
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

                // Runs at execute() time, inside a real submit â€” so the target learns
                // its images have left UNDEFINED exactly when that becomes true, and a
                // recorded-but-never-submitted command buffer marks nothing.
                target->on_rendering_recorded();
            });
        in_rendering_ = false;
        return *this;
    }

    // Explicit override for split-screen and similar. The no-argument version is
    // gone: begin_rendering already covers the whole-target case.
    CommandBuffer& set_viewport(float x, float y, float width, float height)
    {
        commands_.push_back(
            [x, y, width, height](VkCommandBuffer cmd, const FrameContext&)
            {
                VkViewport viewport{
                    .x = x, .y = y, .width = width, .height = height, .minDepth = 0.0f, .maxDepth = 1.0f};
                vkCmdSetViewport(cmd, 0, 1, &viewport);
            });
        return *this;
    }

    CommandBuffer& set_scissor(std::int32_t x, std::int32_t y, std::uint32_t width, std::uint32_t height)
    {
        commands_.push_back(
            [x, y, width, height](VkCommandBuffer cmd, const FrameContext&)
            {
                VkRect2D scissor{.offset = {x, y}, .extent = {width, height}};
                vkCmdSetScissor(cmd, 0, 1, &scissor);
            });
        return *this;
    }

    CommandBuffer& bind_pipeline(std::shared_ptr<Pipeline> pipeline)
    {
        commands_.push_back([pipeline](VkCommandBuffer cmd, const FrameContext&)
                            { vkCmdBindPipeline(cmd, pipeline->bind_point(), pipeline->get()); });
        return *this;
    }

    CommandBuffer& bind_vertex_buffer(std::shared_ptr<Buffer> buffer)
    {
        // The read truly happens at draw, but a barrier placed before the bind
        // is still before the draw — sound, and simpler than deferring it.
        track_use_(buffer, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT, false);
        commands_.push_back(
            [buffer](VkCommandBuffer cmd, const FrameContext&)
            {
                VkBuffer vertexBuffers[] = {buffer->get()};
                VkDeviceSize offsets[] = {0};
                vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
            });
        return *this;
    }

    CommandBuffer& bind_index_buffer(std::shared_ptr<Buffer> buffer)
    {
        track_use_(buffer, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_ACCESS_INDEX_READ_BIT, false);
        commands_.push_back(
            [buffer](VkCommandBuffer cmd, const FrameContext&)
            {
                // Derived from the buffer rather than hardcoded to UINT32: create_buffer
                // accepts UINT16 indices, which used to be read back at half count.
                vkCmdBindIndexBuffer(cmd, buffer->get(), 0, buffer->index_type());
            });
        return *this;
    }

    CommandBuffer& draw(uint32_t vertexCount)
    {
        track_draw_();
        commands_.push_back([vertexCount](VkCommandBuffer cmd, const FrameContext&)
                            { vkCmdDraw(cmd, vertexCount, 1, 0, 0); });
        return *this;
    }

    CommandBuffer& draw_indexed(uint32_t indexCount, uint32_t firstIndex = 0, int32_t vertexOffset = 0)
    {
        track_draw_();
        commands_.push_back([indexCount, firstIndex, vertexOffset](VkCommandBuffer cmd, const FrameContext&)
                            { vkCmdDrawIndexed(cmd, indexCount, 1, firstIndex, vertexOffset, 0); });
        return *this;
    }

    CommandBuffer& draw_indexed_instanced(
        uint32_t indexCount,
        uint32_t instanceCount,
        uint32_t firstIndex = 0,
        int32_t vertexOffset = 0)
    {
        track_draw_();
        commands_.push_back(
            [indexCount, instanceCount, firstIndex, vertexOffset](VkCommandBuffer cmd, const FrameContext&)
            { vkCmdDrawIndexed(cmd, indexCount, instanceCount, firstIndex, vertexOffset, 0); });
        return *this;
    }

    CommandBuffer& dispatch(uint32_t groupCountX, uint32_t groupCountY = 1, uint32_t groupCountZ = 1)
    {
        track_dispatch_();
        commands_.push_back([groupCountX, groupCountY, groupCountZ](VkCommandBuffer cmd, const FrameContext&)
                            { vkCmdDispatch(cmd, groupCountX, groupCountY, groupCountZ); });
        return *this;
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
            return std::unexpected(err_resource(
                "cmd.barrier() is not allowed inside a rendering scope; "
                "record it before begin_rendering"));
        }
        const StageAccess s = to_vk(src);
        const StageAccess d = to_vk(dst);
        record_barrier_(std::move(buffer), {s.stages, d.stages, s.access, d.access});
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
            return std::unexpected(err_resource(
                "cmd.barrier() is not allowed inside a rendering scope; "
                "record it before begin_rendering"));
        }
        const auto old_layout = image_layout_for(src);
        const auto new_layout = image_layout_for(dst);
        if (!old_layout || !new_layout)
        {
            return std::unexpected(err_resource(
                "cmd.barrier(image, ...) takes Access.SHADER_WRITE (GENERAL) or "
                "Access.SHADER_READ (SHADER_READ_ONLY); other accesses are buffer-only"));
        }
        const StageAccess s = to_vk(src);
        const StageAccess d = to_vk(dst);
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
            return std::unexpected(err_resource(
                "cmd.generate_mipmaps() is not allowed inside a rendering scope; "
                "record it before begin_rendering"));
        }
        if (image->mip_levels() <= 1)
        {
            return std::unexpected(err_resource(
                "generate_mipmaps: image has a single mip level; create it with "
                "mip_levels>1 (empty) or mipmaps=True (from pixels/files)"));
        }
        if (!Image::can_generate_mips(*context_, image->format()))
        {
            return std::unexpected(err_resource(
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
        const StageAccess s = to_vk(src);
        Image* img = image.get();
        commands_.push_back(
            [image = std::move(image), layout = *src_layout, s](VkCommandBuffer cmd, const FrameContext&)
            { image->record_generate_mipmaps(cmd, layout, s.stages, s.access); });
        // The image now rests in SHADER_READ_ONLY across every level; keep the
        // tracker in sync so a later automatic sample emits no extra transition.
        if (auto_barriers_)
        {
            tracker_.note_image_layout(
                img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, kAllShaderStages, VK_ACCESS_SHADER_READ_BIT);
        }
        return {};
    }

    // ── GPU timers ──────────────────────────────────────────────────────────
    //
    // A GPU timer is a pair of query slots — exactly a Vulkan timestamp query.
    // The Python-facing handle (class Timer, in main.cpp) owns one such pair:
    // cmd.timer() records the opening timestamp and hands back the handle, which
    // is stopped explicitly (t.stop()) or by a `with`, and read back off itself
    // (t.ms). The handle IS the identity — no name, no key — so multiple, nested
    // and overlapping timers all just work.
    //
    // Unlike 0.8's frame.gpu_time_ms this needs no window and no begin_frame:
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

    // The measured time of one timer in milliseconds, or nullopt when:
    // timestamps are unsupported, the handle is from a superseded recording, or
    // the results are not ready (a submit still in flight — a blocking headless
    // submit always is).
    std::optional<double> read_timer(std::size_t index, std::uint64_t generation) const
    {
        if (timer_pool_ == VK_NULL_HANDLE || generation != recording_generation_ || index >= timer_count_)
        {
            return std::nullopt;
        }
        std::uint64_t ts[2] = {0, 0};
        if (vkGetQueryPoolResults(
                context_->device(),
                timer_pool_,
                static_cast<std::uint32_t>(2 * index),
                2,
                sizeof(ts),
                ts,
                sizeof(std::uint64_t),
                VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
        {
            return std::nullopt; // VK_NOT_READY: submit not finished
        }
        const std::uint64_t mask = timer_valid_bits_ >= 64 ? ~std::uint64_t{0}
                                                           : ((std::uint64_t{1} << timer_valid_bits_) - 1);
        const std::uint64_t delta = (ts[1] - ts[0]) & mask;
        return static_cast<double>(delta) * static_cast<double>(timer_period_) / 1.0e6;
    }

    // No stage argument: the Pipeline already knows which stages its push constant
    // range covers, so passing a mismatched one was a validation error for no gain.
    CommandBuffer& push_constants(std::shared_ptr<Pipeline> pipeline, uint32_t offset, uint32_t size, const void* data)
    {
        std::vector<uint8_t> buffer(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + size);
        commands_.push_back(
            [pipeline, offset, size, buffer](VkCommandBuffer cmd, const FrameContext&)
            {
                vkCmdPushConstants(
                    cmd, pipeline->layout(), pipeline->push_constant_stages(), offset, size, buffer.data());
            });
        return *this;
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
                vkCmdBindDescriptorSets(cmd, pipeline->bind_point(), pipeline->layout(), setIndex, 1, &set, 0, nullptr);
            });
        return *this;
    }

    const std::vector<std::shared_ptr<DescriptorSet>>& used_sets() const
    {
        return used_sets_;
    }

    VkCommandBuffer get(std::uint32_t frame_index) const
    {
        return command_buffers_[frame_index];
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
                vkCmdResetQueryPool(vkCmd, timer_pool_, 0, timer_capacity_);
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
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_UNIFORM_READ_BIT |
                                 VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT};
            constexpr VkPipelineStageFlags stages = kAllShaderStages | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
            vkCmdPipelineBarrier(vkCmd, stages, stages, 0, 1, &barrier, 0, nullptr, 0, nullptr);
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
            [buffer = std::move(buffer), b](VkCommandBuffer cmd, const FrameContext&)
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
                vkCmdPipelineBarrier(cmd, b.src_stages, b.dst_stages, 0, 0, nullptr, 1, &barrier, 0, nullptr);
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
            [image = std::move(image), b](VkCommandBuffer cmd, const FrameContext&)
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
                        .layerCount = image->array_layers()}};
                vkCmdPipelineBarrier(cmd, b.src_stages, b.dst_stages, 0, 0, nullptr, 0, nullptr, 1, &barrier);
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
            [this, slot, stage](VkCommandBuffer cmd, const FrameContext&)
            {
                if (timer_pool_ != VK_NULL_HANDLE)
                {
                    vkCmdWriteTimestamp(cmd, stage, timer_pool_, slot);
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
            context_->defer_destroy([device = context_->device(), pool = timer_pool_]
                                    { vkDestroyQueryPool(device, pool, nullptr); });
            timer_pool_ = VK_NULL_HANDLE;
        }
        VkQueryPoolCreateInfo poolInfo{
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = static_cast<std::uint32_t>(needed),
            .pipelineStatistics = 0};
        if (vkCreateQueryPool(context_->device(), &poolInfo, nullptr, &timer_pool_) != VK_SUCCESS)
        {
            timer_pool_ = VK_NULL_HANDLE;
            return;
        }
        timer_capacity_ = static_cast<std::uint32_t>(needed);
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
        if (auto b = tracker_.use(buffer.get(), stages, access, writes))
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

    // Storage buffers in graphics shaders are conservatively READS. Writes are
    // invisible without shader reflection — the documented limit of auto mode;
    // cmd.barrier() covers that case by hand.
    void track_draw_()
    {
        if (!auto_barriers_)
        {
            return;
        }
        for (auto& [idx, set] : bound_graphics_sets_)
        {
            for (const auto& bb : set->buffers())
            {
                if (bb.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                {
                    track_use_(
                        bb.buffer,
                        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                        VK_ACCESS_SHADER_READ_BIT,
                        false);
                }
                else if (bb.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                {
                    track_use_(
                        bb.buffer,
                        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                        VK_ACCESS_UNIFORM_READ_BIT,
                        false);
                }
            }
            // A sampled image only needs a barrier if the tracker has already
            // seen it — i.e. compute wrote it earlier in this recording and it
            // is still in GENERAL. An uploaded texture the tracker never saw
            // rests in SHADER_READ_ONLY and is left untouched (the pre-0.9
            // behaviour), so ordinary texturing pays nothing.
            for (const auto& bi : set->images())
            {
                if (bi.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER && tracker_.tracks(bi.image.get()))
                {
                    track_image_use_(
                        bi.image,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                        VK_ACCESS_SHADER_READ_BIT,
                        false);
                }
            }
        }
    }

    // Compute SSBOs are conservatively READ+WRITE: without reflection there is
    // no telling a `readonly buffer` from a mutated one, and the pessimism
    // costs one barrier, not correctness.
    void track_dispatch_()
    {
        if (!auto_barriers_)
        {
            return;
        }
        for (auto& [idx, set] : bound_compute_sets_)
        {
            for (const auto& bb : set->buffers())
            {
                if (bb.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                {
                    track_use_(
                        bb.buffer,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                        true);
                }
                else if (bb.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                {
                    track_use_(bb.buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_UNIFORM_READ_BIT, false);
                }
            }
            // Storage images are conservatively READ+WRITE, like storage
            // buffers: without reflection a `readonly` image is indistinguishable
            // from a written one, and GENERAL is the only layout either is
            // accessed in. The transition to GENERAL (from UNDEFINED on first
            // use, or from a prior use's layout) rides on this barrier.
            for (const auto& bi : set->images())
            {
                if (bi.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                {
                    track_image_use_(
                        bi.image,
                        VK_IMAGE_LAYOUT_GENERAL,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                        true);
                }
            }
        }
    }

    std::shared_ptr<Context> context_;
    std::vector<VkCommandBuffer> command_buffers_;
    std::vector<std::function<void(VkCommandBuffer, const FrameContext&)>> commands_;
    std::vector<std::shared_ptr<DescriptorSet>> used_sets_;

    // ── record-time state (reset by begin(), never touched at execute) ──
    bool auto_barriers_ = true;
    ResourceTracker tracker_;
    bool tracked_writes_ = false;
    bool in_rendering_ = false;
    std::size_t rendering_insert_pos_ = 0;
    std::unordered_map<uint32_t, std::shared_ptr<DescriptorSet>> bound_graphics_sets_;
    std::unordered_map<uint32_t, std::shared_ptr<DescriptorSet>> bound_compute_sets_;

    // ── GPU timers (query pool survives begin(); results read after submit) ──
    VkQueryPool timer_pool_ = VK_NULL_HANDLE;
    std::uint32_t timer_capacity_ = 0;
    float timer_period_ = 0.0f;
    std::uint32_t timer_valid_bits_ = 0;
    std::optional<bool> timer_supported_;    // queried once, lazily
    std::size_t timer_count_ = 0;            // timers declared this recording
    std::uint64_t recording_generation_ = 0; // bumped by begin(); stale-handle guard
};
