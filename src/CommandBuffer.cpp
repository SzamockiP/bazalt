#include "CommandBuffer.hpp"

// The free record_image_* verbs live here, and so do Image::can_blit /
// can_generate_mips. The header reaches them transitively; this TU names the
// dependency it actually uses.
#include "Image.hpp"

#include <algorithm>
#include <format>

std::expected<std::shared_ptr<CommandBuffer>, Error> CommandBuffer::create(
    Context& context,
    std::optional<bool> auto_barriers)
{
    auto ctx = context.shared_from_this();
    auto cmd = std::shared_ptr<CommandBuffer>(new CommandBuffer(ctx));
    // Per-pass override of the Context-wide mode, so one hot pass can go
    // manual without flipping the whole application.
    cmd->auto_barriers_ = auto_barriers.value_or(ctx->auto_barriers());
    return cmd;
}

CommandBuffer::~CommandBuffer()
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
    }
}

CommandBuffer& CommandBuffer::begin()
{
    commands_.clear();
    used_sets_.clear();
    used_buffers_.clear();
    bound_graphics_sets_.clear();
    bound_compute_sets_.clear();
    bound_graphics_pipeline_.reset();
    bound_compute_pipeline_.reset();
    bound_last_pipeline_.reset();
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

// The transition half of opening a rendering scope: every attachment enters
// its attachment layout. Split from the begin half in 0.28, because the graph
// executor owns these transitions itself (its compile decides them with the
// whole frame in view) while the inline recorder keeps recording both halves
// back to back, in this order.
void record_render_pass_transitions_in(const VolkDeviceTable& vk, VkCommandBuffer cmd, RenderTarget& rt, bool preserve)
{
    // A depth attachment that carries a stencil aspect is one image
    // in one layout: DEPTH_ATTACHMENT_OPTIMAL covers the depth
    // aspect only, so a combined format needs the combined layout,
    // and every barrier, view and attachment info below reads both
    // from here.
    const VkImageAspectFlags depth_aspect = aspect_mask_for(rt.depth_format());
    const bool stencil = has_stencil(rt.depth_format());
    const VkImageLayout depth_layout = stencil ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                                               : VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;

    // Which layer/mip each attachment barrier must transition. Defaults
    // to {layer 0, mip 0, one of each}; a SubresourceTarget narrows it to
    // the single subresource its view renders into (render-to-layer/mip).
    const RenderTarget::Subresource color_sr = rt.color_subresource();
    const RenderTarget::Subresource depth_sr = rt.depth_subresource();

    // Every colour attachment enters COLOR_ATTACHMENT_OPTIMAL. UNDEFINED
    // as the source: contents are cleared each pass anyway.
    //
    // Except when preserving, where UNDEFINED discards exactly what
    // is about to be loaded. The source is then the layout the
    // previous pass retired to (the transitions-out half below), and the
    // source stage covers both ways the attachment can have got there:
    // written by an earlier pass, or sampled since.
    const VkImageLayout color_old_layout = preserve ? rt.final_layout() : VK_IMAGE_LAYOUT_UNDEFINED;
    const VkAccessFlags color_src_access = preserve ? (VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT)
                                                    : 0;
    const VkPipelineStageFlags color_src_stage =
        preserve ? (VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)
                 : VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    // LOAD_OP_LOAD reads the attachment, so preserving needs the
    // read bit as well as the write.
    const VkAccessFlags color_dst_access = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                           (preserve ? VK_ACCESS_COLOR_ATTACHMENT_READ_BIT : 0);
    for (uint32_t i = 0; i < rt.color_count(); ++i)
    {
        record_image_transition(
            vk,
            cmd,
            rt.color_image(i),
            color_old_layout,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            color_src_access,
            color_dst_access,
            color_src_stage,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            color_sr.base_mip,
            color_sr.mip_count,
            color_sr.layer_count,
            color_sr.base_layer);

        // With MSAA the single-sample resolve target is a second
        // attachment written this pass — it needs the same transition.
        if (rt.color_resolve_image(i) != VK_NULL_HANDLE)
        {
            record_image_transition(
                vk,
                cmd,
                rt.color_resolve_image(i),
                color_old_layout,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                color_src_access,
                color_dst_access,
                color_src_stage,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                color_sr.base_mip,
                color_sr.mip_count,
                color_sr.layer_count,
                color_sr.base_layer);
        }
    }

    // Depth follows colour: preserving takes it from the layout the
    // previous pass left it in, and a swapchain's scratch depth never
    // leaves DEPTH_ATTACHMENT_OPTIMAL, which is exactly what
    // depth_final_layout() reports for it.
    const VkImageLayout depth_old_layout = preserve ? rt.depth_final_layout() : VK_IMAGE_LAYOUT_UNDEFINED;
    const VkAccessFlags depth_src_access =
        preserve ? (VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT) : 0;
    const VkPipelineStageFlags depth_src_stage =
        preserve ? (VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)
                 : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    if (rt.depth_image() != VK_NULL_HANDLE)
    {
        const VkAccessFlags depth_dst_access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                               (preserve ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT : 0);
        record_image_transition(
            vk,
            cmd,
            rt.depth_image(),
            depth_old_layout,
            depth_layout,
            depth_src_access,
            depth_dst_access,
            depth_src_stage,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            depth_aspect,
            depth_sr.base_mip,
            depth_sr.mip_count,
            depth_sr.layer_count,
            depth_sr.base_layer);

        // MSAA depth resolves into a single-sample image (offscreen
        // only — a swapchain's scratch depth has no resolve target).
        if (rt.depth_resolve_image() != VK_NULL_HANDLE)
        {
            record_image_transition(
                vk,
                cmd,
                rt.depth_resolve_image(),
                depth_old_layout,
                depth_layout,
                depth_src_access,
                depth_dst_access,
                depth_src_stage,
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                depth_aspect,
                depth_sr.base_mip,
                depth_sr.mip_count,
                depth_sr.layer_count,
                depth_sr.base_layer);
        }
    }
}

// The begin half: attachment info, vkCmdBeginRendering, and the whole-target
// viewport and scissor.
void record_render_pass_begin(
    const VolkDeviceTable& vk,
    VkCommandBuffer cmd,
    RenderTarget& rt_ref,
    const std::optional<std::vector<std::array<float, 4>>>& clear_colors,
    float clear_depth,
    std::uint32_t clear_stencil)
{
    RenderTarget* rt = &rt_ref;
    const bool preserve = !clear_colors.has_value();
    const bool stencil = has_stencil(rt->depth_format());
    const VkImageLayout depth_layout = stencil ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                                               : VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;

    std::vector<VkRenderingAttachmentInfo> colorAttachments;
    colorAttachments.reserve(rt->color_count());
    for (uint32_t i = 0; i < rt->color_count(); ++i)
    {
        // One entry clears every attachment, N entries clear attachment
        // i with entry i, and a preserving pass has no clear at all.
        const std::array<float, 4> cc = [&]() -> std::array<float, 4>
        {
            if (preserve || clear_colors->empty())
            {
                return {0.0f, 0.0f, 0.0f, 1.0f};
            }
            return i < clear_colors->size() ? (*clear_colors)[i] : (*clear_colors)[0];
        }();
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
             .resolveImageLayout = resolve ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
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
        .pStencilAttachment = (stencil && rt->depth_view() != VK_NULL_HANDLE) ? &stencilAttachment : nullptr};

    vk.vkCmdBeginRendering(cmd, &renderingInfo);

    // Emitted automatically: set_viewport()/set_scissor() took no arguments
    // and silently read the swapchain, which is magic — just less legible
    // than doing it here. set_viewport(x, y, w, h) remains for the cases
    // that genuinely want something other than the whole target.
    VkViewport viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(rt->extent().width),
        .height = static_cast<float>(rt->extent().height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f};
    vk.vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{.offset = {0, 0}, .extent = rt->extent()};
    vk.vkCmdSetScissor(cmd, 0, 1, &scissor);
}

void record_render_pass_end(const VolkDeviceTable& vk, VkCommandBuffer cmd)
{
    vk.vkCmdEndRendering(cmd);
}

void record_render_pass_transitions_out(const VolkDeviceTable& vk, VkCommandBuffer cmd, RenderTarget& rt_ref)
{
    RenderTarget* target = &rt_ref;

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
        VkImage final_image = target->color_resolve_image(i) != VK_NULL_HANDLE ? target->color_resolve_image(i)
                                                                               : target->color_image(i);
        // The retire NAMES who reads next, and that is not decoration: a layout
        // transition is itself a write, so a destination scope of
        // (BOTTOM_OF_PIPE, 0) leaves the next reader unsynchronized against it.
        // A pass that samples this attachment later in the same submit —
        // render into a texture, then read it, which is what a G-buffer or a
        // post-process chain is — then reports READ_AFTER_WRITE under sync
        // validation. Depth has always named its reader (below); colour said
        // nothing, and the asymmetry was the bug.
        //
        // A swapchain is the exception and keeps the empty scope: PRESENT_SRC
        // is consumed by the presentation engine, which the present semaphore
        // orders, and no shader may touch it at all.
        const bool presented = target->final_layout() == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        record_image_transition(
            vk,
            cmd,
            final_image,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            target->final_layout(),
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            presented ? 0 : VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            presented ? VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            color_sr.base_mip,
            color_sr.mip_count,
            color_sr.layer_count,
            color_sr.base_layer);

        // A kept multisampled image retires too, since 0.25, because it
        // is then readable: target.multisampled_color[i] goes into a
        // sampler2DMS for a custom resolve. It goes to
        // SHADER_READ_ONLY_OPTIMAL rather than to final_layout(), and
        // that difference is the point — a swapchain's final layout is
        // PRESENT_SRC_KHR, and a multisampled image is never the thing
        // that gets presented.
        if (target->keep_samples() && target->color_resolve_image(i) != VK_NULL_HANDLE)
        {
            record_image_transition(
                vk,
                cmd,
                target->color_image(i),
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                color_sr.base_mip,
                color_sr.mip_count,
                color_sr.layer_count,
                color_sr.base_layer);
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
        VkImage final_depth = target->depth_resolve_image() != VK_NULL_HANDLE ? target->depth_resolve_image()
                                                                              : target->depth_image();
        record_image_transition(
            vk,
            cmd,
            final_depth,
            depth_layout,
            target->depth_final_layout(),
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            depth_aspect,
            depth_sr.base_mip,
            depth_sr.mip_count,
            depth_sr.layer_count,
            depth_sr.base_layer);

        // The multisampled depth follows the multisampled colour, for
        // the reason the colour comment gives. Symmetric on purpose:
        // "the colour samples are readable and the depth samples are
        // not" would be a second rule to remember, and it would show up
        // as a validation error rather than as a message.
        if (target->keep_samples() && target->depth_resolve_image() != VK_NULL_HANDLE)
        {
            record_image_transition(
                vk,
                cmd,
                target->depth_image(),
                depth_layout,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                depth_aspect,
                depth_sr.base_mip,
                depth_sr.mip_count,
                depth_sr.layer_count,
                depth_sr.base_layer);
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
    target->record_even_out(vk, cmd);
}

CommandBuffer& CommandBuffer::set_viewport(float x, float y, float width, float height)
{
    commands_.emplace_back(
        [x, y, width, height](VkCommandBuffer cmd, const FrameContext& frame)
        {
            VkViewport viewport{.x = x, .y = y, .width = width, .height = height, .minDepth = 0.0f, .maxDepth = 1.0f};
            frame.vk->vkCmdSetViewport(cmd, 0, 1, &viewport);
        });
    return *this;
}

CommandBuffer& CommandBuffer::set_scissor(std::int32_t x, std::int32_t y, std::uint32_t width, std::uint32_t height)
{
    commands_.emplace_back(
        [x, y, width, height](VkCommandBuffer cmd, const FrameContext& frame)
        {
            VkRect2D scissor{.offset = {x, y}, .extent = {width, height}};
            frame.vk->vkCmdSetScissor(cmd, 0, 1, &scissor);
        });
    return *this;
}

CommandBuffer& CommandBuffer::bind_pipeline(const std::shared_ptr<Pipeline>& pipeline)
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
    commands_.emplace_back([pipeline](VkCommandBuffer cmd, const FrameContext& frame)
                           { frame.vk->vkCmdBindPipeline(cmd, pipeline->bind_point(), pipeline->get()); });
    return *this;
}

CommandBuffer& CommandBuffer::bind_vertex_buffer(const std::shared_ptr<Buffer>& buffer, std::uint32_t binding)
{
    // The read truly happens at draw, but a barrier placed before the bind
    // is still before the draw — sound, and simpler than deferring it.
    track_use_(buffer, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT, false);
    record_buffer_use_(buffer);
    commands_.emplace_back(
        [buffer, binding](VkCommandBuffer cmd, const FrameContext& frame)
        {
            const std::array<VkBuffer, 1> vertexBuffers = {buffer->get()};
            const std::array<VkDeviceSize, 1> offsets = {0};
            frame.vk->vkCmdBindVertexBuffers(cmd, binding, 1, vertexBuffers.data(), offsets.data());
        });
    return *this;
}

std::expected<void, Error> CommandBuffer::bind_index_buffer(const std::shared_ptr<Buffer>& buffer)
{
    if (!buffer)
    {
        return std::unexpected(err_resource("bind_index_buffer: buffer is null"));
    }
    // STORAGE is accepted because it carries INDEX_BUFFER_BIT since 0.29, for
    // the reason buffer_usage_for gives. VERTEX and UNIFORM do not, and binding
    // one used to fail inside the layers instead of here.
    if (const BufferType type = buffer->buffer_type(); type != BufferType::INDEX && type != BufferType::STORAGE)
    {
        return std::unexpected(err_resource(
            std::format(
                "bind_index_buffer: indices must live in a BufferType.INDEX buffer, or in a "
                "BufferType.STORAGE one when a compute shader writes them. This is a {} buffer.",
                buffer_type_name(type))));
    }
    track_use_(buffer, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_ACCESS_INDEX_READ_BIT, false);
    record_buffer_use_(buffer);
    commands_.emplace_back(
        [buffer](VkCommandBuffer cmd, const FrameContext& frame)
        {
            // Derived from the buffer rather than hardcoded to UINT32: create_buffer
            // accepts UINT16 indices, which used to be read back at half count.
            frame.vk->vkCmdBindIndexBuffer(cmd, buffer->get(), 0, buffer->index_type());
        });
    return {};
}

CommandBuffer& CommandBuffer::draw(uint32_t vertexCount, uint32_t instances)
{
    track_draw_();
    commands_.emplace_back([vertexCount, instances](VkCommandBuffer cmd, const FrameContext& frame)
                           { frame.vk->vkCmdDraw(cmd, vertexCount, instances, 0, 0); });
    return *this;
}

CommandBuffer& CommandBuffer::draw_indexed(
    uint32_t indexCount,
    uint32_t firstIndex,
    int32_t vertexOffset,
    uint32_t instances)
{
    track_draw_();
    commands_.emplace_back(
        [indexCount, firstIndex, vertexOffset, instances](VkCommandBuffer cmd, const FrameContext& frame)
        { frame.vk->vkCmdDrawIndexed(cmd, indexCount, instances, firstIndex, vertexOffset, 0); });
    return *this;
}

CommandBuffer& CommandBuffer::dispatch(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ)
{
    track_dispatch_();
    commands_.emplace_back([groupCountX, groupCountY, groupCountZ](VkCommandBuffer cmd, const FrameContext& frame)
                           { frame.vk->vkCmdDispatch(cmd, groupCountX, groupCountY, groupCountZ); });
    return *this;
}

std::expected<void, Error> CommandBuffer::draw_indirect(
    std::shared_ptr<Buffer> buffer,
    VkDeviceSize offset,
    std::uint32_t count,
    std::shared_ptr<Buffer> count_buffer,
    VkDeviceSize count_offset,
    std::uint32_t stride)
{
    // 16 bytes: vertexCount, instanceCount, firstVertex, firstInstance.
    // stride=0 means "packed", which is the array a compute shader usually
    // writes. A larger stride lets the arguments be INTERLEAVED with per-draw
    // data — a material index, a bounding sphere — so one buffer carries both
    // instead of two buffers that have to stay index-aligned with each other.
    const VkDeviceSize draw_stride = stride ? stride : sizeof(VkDrawIndirectCommand);
    if (auto e = check_indirect_(buffer, offset, count, draw_stride, sizeof(VkDrawIndirectCommand), "draw_indirect");
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
    commands_.emplace_back(
        [buffer = std::move(buffer), offset, count, count_buffer = std::move(count_buffer), count_offset, draw_stride](
            VkCommandBuffer cmd, const FrameContext& frame)
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

std::expected<void, Error> CommandBuffer::draw_indexed_indirect(
    std::shared_ptr<Buffer> buffer,
    VkDeviceSize offset,
    std::uint32_t count,
    std::shared_ptr<Buffer> count_buffer,
    VkDeviceSize count_offset,
    std::uint32_t stride)
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
    commands_.emplace_back(
        [buffer = std::move(buffer), offset, count, count_buffer = std::move(count_buffer), count_offset, draw_stride](
            VkCommandBuffer cmd, const FrameContext& frame)
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

std::expected<void, Error> CommandBuffer::dispatch_indirect(std::shared_ptr<Buffer> buffer, VkDeviceSize offset)
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
    commands_.emplace_back([buffer = std::move(buffer), offset](VkCommandBuffer cmd, const FrameContext& frame)
                           { frame.vk->vkCmdDispatchIndirect(cmd, buffer->get(), offset); });
    return {};
}

std::expected<void, Error> CommandBuffer::barrier(std::shared_ptr<Buffer> buffer, Access src, Access dst)
{
    if (!buffer)
    {
        return std::unexpected(err_resource("barrier: buffer is null"));
    }
    const StageAccess s = to_vk(src, context_->all_shader_stages());
    const StageAccess d = to_vk(dst, context_->all_shader_stages());
    // Tell the fold what this barrier just established, for the reason the
    // image overload below does it: the caller expressed the dependency here,
    // so a later automatic use must not put its first-use floor on top of it,
    // and a neighbouring pass must order against the consumers named here.
    // Reported whatever auto_barriers_ says — a manual pass's barriers are
    // exactly what the passes around it need to know.
    note_buffer_state_(buffer, d.stages, d.access);
    commands_.emplace_back(
        [buffer = std::move(buffer), s, d](VkCommandBuffer cmd, const FrameContext& frame)
        {
            VkBufferMemoryBarrier barrier{
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = s.access,
                .dstAccessMask = d.access,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                // Resolved at replay, never captured: a DynamicBuffer has one
                // handle per frame in flight.
                .buffer = buffer->get(),
                .offset = 0,
                .size = VK_WHOLE_SIZE};
            frame.vk->vkCmdPipelineBarrier(cmd, s.stages, d.stages, 0, 0, nullptr, 1, &barrier, 0, nullptr);
        });
    return {};
}

std::expected<void, Error> CommandBuffer::barrier(std::shared_ptr<Image> image, Access src, Access dst)
{
    if (!image)
    {
        return std::unexpected(err_resource("barrier: image is null"));
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
    // Tell the fold the layout this leaves the image in, or a later automatic
    // use would transition it again from a stale one — a validation error plus
    // a useless barrier.
    note_image_state_(image, *new_layout, d.stages, d.access);
    commands_.emplace_back(
        [image = std::move(image), old = *old_layout, now = *new_layout, s, d](
            VkCommandBuffer cmd, const FrameContext& frame)
        {
            // All mips and all layers together: the fold holds one layout per
            // image, and a cube or an array is used as a whole.
            record_image_transition(
                *frame.vk,
                cmd,
                image->vk_image(),
                old,
                now,
                s.access,
                d.access,
                s.stages,
                d.stages,
                VK_IMAGE_ASPECT_COLOR_BIT,
                0,
                image->mip_levels(),
                image->barrier_layers(image->array_layers()));
        });
    return {};
}

std::expected<void, Error> CommandBuffer::generate_mipmaps(const std::shared_ptr<Image>& image, Access src)
{
    if (!image)
    {
        return std::unexpected(err_resource("generate_mipmaps: image is null"));
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
    commands_.emplace_back([image, layout = *src_layout, s](VkCommandBuffer cmd, const FrameContext& frame)
                           { image->record_generate_mipmaps(cmd, layout, s.stages, s.access); });
    // The image now rests in SHADER_READ_ONLY across every level; keep the
    // tracker in sync so a later automatic sample emits no extra transition.
    note_image_state_(
        image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, context_->all_shader_stages(), VK_ACCESS_SHADER_READ_BIT);
    return {};
}

std::expected<void, Error> CommandBuffer::copy_image(
    const std::shared_ptr<Image>& src,
    const std::shared_ptr<Image>& dst,
    Access src_access)
{
    if (!src || !dst)
    {
        return std::unexpected(err_resource("copy_image: image is null"));
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
    commands_.emplace_back([src, dst, layout = *src_layout](VkCommandBuffer cmd, const FrameContext& frame)
                           { record_image_copy(*frame.vk, cmd, *src, *dst, layout); });
    finish_image_transfer_(src, dst);
    return {};
}

std::expected<void, Error> CommandBuffer::blit_image(
    const std::shared_ptr<Image>& src,
    const std::shared_ptr<Image>& dst,
    Access src_access,
    VkFilter filter)
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
    commands_.emplace_back([src, dst, layout = *src_layout, filter](VkCommandBuffer cmd, const FrameContext& frame)
                           { record_image_blit(*frame.vk, cmd, *src, *dst, layout, filter); });
    finish_image_transfer_(src, dst);
    return {};
}

std::expected<void, Error> CommandBuffer::copy_buffer(
    std::shared_ptr<Buffer> src,
    std::shared_ptr<Buffer> dst,
    VkDeviceSize src_offset,
    VkDeviceSize dst_offset,
    VkDeviceSize size)
{
    if (!src || !dst)
    {
        return std::unexpected(err_resource("copy_buffer: buffer is null"));
    }
    const VkDeviceSize length = size != 0 ? size : bytes_after(src->size(), src_offset);
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
    commands_.emplace_back(
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

std::expected<void, Error> CommandBuffer::fill_buffer(
    std::shared_ptr<Buffer> buffer,
    std::uint32_t value,
    VkDeviceSize offset,
    VkDeviceSize size)
{
    if (!buffer)
    {
        return std::unexpected(err_resource("fill_buffer: buffer is null"));
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
    const VkDeviceSize length = size != 0 ? size : bytes_after(buffer->size(), offset);
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
    commands_.emplace_back(
        [buffer = std::move(buffer), value, offset, length](VkCommandBuffer cmd, const FrameContext& frame)
        { frame.vk->vkCmdFillBuffer(cmd, buffer->get(), offset, length, value); });
    return {};
}

std::expected<void, Error> CommandBuffer::clear_image(const std::shared_ptr<Image>& image, std::array<float, 4> color)
{
    if (!image)
    {
        return std::unexpected(err_resource("clear_image: image is null"));
    }
    if (format_info(image->format()).depth)
    {
        return std::unexpected(err_resource(
            "clear_image: a depth image is cleared by the pass that renders into it "
            "(cmd.rendering(target, clear_depth=...))"));
    }
    commands_.emplace_back([image, color](VkCommandBuffer cmd, const FrameContext& frame)
                           { record_image_clear(*frame.vk, cmd, *image, color); });
    image->mark_has_contents(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    note_image_state_(
        image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, context_->all_shader_stages(), VK_ACCESS_SHADER_READ_BIT);
    return {};
}

std::size_t CommandBuffer::start_timer()
{
    const std::size_t index = timer_count_++;
    record_timer_write_(static_cast<std::uint32_t>(2 * index), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
    return index;
}

void CommandBuffer::stop_timer(std::size_t index)
{
    record_timer_write_(static_cast<std::uint32_t>((2 * index) + 1), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
}

CommandBuffer::TimerReading CommandBuffer::read_timer(std::size_t index, std::uint64_t generation) const
{
    // First, because a stale handle's index may be out of range for the
    // recording that replaced it — that is still the stale handle's fault.
    if (generation != recording_generation_)
    {
        return {.status = QueryStatus::Superseded};
    }
    // timer_supported_ is only filled in at the first execute, so an unset
    // one means "nobody has asked the device yet", which is NotReady rather
    // than a verdict.
    if (timer_supported_.has_value() && !*timer_supported_)
    {
        return {.status = QueryStatus::Unsupported};
    }
    if (timer_pool_ == VK_NULL_HANDLE || index >= timer_count_)
    {
        return {.status = QueryStatus::NotReady};
    }
    std::array<std::uint64_t, 2> ts = {0, 0};
    if (context_->vk().vkGetQueryPoolResults(
            context_->device(),
            timer_pool_,
            static_cast<std::uint32_t>(2 * index),
            2,
            sizeof(ts),
            ts.data(),
            sizeof(std::uint64_t),
            VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
    {
        return {.status = QueryStatus::NotReady}; // VK_NOT_READY: submit not finished
    }
    const std::uint64_t mask = timer_valid_bits_ >= 64 ? ~std::uint64_t{0}
                                                       : ((std::uint64_t{1} << timer_valid_bits_) - 1);
    const std::uint64_t delta = (ts[1] - ts[0]) & mask;
    return {.status = QueryStatus::Ok, .ms = static_cast<double>(delta) * static_cast<double>(timer_period_) / 1.0e6};
}

CommandBuffer& CommandBuffer::begin_label(const std::string& name)
{
    commands_.emplace_back(
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

CommandBuffer& CommandBuffer::end_label()
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
    commands_.emplace_back(
        [](VkCommandBuffer cmd, const FrameContext&)
        {
            if (vkCmdEndDebugUtilsLabelEXT != nullptr)
            {
                vkCmdEndDebugUtilsLabelEXT(cmd);
            }
        });
    return *this;
}

std::expected<std::size_t, Error> CommandBuffer::start_occlusion_query()
{
    // "Inside a rendering scope" is now the PASS KIND, and the binding checks
    // it before this runs (VerbScope::Render). The recorder cannot check it
    // any more and must not try: a pass never sets in_rendering_, because the
    // graph opens the rendering scope in its executor rather than into the
    // pass's own command list — so the old guard refused every legal query.
    const std::size_t index = occlusion_count_++;
    commands_.emplace_back(
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

void CommandBuffer::stop_occlusion_query(std::size_t index)
{
    commands_.emplace_back(
        [this, index](VkCommandBuffer cmd, const FrameContext& frame)
        {
            if (occlusion_pool_ != VK_NULL_HANDLE)
            {
                frame.vk->vkCmdEndQuery(cmd, occlusion_pool_, static_cast<std::uint32_t>(index));
            }
        });
}

CommandBuffer::OcclusionReading CommandBuffer::read_occlusion_query(std::size_t index, std::uint64_t generation) const
{
    if (generation != recording_generation_)
    {
        return {.status = QueryStatus::Superseded};
    }
    if (occlusion_pool_ == VK_NULL_HANDLE || index >= occlusion_count_)
    {
        return {.status = QueryStatus::NotReady};
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
        return {.status = QueryStatus::NotReady}; // VK_NOT_READY: submit not finished
    }
    return {.status = QueryStatus::Ok, .samples = samples};
}

std::expected<void, Error> CommandBuffer::push_constants(uint32_t offset, uint32_t size, const void* data)
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

CommandBuffer& CommandBuffer::push_constants(
    const std::shared_ptr<Pipeline>& pipeline,
    uint32_t offset,
    uint32_t size,
    const void* data)
{
    std::vector<uint8_t> buffer(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + size);
    commands_.emplace_back(
        [pipeline, offset, size, buffer](VkCommandBuffer cmd, const FrameContext& frame)
        {
            frame.vk->vkCmdPushConstants(
                cmd, pipeline->layout(), pipeline->push_constant_stages(), offset, size, buffer.data());
        });
    return *this;
}

std::expected<void, Error> CommandBuffer::bind_descriptor_set(const std::shared_ptr<DescriptorSet>& descSet)
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
    bind_descriptor_set(descSet, pipeline, set_index);
    return {};
}

CommandBuffer& CommandBuffer::bind_descriptor_set(
    const std::shared_ptr<DescriptorSet>& descSet,
    const std::shared_ptr<Pipeline>& pipeline,
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
    commands_.emplace_back(
        [descSet, pipeline, setIndex](VkCommandBuffer cmd, const FrameContext& frame)
        {
            VkDescriptorSet set = descSet->get(frame.frame_index);
            frame.vk->vkCmdBindDescriptorSets(
                cmd, pipeline->bind_point(), pipeline->layout(), setIndex, 1, &set, 0, nullptr);
        });
    return *this;
}

void CommandBuffer::reset_query_pools(VkCommandBuffer vkCmd, const FrameContext& frame)
{
    // Timer query pool: created/grown here (the scope count is known once
    // recording is done) and reset before any command runs — timestamps
    // must be reset before they are written, and vkCmdResetQueryPool is
    // illegal inside a render pass, so the top of a replay is the one safe
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
    // BEGIN inside one, so the top of a replay is the only spot that serves
    // both halves.
    if (occlusion_count_ > 0)
    {
        ensure_occlusion_pool_(occlusion_count_);
        if (occlusion_pool_ != VK_NULL_HANDLE)
        {
            frame.vk->vkCmdResetQueryPool(vkCmd, occlusion_pool_, 0, occlusion_capacity_);
        }
    }
}

void CommandBuffer::finish_image_transfer_(const std::shared_ptr<Image>& src, const std::shared_ptr<Image>& dst)
{
    src->mark_has_contents(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    dst->mark_has_contents(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    for (const std::shared_ptr<Image>& image : {src, dst})
    {
        note_image_state_(
            image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, context_->all_shader_stages(), VK_ACCESS_SHADER_READ_BIT);
    }
}

void CommandBuffer::note_buffer_state_(
    const std::shared_ptr<Buffer>& buffer,
    VkPipelineStageFlags dst_stages,
    VkAccessFlags dst_access)
{
    event_sink_->push_back(
        {.kind = UseEvent::Kind::BufferNote,
         .buffer = buffer,
         .stages = dst_stages,
         .access = dst_access,
         .position = commands_.size()});
}

void CommandBuffer::note_image_state_(
    const std::shared_ptr<Image>& image,
    VkImageLayout layout,
    VkPipelineStageFlags dst_stages,
    VkAccessFlags dst_access)
{
    event_sink_->push_back(
        {.kind = UseEvent::Kind::ImageNote,
         .image = image,
         .layout = layout,
         .stages = dst_stages,
         .access = dst_access,
         .position = commands_.size()});
}

void CommandBuffer::record_timer_write_(std::uint32_t slot, VkPipelineStageFlagBits stage)
{
    commands_.emplace_back(
        [this, slot, stage](VkCommandBuffer cmd, const FrameContext& frame)
        {
            if (timer_pool_ != VK_NULL_HANDLE)
            {
                frame.vk->vkCmdWriteTimestamp(cmd, stage, timer_pool_, slot);
            }
        });
}

void CommandBuffer::ensure_timer_pool_(std::size_t needed)
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

void CommandBuffer::ensure_occlusion_pool_(std::size_t needed)
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

void CommandBuffer::record_buffer_use_(const std::shared_ptr<Buffer>& buffer)
{
    if (buffer && buffer->upload_serial() != 0)
    {
        used_buffers_.push_back(buffer);
    }
}

std::expected<void, Error> CommandBuffer::check_indirect_(
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
        return std::unexpected(err_resource(std::format("{}: offset must be a multiple of 4, got {}", what, offset)));
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
    const VkDeviceSize needed = (static_cast<VkDeviceSize>(count - 1) * stride) + argument_size;
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

std::expected<void, Error> CommandBuffer::check_count_buffer_(
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

void CommandBuffer::track_indirect_(const std::shared_ptr<Buffer>& buffer)
{
    track_use_(buffer, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT, false);
    record_buffer_use_(buffer);
}

void CommandBuffer::track_use_(
    const std::shared_ptr<Buffer>& buffer,
    VkPipelineStageFlags stages,
    VkAccessFlags access,
    bool writes)
{
    if (!auto_barriers_)
    {
        return;
    }
    // Only a STORAGE buffer carries VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, so it
    // is the only type a shader can write. The fold uses that to narrow its
    // first-use floor rather than to switch it off — every type can be written
    // by copy_buffer and fill_buffer.
    const bool shader_writable = buffer->buffer_type() == BufferType::STORAGE;
    event_sink_->push_back(
        {.kind = UseEvent::Kind::BufferUse,
         .buffer = buffer,
         .stages = stages,
         .access = access,
         .writes = writes,
         .shader_writable = shader_writable,
         .position = commands_.size()});
}

void CommandBuffer::track_image_use_(
    const std::shared_ptr<Image>& image,
    VkImageLayout layout,
    VkPipelineStageFlags stages,
    VkAccessFlags access,
    bool writes,
    bool only_if_tracked)
{
    if (!auto_barriers_)
    {
        return;
    }
    event_sink_->push_back(
        {.kind = UseEvent::Kind::ImageUse,
         .image = image,
         .layout = layout,
         .stages = stages,
         .access = access,
         .writes = writes,
         .only_if_tracked = only_if_tracked,
         .position = commands_.size()});
}

bool CommandBuffer::pipeline_writes_(
    const std::shared_ptr<Pipeline>& pipeline,
    std::uint32_t set,
    std::uint32_t binding)
{
    if (!pipeline)
    {
        return true;
    }
    return std::ranges::any_of(
        pipeline->shaders(),
        [&](const std::shared_ptr<ShaderModule>& s) { return s && s->reflection().writes(set, binding); });
}

void CommandBuffer::track_descriptor_uses_(
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
            else if (bi.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
            {
                // A sampled image only needs a barrier if something wrote it
                // earlier and it is still in GENERAL. An uploaded texture the
                // tracker never saw rests in SHADER_READ_ONLY and is left
                // untouched, so ordinary texturing pays nothing — and a
                // transition of it would DISCARD it, since the tracker's
                // starting layout is UNDEFINED.
                //
                // Who answers "was it written" differs by mode, which is what
                // the flag carries: inline mode asks its own tracker right
                // here, while in a pass the writer is a DIFFERENT pass, so the
                // question can only be settled when the graph folds them.
                track_image_use_(
                    bi.image,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    stages,
                    VK_ACCESS_SHADER_READ_BIT,
                    false,
                    /*only_if_tracked=*/true);
            }
        }
    }
}

void CommandBuffer::track_draw_()
{
    track_descriptor_uses_(bound_graphics_sets_, bound_graphics_pipeline_, context_->all_shader_stages());
}

void CommandBuffer::track_dispatch_()
{
    track_descriptor_uses_(bound_compute_sets_, bound_compute_pipeline_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}
