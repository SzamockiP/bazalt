#include "Graph.hpp"

// record_image_transition and Image::barrier_layers live here; the header
// reaches them transitively, this TU names the dependency it uses.
#include "Image.hpp"

#include <algorithm>
#include <format>

// ── Pass ────────────────────────────────────────────────────────────────────

void Pass::set_enabled(bool on)
{
    if (enabled_ == on)
    {
        return;
    }
    enabled_ = on;
    mark_graph_dirty();
}

void Pass::mark_graph_dirty()
{
    if (auto g = graph_.lock())
    {
        g->mark_dirty();
    }
}

std::expected<void, Error> Pass::guard(VerbScope scope, const char* verb) const
{
    if (removed_)
    {
        return std::unexpected(err_state(
            std::format("{}: this pass was removed from its graph (remove() or reset()). Add a new pass.", verb)));
    }
    if (sealed_)
    {
        return std::unexpected(err_state(
            std::format(
                "{}: this pass is sealed — its `with` block ended, or the graph was already "
                "submitted. Record into a new pass, or graph.reset() and rebuild.",
                verb)));
    }
    if (scope == VerbScope::Render && target_ == nullptr)
    {
        return std::unexpected(err_state(
            std::format(
                "{}: this pass has no render target, so it cannot draw. Create the pass with "
                "one: graph.add_pass(target, ...)",
                verb)));
    }
    if (scope == VerbScope::General && target_ != nullptr)
    {
        return std::unexpected(err_state(
            std::format(
                "{}: a render pass is one rendering scope, and Vulkan forbids this command "
                "inside one. Record it in a pass without a target: graph.add_pass()",
                verb)));
    }
    return {};
}

// ── Graph ───────────────────────────────────────────────────────────────────

std::expected<std::shared_ptr<Graph>, Error> Graph::create(Context& context)
{
    auto ctx = context.shared_from_this();
    auto graph = std::shared_ptr<Graph>(new Graph(ctx));
    graph->command_buffers_.resize(ctx->frames_in_flight(), VK_NULL_HANDLE);

    VkCommandBufferAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = ctx->command_pool(),
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = ctx->frames_in_flight()};

    if (auto e = check(
            ctx->vk().vkAllocateCommandBuffers(ctx->device(), &allocInfo, graph->command_buffers_.data()),
            "allocate graph command buffers",
            ErrorCode::Resource))
    {
        return std::unexpected(*e);
    }
    return graph;
}

Graph::~Graph()
{
    // Python handles to passes may outlive the graph; detach them so their
    // verbs report "removed" instead of touching freed state.
    for (auto& pass : passes_)
    {
        pass->removed_ = true;
    }
    if (context_ && !command_buffers_.empty())
    {
        context_->defer_destroy(
            [vk = &context_->vk(),
             device = context_->device(),
             pool = context_->command_pool(),
             buffers = std::move(command_buffers_)]
            { vk->vkFreeCommandBuffers(device, pool, static_cast<uint32_t>(buffers.size()), buffers.data()); });
    }
}

std::shared_ptr<Pass> Graph::add_pass(
    std::shared_ptr<RenderTarget> target,
    std::optional<std::vector<std::array<float, 4>>> clear_colors,
    float clear_depth,
    std::uint32_t clear_stencil,
    std::string name,
    QueueKind queue,
    std::optional<bool> auto_barriers)
{
    auto pass = std::make_shared<Pass>();
    pass->graph_ = weak_from_this();
    pass->target_ = std::move(target);
    pass->clear_colors_ = std::move(clear_colors);
    pass->clear_depth_ = clear_depth;
    pass->clear_stencil_ = clear_stencil;
    pass->name_ = std::move(name);
    pass->queue_ = queue;
    // The recorder owns no VkCommandBuffer of its own — the graph replays
    // every pass into its per-slot buffer — so creating one cannot fail.
    pass->recorder_ = CommandBuffer::create(*context_, auto_barriers).value();
    pass->recorder_->set_event_sink(&pass->events_);
    passes_.push_back(pass);
    dirty_ = true;
    return pass;
}

std::expected<void, Error> Graph::remove(const std::shared_ptr<Pass>& pass)
{
    const auto it = std::ranges::find(passes_, pass);
    if (it == passes_.end())
    {
        return std::unexpected(err_state(
            pass && pass->removed() ? "remove(): this pass was already removed"
                                    : "remove(): this pass does not belong to this graph"));
    }
    (*it)->removed_ = true;
    passes_.erase(it);
    dirty_ = true;
    return {};
}

void Graph::reset()
{
    for (auto& pass : passes_)
    {
        // begin() bumps the recorder's generation, so a Timer or an occlusion
        // handle made before the reset reports Superseded instead of reading
        // slots that now belong to nothing.
        pass->recorder_->begin();
        pass->removed_ = true;
    }
    passes_.clear();
    compiled_.clear();
    dirty_ = true;
}

std::expected<void, Error> Graph::claim_for_frame(std::uint64_t serial)
{
    if (recorded_serial_ == serial)
    {
        return std::unexpected(err_state(
            "This Graph was already submitted in the current frame. Each window needs its "
            "own Graph — one holds a single command buffer per frame slot, so replaying it "
            "twice would overwrite work still in flight."));
    }
    recorded_serial_ = serial;
    return {};
}

Graph::BarrierBatch& Graph::batch_at_(CompiledPass& cp, std::size_t position)
{
    if (cp.pass->is_render())
    {
        return cp.entry;
    }
    if (cp.mid.empty() || cp.mid.back().first != position)
    {
        cp.mid.emplace_back(position, BarrierBatch{});
    }
    return cp.mid.back().second;
}

void Graph::compile_()
{
    compiled_.clear();
    tracked_writes_ = false;

    // Sealing is what lets this fold trust every pass's use list; disabled
    // passes seal too — toggling one back on recompiles, not re-records.
    for (auto& pass : passes_)
    {
        pass->sealed_ = true;
    }

    // One tracker over every enabled pass, in add order. Inside the graph a
    // use has a visible predecessor, so it gets a precise edge; the tracker's
    // own first-use floors remain the answer for writers OUTSIDE the graph —
    // other graphs and previous frames — exactly the 0.24 argument, one level
    // up.
    ResourceTracker tracker;
    tracker.set_all_shader_stages(context_->all_shader_stages());

    // The enabled passes, and the look-ahead over them — both decided BEFORE
    // the fold, because the fold has to know whether a pass's exit transition
    // is going to run. It is a structural question (which passes share a target
    // and preserve it), so nothing here depends on what they recorded.
    for (auto& pass : passes_)
    {
        if (pass->enabled())
        {
            compiled_.push_back(CompiledPass{.pass = pass.get()});
        }
    }

    // The priced "a preserved second pass re-transitions the attachment" entry
    // being paid: when the immediately next enabled pass renders into the SAME
    // target and preserves, the attachment stays in its attachment layout
    // across the seam — no retire, no re-enter, one execution barrier.
    // Consecutive only: a pass in between could move the image, and then the
    // retire is what keeps the layouts honest.
    for (std::size_t k = 0; k + 1 < compiled_.size(); ++k)
    {
        const Pass& a = *compiled_[k].pass;
        const Pass& b = *compiled_[k + 1].pass;
        if (a.is_render() && b.is_render() && a.target() == b.target() && b.preserve())
        {
            compiled_[k].elide_exit = true;
            compiled_[k + 1].elide_entry = true;
        }
    }

    for (CompiledPass& cp : compiled_)
    {
        Pass* pass = cp.pass;

        // A render pass builds its entry transition from the RenderTarget, not
        // from any state: preserving means "come from final_layout()". That is
        // a lie whenever something else moved the image since — a compute pass
        // that wrote it as a storage image, say — and the transition is then
        // illegal from a layout the device is not in. The elided seam cannot
        // hit it (nothing runs between the two passes), so this corrects the
        // un-elided case, which is the only one where a predecessor exists.
        if (pass->is_render() && pass->preserve() && !cp.elide_entry)
        {
            correct_preserve_entry_(cp, tracker);
        }

        for (const UseEvent& e : pass->events())
        {
            switch (e.kind)
            {
                case UseEvent::Kind::BufferUse:
                {
                    tracked_writes_ |= e.writes;
                    if (auto b = tracker.use(e.buffer.get(), e.stages, e.access, e.writes, e.shader_writable))
                    {
                        batch_at_(cp, e.position).buffers.emplace_back(e.buffer, *b);
                    }
                    break;
                }
                case UseEvent::Kind::ImageUse:
                {
                    // The sampled-image rule: leave an image no pass in this
                    // graph has written alone. It rests in SHADER_READ_ONLY
                    // already, and the fold's starting layout is UNDEFINED, so
                    // a transition here would discard an uploaded texture.
                    if (e.only_if_tracked && !tracker.tracks(e.image.get()))
                    {
                        break;
                    }
                    tracked_writes_ |= e.writes;
                    if (auto b = tracker.use_image(e.image.get(), e.layout, e.stages, e.access, e.writes))
                    {
                        batch_at_(cp, e.position).images.emplace_back(e.image, *b);
                    }
                    break;
                }
                case UseEvent::Kind::BufferNote:
                    tracker.note_buffer_access(e.buffer.get(), e.stages, e.access);
                    break;
                case UseEvent::Kind::ImageNote:
                    tracker.note_image_layout(e.image.get(), e.layout, e.stages, e.access);
                    break;
            }
        }

        // What the pass just wrote as attachments. An attachment write is not a
        // descriptor use, so no UseEvent reports it, and without this the fold
        // would not know that the next pass sampling `target.color[0]` reads
        // what this one drew — the most ordinary thing a frame does, and a
        // READ_AFTER_WRITE hazard without a barrier.
        //
        // Skipped when the exit is elided, because then the image is still in
        // its attachment layout and the pass that follows renders into it
        // rather than reading it. The LAST pass of such a chain retires
        // normally and reports here.
        if (pass->is_render() && !cp.elide_exit)
        {
            note_attachment_writes_(*pass, tracker);
        }
    }

    dirty_ = false;
}

void Graph::correct_preserve_entry_(CompiledPass& cp, ResourceTracker& tracker)
{
    RenderTarget& rt = *cp.pass->target();
    const VkImageLayout wanted = rt.final_layout();
    for (const auto& image : rt.written_color_images())
    {
        const auto known = tracker.layout_of(image.get());
        if (!known || *known == wanted)
        {
            continue;
        }
        cp.entry.images.emplace_back(
            image,
            ResourceTracker::ImageBarrier{
                .old_layout = *known,
                .new_layout = wanted,
                .src_stages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                .dst_stages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                .src_access = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                .dst_access = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT});
        // The pass's own entry transition now starts where it says it does.
        tracker.note_image_layout(
            image.get(), wanted, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT);
    }
}

void Graph::note_attachment_writes_(const Pass& pass, ResourceTracker& tracker)
{
    RenderTarget& rt = *pass.target();
    // What the pass's own exit transition already made available: it retires
    // the attachments naming the fragment shader as the reader (see
    // record_render_pass_transitions_out), so a fragment read needs nothing
    // more and any other stage gets a barrier from the fold.
    const VkPipelineStageFlags retired_to = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    for (const auto& image : rt.written_color_images())
    {
        tracker.note_image_write(
            image.get(),
            rt.final_layout(),
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            retired_to,
            VK_ACCESS_SHADER_READ_BIT);
    }
    if (const auto& depth = rt.written_depth_image())
    {
        tracker.note_image_write(
            depth.get(),
            rt.depth_final_layout(),
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            retired_to,
            VK_ACCESS_SHADER_READ_BIT);
    }
}

void Graph::BarrierBatch::record(VkCommandBuffer cmd, const FrameContext& frame) const
{
    VkPipelineStageFlags src = 0;
    VkPipelineStageFlags dst = 0;

    std::vector<VkBufferMemoryBarrier> bufs;
    bufs.reserve(buffers.size());
    for (const auto& [buffer, b] : buffers)
    {
        src |= b.src_stages;
        dst |= b.dst_stages;
        bufs.push_back(
            {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
             .pNext = nullptr,
             .srcAccessMask = b.src_access,
             .dstAccessMask = b.dst_access,
             .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .buffer = buffer->get(),
             .offset = 0,
             .size = VK_WHOLE_SIZE});
    }

    std::vector<VkImageMemoryBarrier> imgs;
    imgs.reserve(images.size());
    for (const auto& [image, b] : images)
    {
        src |= b.src_stages;
        dst |= b.dst_stages;
        // All mips and all layers: the fold holds one layout per image. The
        // aspect comes from the FORMAT rather than being COLOR — a depth
        // image reaches this path as soon as a pass samples the depth another
        // pass rendered, and naming COLOR on it is a barrier the layers
        // reject outright.
        imgs.push_back(
            {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
             .pNext = nullptr,
             .srcAccessMask = b.src_access,
             .dstAccessMask = b.dst_access,
             .oldLayout = b.old_layout,
             .newLayout = b.new_layout,
             .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .image = image->vk_image(),
             .subresourceRange = {
                 .aspectMask = image->aspect(),
                 .baseMipLevel = 0,
                 .levelCount = image->mip_levels(),
                 .baseArrayLayer = 0,
                 .layerCount = image->barrier_layers(image->array_layers())}});
    }

    if (bufs.empty() && imgs.empty())
    {
        return;
    }
    frame.vk->vkCmdPipelineBarrier(
        cmd,
        src != 0 ? src : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        dst != 0 ? dst : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        0,
        0,
        nullptr,
        static_cast<uint32_t>(bufs.size()),
        bufs.empty() ? nullptr : bufs.data(),
        static_cast<uint32_t>(imgs.size()),
        imgs.empty() ? nullptr : imgs.data());
}

namespace
{

    // The elided-entry seam: the attachment never left its attachment layout, so
    // the pair of transitions becomes one execution-and-memory barrier —
    // attachment writes of the previous pass before this pass's load and write.
    // No resolve images to cover: MSAA cannot preserve, so it cannot elide.
    void record_inplace_attachment_barrier(const VolkDeviceTable& vk, VkCommandBuffer cmd, RenderTarget& rt)
    {
        const RenderTarget::Subresource color_sr = rt.color_subresource();
        for (uint32_t i = 0; i < rt.color_count(); ++i)
        {
            record_image_transition(
                vk,
                cmd,
                rt.color_image(i),
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT,
                color_sr.base_mip,
                color_sr.mip_count,
                color_sr.layer_count,
                color_sr.base_layer);
        }
        if (rt.depth_image() != VK_NULL_HANDLE)
        {
            const VkImageAspectFlags depth_aspect = aspect_mask_for(rt.depth_format());
            const VkImageLayout depth_layout = has_stencil(rt.depth_format())
                                                   ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                                                   : VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            const RenderTarget::Subresource depth_sr = rt.depth_subresource();
            record_image_transition(
                vk,
                cmd,
                rt.depth_image(),
                depth_layout,
                depth_layout,
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                depth_aspect,
                depth_sr.base_mip,
                depth_sr.mip_count,
                depth_sr.layer_count,
                depth_sr.base_layer);
        }
    }

} // namespace

void Graph::execute(VkCommandBuffer vkCmd, const FrameContext& frame)
{
    if (dirty_)
    {
        compile_();
    }

    // Query-pool resets are illegal inside a render pass, so every pass's
    // pools reset here, before anything opens — the same spot execute() gives
    // them on the inline path.
    for (auto& cp : compiled_)
    {
        cp.pass->recorder().reset_query_pools(vkCmd, frame);
    }

    // Replay wrap-around, verbatim from the inline recorder: in-graph barriers
    // order uses within one replay, but the same graph ran last frame and may
    // still be in flight — its trailing reads/writes race with this replay's
    // first write. Emitted only when some enabled pass writes a tracked
    // resource: read-only graphs race with nothing.
    if (tracked_writes_)
    {
        VkMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_UNIFORM_READ_BIT |
                             VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT |
                             VK_ACCESS_INDIRECT_COMMAND_READ_BIT};
        const VkPipelineStageFlags stages = context_->all_shader_stages() | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
                                            VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
        frame.vk->vkCmdPipelineBarrier(vkCmd, stages, stages, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    }

    for (const CompiledPass& cp : compiled_)
    {
        Pass& pass = *cp.pass;
        CommandBuffer& rec = pass.recorder();
        cp.entry.record(vkCmd, frame);
        if (pass.is_render())
        {
            RenderTarget& rt = *pass.target();
            if (cp.elide_entry)
            {
                record_inplace_attachment_barrier(*frame.vk, vkCmd, rt);
            }
            else
            {
                record_render_pass_transitions_in(*frame.vk, vkCmd, rt, pass.preserve());
            }
            record_render_pass_begin(*frame.vk, vkCmd, rt, pass.clear_colors_, pass.clear_depth_, pass.clear_stencil_);
            rec.replay_range(vkCmd, frame, 0, rec.command_count());
            record_render_pass_end(*frame.vk, vkCmd);
            if (!cp.elide_exit)
            {
                record_render_pass_transitions_out(*frame.vk, vkCmd, rt);
            }
        }
        else
        {
            std::size_t at = 0;
            for (const auto& [position, batch] : cp.mid)
            {
                rec.replay_range(vkCmd, frame, at, position);
                batch.record(vkCmd, frame);
                at = position;
            }
            rec.replay_range(vkCmd, frame, at, rec.command_count());
        }
    }
}
