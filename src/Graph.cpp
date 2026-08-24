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

std::expected<void, Error> Pass::guard(VerbScope scope, const char* verb, QueueNeeds needs) const
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
    if (needs == QueueNeeds::Graphics && queue_ != QueueKind::Graphics)
    {
        return std::unexpected(err_state(
            std::format(
                "{}: a pass on Queue.{} cannot blit — vkCmdBlitImage needs a graphics "
                "queue. Put this pass on Queue.GRAPHICS, or use copy_image for a copy that "
                "does not resize.",
                verb,
                queue_name(queue_))));
    }
    if (needs == QueueNeeds::Shader && queue_ == QueueKind::Transfer)
    {
        return std::unexpected(err_state(
            std::format(
                "{}: a pass on Queue.TRANSFER cannot run shaders — a transfer family runs "
                "copies only. Put this pass on Queue.GRAPHICS or Queue.COMPUTE.",
                verb)));
    }
    return {};
}

// ── Graph ───────────────────────────────────────────────────────────────────

std::expected<std::shared_ptr<Graph>, Error> Graph::create(Context& context)
{
    // Nothing is allocated here since 0.29: how many command buffers a graph
    // needs follows from its batches, and a fresh graph has no passes yet.
    // compile() allocates, and grows the rows when a rebuild adds a batch.
    return std::shared_ptr<Graph>(new Graph(context.shared_from_this()));
}

Graph::~Graph()
{
    // Python handles to passes may outlive the graph; detach them so their
    // verbs report "removed" instead of touching freed state.
    for (auto& pass : passes_)
    {
        pass->removed_ = true;
    }
    if (!context_)
    {
        return;
    }
    // One deferred free per queue: a command buffer goes back to the pool it
    // came from, and the two runtimes have one pool each.
    for (std::size_t i = 0; i < kQueueCount; ++i)
    {
        if (command_buffers_[i].empty())
        {
            continue;
        }
        context_->defer_destroy(
            [vk = &context_->vk(),
             device = context_->device(),
             pool = context_->command_pool(static_cast<QueueKind>(i)),
             buffers = std::move(command_buffers_[i])]
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
    // The recorder needs to know which queue will replay it: the timer pool
    // asks that family whether its timestamps are usable.
    pass->recorder_->set_queue(queue);
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
    batches_.clear();
    dirty_ = true;
}

std::expected<void, Error> Graph::claim_for_frame(std::uint64_t serial)
{
    if (recorded_serial_ == serial)
    {
        return std::unexpected(err_state(
            "This Graph was already submitted in the current frame. Each window needs its "
            "own Graph — one holds a single command buffer per batch per frame slot, so "
            "replaying it twice would overwrite work still in flight."));
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

std::expected<void, Error> Graph::compile()
{
    if (!dirty_)
    {
        return {};
    }
    return compile_();
}

std::expected<void, Error> Graph::ensure_command_buffers_()
{
    std::array<std::size_t, kQueueCount> needed{};
    for (const Batch& batch : batches_)
    {
        std::size_t& count = needed[queue_index(batch.queue)];
        count = (std::max)(count, batch.ordinal + 1);
    }

    const std::uint32_t frames = context_->frames_in_flight();
    for (std::size_t i = 0; i < kQueueCount; ++i)
    {
        std::vector<VkCommandBuffer>& row = command_buffers_[i];
        const std::size_t want = needed[i] * frames;
        if (row.size() >= want)
        {
            continue;
        }
        const auto extra = static_cast<std::uint32_t>(want - row.size());
        const std::size_t at = row.size();
        row.resize(want, VK_NULL_HANDLE);
        VkCommandBufferAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr,
            .commandPool = context_->command_pool(static_cast<QueueKind>(i)),
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = extra};
        if (auto e = check(
                context_->vk().vkAllocateCommandBuffers(context_->device(), &allocInfo, row.data() + at),
                "allocate graph command buffers",
                ErrorCode::Resource))
        {
            row.resize(at);
            return std::unexpected(*e);
        }
    }
    return {};
}

namespace
{
    // Human names for the report. Core-1.0 bits only, which is the tracker's
    // whole vocabulary; an unknown bit prints as hex so nothing is hidden.
    std::string stage_names(VkPipelineStageFlags stages)
    {
        if (stages == 0)
        {
            return "NONE";
        }
        if (stages == ~VkPipelineStageFlags{0})
        {
            return "ALL_COMMANDS";
        }
        static constexpr std::array<std::pair<VkPipelineStageFlags, const char*>, 17> kNames = {{
            {VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, "TOP_OF_PIPE"},
            {VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, "DRAW_INDIRECT"},
            {VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, "VERTEX_INPUT"},
            {VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, "VERTEX_SHADER"},
            {VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT, "TESS_CONTROL"},
            {VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT, "TESS_EVALUATION"},
            {VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT, "GEOMETRY_SHADER"},
            {VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, "FRAGMENT_SHADER"},
            {VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, "EARLY_FRAGMENT_TESTS"},
            {VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, "LATE_FRAGMENT_TESTS"},
            {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, "COLOR_ATTACHMENT_OUTPUT"},
            {VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, "COMPUTE_SHADER"},
            {VK_PIPELINE_STAGE_TRANSFER_BIT, "TRANSFER"},
            {VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, "BOTTOM_OF_PIPE"},
            {VK_PIPELINE_STAGE_HOST_BIT, "HOST"},
            {VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, "ALL_GRAPHICS"},
            {VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, "ALL_COMMANDS"},
        }};
        std::string out;
        VkPipelineStageFlags rest = stages;
        for (const auto& [bit, name] : kNames)
        {
            if ((rest & bit) != 0)
            {
                out += out.empty() ? name : std::string("|") + name;
                rest &= ~bit;
            }
        }
        if (rest != 0)
        {
            out += std::format("{}0x{:x}", out.empty() ? "" : "|", rest);
        }
        return out;
    }

    std::string access_names(VkAccessFlags access)
    {
        if (access == 0)
        {
            return "NONE";
        }
        if (access == ~VkAccessFlags{0})
        {
            return "ANY";
        }
        static constexpr std::array<std::pair<VkAccessFlags, const char*>, 17> kNames = {{
            {VK_ACCESS_INDIRECT_COMMAND_READ_BIT, "INDIRECT_COMMAND_READ"},
            {VK_ACCESS_INDEX_READ_BIT, "INDEX_READ"},
            {VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT, "VERTEX_ATTRIBUTE_READ"},
            {VK_ACCESS_UNIFORM_READ_BIT, "UNIFORM_READ"},
            {VK_ACCESS_INPUT_ATTACHMENT_READ_BIT, "INPUT_ATTACHMENT_READ"},
            {VK_ACCESS_SHADER_READ_BIT, "SHADER_READ"},
            {VK_ACCESS_SHADER_WRITE_BIT, "SHADER_WRITE"},
            {VK_ACCESS_COLOR_ATTACHMENT_READ_BIT, "COLOR_ATTACHMENT_READ"},
            {VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, "COLOR_ATTACHMENT_WRITE"},
            {VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT, "DEPTH_STENCIL_READ"},
            {VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, "DEPTH_STENCIL_WRITE"},
            {VK_ACCESS_TRANSFER_READ_BIT, "TRANSFER_READ"},
            {VK_ACCESS_TRANSFER_WRITE_BIT, "TRANSFER_WRITE"},
            {VK_ACCESS_HOST_READ_BIT, "HOST_READ"},
            {VK_ACCESS_HOST_WRITE_BIT, "HOST_WRITE"},
            {VK_ACCESS_MEMORY_READ_BIT, "MEMORY_READ"},
            {VK_ACCESS_MEMORY_WRITE_BIT, "MEMORY_WRITE"},
        }};
        std::string out;
        VkAccessFlags rest = access;
        for (const auto& [bit, name] : kNames)
        {
            if ((rest & bit) != 0)
            {
                out += out.empty() ? name : std::string("|") + name;
                rest &= ~bit;
            }
        }
        if (rest != 0)
        {
            out += std::format("{}0x{:x}", out.empty() ? "" : "|", rest);
        }
        return out;
    }

    const char* layout_name(VkImageLayout layout)
    {
        switch (layout)
        {
            case VK_IMAGE_LAYOUT_UNDEFINED:
                return "UNDEFINED";
            case VK_IMAGE_LAYOUT_GENERAL:
                return "GENERAL";
            case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
                return "COLOR_ATTACHMENT_OPTIMAL";
            case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
                return "DEPTH_STENCIL_ATTACHMENT_OPTIMAL";
            case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
                return "DEPTH_STENCIL_READ_ONLY_OPTIMAL";
            case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
                return "SHADER_READ_ONLY_OPTIMAL";
            case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
                return "TRANSFER_SRC_OPTIMAL";
            case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
                return "TRANSFER_DST_OPTIMAL";
            case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
                return "PRESENT_SRC_KHR";
            default:
                return "OTHER";
        }
    }

    std::string describe(const Image& image)
    {
        if (!image.name().empty())
        {
            return std::format("image \"{}\"", image.name());
        }
        return std::format("image {}x{} {}", image.width(), image.height(), format_name(image.format()));
    }

    std::string describe(const Buffer& buffer)
    {
        if (!buffer.name().empty())
        {
            return std::format("buffer \"{}\"", buffer.name());
        }
        return std::format("buffer {} B", buffer.size());
    }

    const char* access_member_name(Access access)
    {
        switch (access)
        {
            case Access::SHADER_READ:
                return "SHADER_READ";
            case Access::SHADER_WRITE:
                return "SHADER_WRITE";
            case Access::VERTEX_READ:
                return "VERTEX_READ";
            case Access::INDEX_READ:
                return "INDEX_READ";
            case Access::UNIFORM_READ:
                return "UNIFORM_READ";
            case Access::INDIRECT_READ:
                return "INDIRECT_READ";
            case Access::TRANSFER_WRITE:
                return "TRANSFER_WRITE";
            case Access::TRANSFER_READ:
                return "TRANSFER_READ";
        }
        return "SHADER_READ";
    }

    // The bz.Access member that names a (stages, access, layout) triple, for
    // the lint's fix line. Exact subset first; then, for the descriptor walk's
    // READ|WRITE, the member that carries the write bit. nullopt when no
    // member fits — a render-pass access, a depth aspect — and the fix line
    // says auto_barriers=True instead of guessing.
    std::optional<Access> access_for(
        VkPipelineStageFlags stages,
        VkAccessFlags access,
        std::optional<VkImageLayout> layout,
        VkPipelineStageFlags all_shader_stages)
    {
        constexpr std::array kMembers = {
            Access::SHADER_READ,
            Access::SHADER_WRITE,
            Access::VERTEX_READ,
            Access::INDEX_READ,
            Access::UNIFORM_READ,
            Access::INDIRECT_READ,
            Access::TRANSFER_WRITE,
            Access::TRANSFER_READ};
        for (const Access member : kMembers)
        {
            const StageAccess sa = to_vk(member, all_shader_stages);
            if (layout.has_value() && image_layout_for(member) != layout)
            {
                continue;
            }
            if ((stages & ~sa.stages) == 0 && (access & ~sa.access) == 0 && access != 0)
            {
                return member;
            }
        }
        for (const Access member : kMembers)
        {
            const StageAccess sa = to_vk(member, all_shader_stages);
            if (layout.has_value() && image_layout_for(member) != layout)
            {
                continue;
            }
            if ((stages & ~sa.stages) == 0 && (sa.access & access & VK_ACCESS_SHADER_WRITE_BIT) != 0)
            {
                return member;
            }
        }
        return std::nullopt;
    }

    std::string pass_label(const Pass& pass, std::size_t index)
    {
        return pass.name().empty() ? std::format("pass [{}]", index) : std::format("pass \"{}\"", pass.name());
    }
} // namespace

template <typename State>
void Graph::record_explain_(
    std::size_t pass,
    const std::shared_ptr<Buffer>& buffer,
    const std::shared_ptr<Image>& image,
    const State* prev,
    bool had_prev,
    QueueKind queue,
    std::size_t waits_before,
    const std::vector<std::size_t>& waits,
    const std::optional<ResourceTracker::ImageBarrier>& barrier)
{
    static_cast<void>(queue);
    // Who produced the dependency, read from the state as it was BEFORE the
    // use: the last writer when there is one, else the last reader (a WAR
    // edge). kNoPass means nothing in this graph — the floor's territory.
    const auto producer = [&]() -> std::pair<std::size_t, bool>
    {
        if (prev != nullptr && prev->write_pass != ResourceTracker::kNoPass)
        {
            return {prev->write_pass, true};
        }
        if (prev != nullptr)
        {
            for (std::size_t q = 0; q < kQueueCount; ++q)
            {
                if (prev->read_pass[q] != ResourceTracker::kNoPass)
                {
                    return {prev->read_pass[q], false};
                }
            }
        }
        return {ResourceTracker::kNoPass, true};
    }();

    if (waits.size() > waits_before)
    {
        explain_.push_back(
            {.kind = ExplainEntry::Kind::Wait,
             .pass = pass,
             .buffer = buffer,
             .image = image,
             .producer = producer.first,
             .producer_wrote = producer.second,
             .b = barrier.value_or(ResourceTracker::ImageBarrier{})});
    }
    if (barrier)
    {
        const bool floor = !had_prev || producer.first == ResourceTracker::kNoPass;
        explain_.push_back(
            {.kind = floor ? ExplainEntry::Kind::Floor : ExplainEntry::Kind::Barrier,
             .pass = pass,
             .buffer = buffer,
             .image = image,
             .producer = floor ? ResourceTracker::kNoPass : producer.first,
             .producer_wrote = producer.second,
             .b = *barrier});
    }
}

std::expected<std::string, Error> Graph::explain()
{
    if (dirty_)
    {
        if (auto r = compile_(); !r)
        {
            return std::unexpected(r.error());
        }
    }

    std::string out = std::format(
        "graph: {} pass{}, {} batch{}\n",
        compiled_.size(),
        compiled_.size() == 1 ? "" : "es",
        batches_.size(),
        batches_.size() == 1 ? "" : "es");

    for (std::size_t i = 0; i < compiled_.size(); ++i)
    {
        const CompiledPass& cp = compiled_[i];
        const Pass& pass = *cp.pass;
        const Batch& batch = batches_[cp.batch];

        std::string waits_text;
        for (const std::size_t j : batch.waits)
        {
            waits_text +=
                std::format("{}batch {} ({})", waits_text.empty() ? "" : ", ", j, queue_name(batches_[j].queue));
        }
        std::string header = std::format(
            "[{}] {}  queue {}  batch {} (#{} on {})  waits: {}",
            i,
            pass.name().empty() ? std::format("pass") : std::format("\"{}\"", pass.name()),
            queue_name(pass.queue()),
            cp.batch,
            batch.ordinal,
            queue_name(batch.queue),
            waits_text.empty() ? "none" : waits_text);
        if (!pass.enabled())
        {
            header += "  (disabled)";
        }
        if (pass.is_render())
        {
            const VkExtent2D extent = pass.target()->extent();
            header += std::format("  render target {}x{}", extent.width, extent.height);
        }
        out += header + "\n";

        for (const ExplainEntry& e : explain_)
        {
            if (e.pass != i)
            {
                continue;
            }
            const std::string what = [&]
            {
                if (e.image)
                {
                    return describe(*e.image);
                }
                if (e.buffer)
                {
                    return describe(*e.buffer);
                }
                return std::string("attachment");
            }();
            const std::string producer = e.producer == ResourceTracker::kNoPass
                                             ? std::string("outside this graph or a previous replay (first use)")
                                             : std::format(
                                                   "{}pass [{}] {}",
                                                   e.producer_wrote ? "" : "reads in ",
                                                   e.producer,
                                                   compiled_[e.producer].pass->name().empty()
                                                       ? std::string()
                                                       : std::format("\"{}\"", compiled_[e.producer].pass->name()));
            switch (e.kind)
            {
                case ExplainEntry::Kind::Barrier:
                case ExplainEntry::Kind::Floor:
                    if (e.image)
                    {
                        out += std::format(
                            "  barrier  {}  {} -> {}\n",
                            what,
                            layout_name(e.b.old_layout),
                            layout_name(e.b.new_layout));
                    }
                    else
                    {
                        out += std::format("  barrier  {}\n", what);
                    }
                    out += std::format(
                        "           {} / {} -> {} / {}\n",
                        stage_names(e.b.src_stages),
                        access_names(e.b.src_access),
                        stage_names(e.b.dst_stages),
                        access_names(e.b.dst_access));
                    out += std::format("           producer: {}\n", producer);
                    break;
                case ExplainEntry::Kind::Wait:
                    out += std::format("  wait     {}  cross-queue: timeline wait, producer: {}\n", what, producer);
                    break;
                case ExplainEntry::Kind::Attachment:
                    out += std::format(
                        "  attachment {}  {} -> {}\n",
                        e.note,
                        layout_name(e.b.old_layout),
                        layout_name(e.b.new_layout));
                    break;
                case ExplainEntry::Kind::Retire:
                    out += std::format(
                        "  retire     {}  {} -> {}\n",
                        e.note,
                        layout_name(e.b.old_layout),
                        layout_name(e.b.new_layout));
                    break;
                case ExplainEntry::Kind::Elided:
                    out += std::format(
                        "  elided     {}  stays {} (the seam with the neighbouring pass is one "
                        "execution barrier)\n",
                        e.note,
                        layout_name(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL));
                    break;
                case ExplainEntry::Kind::Unordered:
                    out += std::format(
                        "  UNORDERED (manual)  {}  no barrier in this pass covers the use, producer: {}\n",
                        what,
                        producer);
                    break;
            }
        }
    }
    return out;
}

void Graph::warn_manual_hazard_(
    std::size_t pass_index,
    const UseEvent& e,
    const ResourceTracker::BufferState* buffer_state,
    const ResourceTracker::ImageState* image_state,
    ResourceTracker::Peek peek)
{
    const Pass& pass = *compiled_[pass_index].pass;
    const bool is_image = e.image != nullptr;
    const std::string what = is_image ? describe(*e.image) : describe(*e.buffer);
    const VkPipelineStageFlags all_shaders = context_->all_shader_stages();

    // The previous state and its producer, from whichever state struct holds
    // them (the two only differ by the layout field).
    bool prev_wrote = false;
    VkPipelineStageFlags prev_stages = 0;
    VkAccessFlags prev_access = 0;
    std::size_t producer = ResourceTracker::kNoPass;
    std::optional<VkImageLayout> prev_layout;
    const auto read_state = [&](const auto* st)
    {
        if (st == nullptr)
        {
            return;
        }
        prev_wrote = st->written;
        prev_stages = st->written ? st->write_stages : st->read_stages;
        prev_access = st->written ? st->write_access : st->read_access;
        producer = st->write_pass;
        if (producer == ResourceTracker::kNoPass)
        {
            for (std::size_t q = 0; q < kQueueCount; ++q)
            {
                if (st->read_pass[q] != ResourceTracker::kNoPass)
                {
                    producer = st->read_pass[q];
                    break;
                }
            }
        }
    };
    read_state(buffer_state);
    read_state(image_state);
    if (image_state != nullptr)
    {
        prev_layout = image_state->layout;
    }

    const std::string in_pass = producer == ResourceTracker::kNoPass
                                    ? std::string("outside this pass")
                                    : std::format("in {}", pass_label(*compiled_[producer].pass, producer));

    // First multi-line bazalt-authored message, kept on purpose: a hazard has
    // four facts, and one line of them does not read.
    std::string text = std::format("Graph: possible hazard in {}\n", pass_label(pass, pass_index));
    text += std::format("  resource: {}\n", what);
    if (is_image)
    {
        text += std::format(
            "  previous: {}, {} / {}, {}\n",
            layout_name(prev_layout.value_or(VK_IMAGE_LAYOUT_UNDEFINED)),
            stage_names(prev_stages),
            access_names(prev_access),
            in_pass);
        text += std::format(
            "  requested: {}, {} / {}\n", layout_name(e.layout), stage_names(e.stages), access_names(e.access));
    }
    else
    {
        text += std::format("  previous: {} / {}, {}\n", stage_names(prev_stages), access_names(prev_access), in_pass);
        text += std::format("  requested: {} / {}\n", stage_names(e.stages), access_names(e.access));
    }
    if (peek.cross_queue)
    {
        text += "  cross-queue: the producer ran on another queue, and a pipeline barrier cannot reach "
                "it — add the pass with auto_barriers=True\n";
    }
    else
    {
        const auto src = access_for(prev_stages, prev_access, std::nullopt, all_shaders);
        const auto dst = access_for(e.stages, e.access, is_image ? std::optional(e.layout) : std::nullopt, all_shaders);
        const std::string handle = [&]
        {
            const std::string& named = is_image ? e.image->name() : e.buffer->name();
            if (!named.empty())
            {
                return named;
            }
            return std::string(is_image ? "<the image>" : "<the buffer>");
        }();
        if (src.has_value() && dst.has_value())
        {
            text += std::format(
                "  fix: p.barrier({}, src=bz.Access.{}, dst=bz.Access.{}) in {}, or add the pass "
                "with auto_barriers=True\n",
                handle,
                access_member_name(*src),
                access_member_name(*dst),
                pass_label(pass, pass_index));
        }
        else
        {
            text += "  fix: no bz.Access names this use, so add the pass with auto_barriers=True\n";
        }
    }

    if (auto logger = context_->logger())
    {
        logger->log(Severity::Warning, Source::General, text);
    }
    explain_.push_back(
        {.kind = ExplainEntry::Kind::Unordered,
         .pass = pass_index,
         .buffer = e.buffer,
         .image = e.image,
         .producer = producer,
         .producer_wrote = prev_wrote});
}

std::expected<void, Error> Graph::compile_()
{
    compiled_.clear();
    batches_.clear();
    explain_.clear();
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

    // The batches: maximal runs of consecutive enabled passes on one queue, in
    // add order. A graph with no enabled pass gets one empty graphics batch,
    // so a submit always has something to signal and a window always has a
    // command buffer to present.
    std::array<std::size_t, kQueueCount> per_queue{};
    for (std::size_t i = 0; i < compiled_.size(); ++i)
    {
        const QueueKind queue = compiled_[i].pass->queue();
        if (batches_.empty() || batches_.back().queue != queue)
        {
            batches_.push_back(
                Batch{.queue = queue, .first = i, .last = i + 1, .ordinal = per_queue[queue_index(queue)]++});
        }
        else
        {
            batches_.back().last = i + 1;
        }
        compiled_[i].batch = batches_.size() - 1;
    }
    if (batches_.empty())
    {
        batches_.push_back(Batch{.queue = QueueKind::Graphics, .first = 0, .last = 0, .ordinal = 0});
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

    // One warning per (pass, resource) per compile: a manual pass repeats a
    // use per draw, and the second line would add nothing.
    std::vector<std::pair<std::size_t, const void*>> warned;
    const auto warn_once = [&](std::size_t pass_idx, const void* resource)
    {
        const auto key = std::make_pair(pass_idx, resource);
        if (std::ranges::find(warned, key) != warned.end())
        {
            return false;
        }
        warned.push_back(key);
        return true;
    };

    for (CompiledPass& cp : compiled_)
    {
        Pass* pass = cp.pass;
        Batch& batch = batches_[cp.batch];
        // Everything the fold learns about this pass is attributed to its
        // batch: a dependency inside one batch is a pipeline barrier, and one
        // that crosses batches on different queues is a semaphore wait, which
        // only the submit can emit.
        const auto pass_index = static_cast<std::size_t>(&cp - compiled_.data());
        tracker.set_batch(cp.batch, batch.queue, pass_index);
        std::vector<std::size_t>& waits = batch.waits;

        // A render pass's attachments are transitioned by the RenderTarget
        // rather than through the tracker, so a pass on the other queue that
        // sampled one has no other way to be ordered against the drawing that
        // is about to overwrite it.
        if (pass->is_render())
        {
            RenderTarget& rt = *pass->target();
            for (const auto& image : rt.written_color_images())
            {
                const auto* prev = tracker.image_state(image.get());
                const std::size_t before = waits.size();
                tracker.cross_queue_touches(image.get(), waits);
                record_explain_(pass_index, nullptr, image, prev, prev != nullptr, batch.queue, before, waits, {});
            }
            if (const auto& depth = rt.written_depth_image())
            {
                const auto* prev = tracker.image_state(depth.get());
                const std::size_t before = waits.size();
                tracker.cross_queue_touches(depth.get(), waits);
                record_explain_(pass_index, nullptr, depth, prev, prev != nullptr, batch.queue, before, waits, {});
            }
        }

        // A render pass builds its entry transition from the RenderTarget, not
        // from any state: preserving means "come from final_layout()". That is
        // a lie whenever something else moved the image since — a compute pass
        // that wrote it as a storage image, say — and the transition is then
        // illegal from a layout the device is not in. The elided seam cannot
        // hit it (nothing runs between the two passes), so this corrects the
        // un-elided case, which is the only one where a predecessor exists.
        if (pass->is_render() && pass->preserve() && !cp.elide_entry)
        {
            correct_preserve_entry_(cp, tracker, waits);
        }

        for (const UseEvent& e : pass->events())
        {
            switch (e.kind)
            {
                case UseEvent::Kind::BufferUse:
                {
                    if (e.manual)
                    {
                        // A manual pass tells the fold only what its notes
                        // say; a use is peeked, never committed — committing
                        // would silently change how the automatic neighbours
                        // order against this pass. The warning changes
                        // nothing the GPU does (rule 2: the pass may know
                        // better — an address-written buffer is invisible
                        // here).
                        const auto peek =
                            tracker.peek_use(e.buffer.get(), e.stages, e.access, e.writes, e.shader_writable);
                        if ((peek.barrier || peek.cross_queue) && warn_once(pass_index, e.buffer.get()))
                        {
                            warn_manual_hazard_(pass_index, e, tracker.state(e.buffer.get()), nullptr, peek);
                        }
                        break;
                    }
                    tracked_writes_ |= e.writes;
                    const auto* prev_state = tracker.state(e.buffer.get());
                    const auto prev = prev_state != nullptr ? std::optional(*prev_state) : std::nullopt;
                    const std::size_t before = waits.size();
                    auto b = tracker.use(e.buffer.get(), e.stages, e.access, e.writes, e.shader_writable, waits);
                    if (b)
                    {
                        batch_at_(cp, e.position).buffers.emplace_back(e.buffer, *b);
                    }
                    std::optional<ResourceTracker::ImageBarrier> as_image;
                    if (b)
                    {
                        as_image = ResourceTracker::ImageBarrier{
                            .old_layout = VK_IMAGE_LAYOUT_UNDEFINED,
                            .new_layout = VK_IMAGE_LAYOUT_UNDEFINED,
                            .src_stages = b->src_stages,
                            .dst_stages = b->dst_stages,
                            .src_access = b->src_access,
                            .dst_access = b->dst_access};
                    }
                    record_explain_(
                        pass_index,
                        e.buffer,
                        nullptr,
                        prev ? &*prev : nullptr,
                        prev.has_value(),
                        batch.queue,
                        before,
                        waits,
                        as_image);
                    break;
                }
                case UseEvent::Kind::ImageUse:
                {
                    // The sampled-image rule: leave an image no pass in this
                    // graph has written alone. It rests in SHADER_READ_ONLY
                    // already, and the fold's starting layout is UNDEFINED, so
                    // a transition here would discard an uploaded texture.
                    //
                    // Recorded as a completed read rather than dropped, though:
                    // it emits nothing (the layout it names is the layout the
                    // image is in), and without it a later pass that RENDERS
                    // into this image has no way to know somebody was reading
                    // it — which across queues is a write-after-read nothing
                    // else would catch.
                    // Per range since 0.30, and load-bearing for a pyramid
                    // pass: the storage write of mip N makes the IMAGE
                    // tracked, so a whole-image question would transition the
                    // uploaded mip 0 this pass also samples — a discard of
                    // the source, by its own pass.
                    if (e.only_if_tracked && !tracker.tracks(e.image.get(), e.range))
                    {
                        tracker.note_image_read(e.image.get(), e.layout, e.stages, e.access, e.range, e.layers, e.mips);
                        break;
                    }
                    if (e.manual)
                    {
                        const auto peek = tracker.peek_use_image(
                            e.image.get(), e.layout, e.stages, e.access, e.writes, e.range, e.layers, e.mips);
                        if ((peek.barrier || peek.cross_queue) && warn_once(pass_index, e.image.get()))
                        {
                            warn_manual_hazard_(pass_index, e, nullptr, tracker.image_state(e.image.get()), peek);
                        }
                        break;
                    }
                    tracked_writes_ |= e.writes;
                    {
                        const auto* prev_state = tracker.image_state(e.image.get());
                        const auto prev = prev_state != nullptr ? std::optional(*prev_state) : std::nullopt;
                        const std::size_t before = waits.size();
                        auto barriers = tracker.use_image(
                            e.image.get(), e.layout, e.stages, e.access, e.writes, waits, e.range, e.layers, e.mips);
                        for (const auto& b : barriers)
                        {
                            batch_at_(cp, e.position).images.emplace_back(e.image, b);
                        }
                        record_explain_(
                            pass_index,
                            nullptr,
                            e.image,
                            prev ? &*prev : nullptr,
                            prev.has_value(),
                            batch.queue,
                            before,
                            waits,
                            barriers.empty() ? std::nullopt : std::optional(barriers.front()));
                    }
                    break;
                }
                case UseEvent::Kind::BufferNote:
                {
                    const auto* prev_state = tracker.state(e.buffer.get());
                    const auto prev = prev_state != nullptr ? std::optional(*prev_state) : std::nullopt;
                    const std::size_t before = waits.size();
                    tracker.note_buffer_access(e.buffer.get(), e.stages, e.access, waits);
                    record_explain_(
                        pass_index,
                        e.buffer,
                        nullptr,
                        prev ? &*prev : nullptr,
                        prev.has_value(),
                        batch.queue,
                        before,
                        waits,
                        {});
                    break;
                }
                case UseEvent::Kind::ImageNote:
                {
                    const auto* prev_state = tracker.image_state(e.image.get());
                    const auto prev = prev_state != nullptr ? std::optional(*prev_state) : std::nullopt;
                    const std::size_t before = waits.size();
                    tracker.note_image_layout(e.image.get(), e.layout, e.stages, e.access, &waits);
                    record_explain_(
                        pass_index,
                        nullptr,
                        e.image,
                        prev ? &*prev : nullptr,
                        prev.has_value(),
                        batch.queue,
                        before,
                        waits,
                        {});
                    break;
                }
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

        // The attachment rows: descriptive, because the executor builds these
        // transitions from the TARGET rather than through the tracker. The
        // report says which layout each attachment enters and retires to, and
        // where the look-ahead elided the seam.
        if (pass->is_render())
        {
            RenderTarget& rt = *pass->target();
            const bool clears = !pass->preserve();
            std::size_t index = 0;
            for (const auto& image : rt.written_color_images())
            {
                ExplainEntry entry{
                    .kind = cp.elide_entry ? ExplainEntry::Kind::Elided : ExplainEntry::Kind::Attachment,
                    .pass = pass_index,
                    .image = image,
                    .b =
                        {.old_layout = clears ? VK_IMAGE_LAYOUT_UNDEFINED : rt.final_layout(),
                         .new_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
                    .note = std::format("color[{}]{}", index, clears ? " (clear)" : " (preserve)")};
                explain_.push_back(std::move(entry));
                ExplainEntry exit{
                    .kind = cp.elide_exit ? ExplainEntry::Kind::Elided : ExplainEntry::Kind::Retire,
                    .pass = pass_index,
                    .image = image,
                    .b = {.old_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, .new_layout = rt.final_layout()},
                    .note = std::format("color[{}]", index)};
                explain_.push_back(std::move(exit));
                ++index;
            }
        }
    }

    // A batch may have collected the same producer several times, and the
    // submit compares each entry against its own index.
    for (Batch& batch : batches_)
    {
        std::ranges::sort(batch.waits);
        const auto duplicates = std::ranges::unique(batch.waits);
        batch.waits.erase(duplicates.begin(), duplicates.end());
    }

    // The layouts the replay really leaves behind. Every verb marks its image
    // when it is RECORDED, and record order is not execution order: a
    // clear_image on an image a later pass binds as storage marked
    // SHADER_READ_ONLY, the draw then moved the image to GENERAL and told
    // nobody, and the next read()'s barrier sourced from a layout the image
    // had already left (DESIGN.md, debt 7). The fold is the only party that
    // knows what every pass did to every subresource, so it is the one that
    // answers.
    final_layouts_.clear();
    for (const auto& [image, states] : tracker.image_states())
    {
        // UNDEFINED means the fold reached the image without ever naming a
        // layout for it. Writing that back would make the next barrier
        // DISCARD the image, which is worse than the stale mark this replaces.
        const auto keep = [this, image = image](VkImageLayout layout, ImageRange range)
        {
            if (layout != VK_IMAGE_LAYOUT_UNDEFINED)
            {
                final_layouts_.push_back({.image = image, .layout = layout, .range = range});
            }
        };
        if (states.split.empty())
        {
            keep(states.whole.layout, {});
            continue;
        }
        for (std::uint32_t layer = 0; layer < states.layers; ++layer)
        {
            for (std::uint32_t mip = 0; mip < states.mips; ++mip)
            {
                keep(
                    states.split[(layer * states.mips) + mip].layout,
                    {.base_layer = layer, .layer_count = 1, .base_mip = mip, .mip_count = 1});
            }
        }
    }

    // Only once the command buffers really exist: a failed allocation used to
    // leave the graph clean AND short a row, so the next submit indexed past
    // the end of it rather than retrying the compile.
    if (auto r = ensure_command_buffers_(); !r)
    {
        return r;
    }
    dirty_ = false;
    return {};
}

void Graph::apply_final_layouts()
{
    for (const auto& [image, layout, range] : final_layouts_)
    {
        if (range.layer_count == 0)
        {
            image->set_layout(layout);
        }
        else
        {
            image->set_layout(layout, range.base_layer, range.layer_count, range.base_mip, range.mip_count);
        }
    }
}

void Graph::correct_preserve_entry_(CompiledPass& cp, ResourceTracker& tracker, std::vector<std::size_t>& waits)
{
    RenderTarget& rt = *cp.pass->target();
    const VkImageLayout wanted = rt.final_layout();
    const auto pass_index = static_cast<std::size_t>(&cp - compiled_.data());
    for (const auto& image : rt.written_color_images())
    {
        if (!tracker.layouts_differ(image.get(), wanted))
        {
            continue;
        }
        const auto* prev_state = tracker.image_state(image.get());
        const auto prev = prev_state != nullptr ? std::optional(*prev_state) : std::nullopt;
        const std::size_t before = waits.size();
        // Through the tracker rather than hand-built since 0.29: the old
        // ALL_COMMANDS source scope covers everything on THIS queue and nothing
        // on the other, and a compute pass that moved the attachment is exactly
        // the case this correction exists for. use_image answers both halves —
        // a barrier for the local predecessor, a wait for the remote one.
        // A whole-image use: it reconciles a split by emitting one barrier
        // per differing subresource, and the note below then collapses it.
        auto barriers = tracker.use_image(
            image.get(),
            wanted,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            /*writes=*/true,
            waits,
            {},
            image->array_layers(),
            image->mip_levels());
        for (const auto& b : barriers)
        {
            cp.entry.images.emplace_back(image, b);
        }
        record_explain_(
            pass_index,
            nullptr,
            image,
            prev ? &*prev : nullptr,
            prev.has_value(),
            cp.pass->queue(),
            before,
            waits,
            barriers.empty() ? std::nullopt : std::optional(barriers.front()));
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
    // Every mask is narrowed to what the replaying queue family supports (see
    // narrow_src/narrow_dst). On a graphics family the legal set is everything,
    // so nothing changes; on a compute-only one the graphics stages go and the
    // dependency they expressed is carried by this batch's semaphore wait.
    const VkPipelineStageFlags legal = frame.legal_stages;
    VkPipelineStageFlags src = 0;
    VkPipelineStageFlags dst = 0;

    std::vector<VkBufferMemoryBarrier> bufs;
    bufs.reserve(buffers.size());
    for (const auto& [buffer, b] : buffers)
    {
        const StageAccess s = narrow_src({.stages = b.src_stages, .access = b.src_access}, legal);
        const StageAccess d = narrow_dst({.stages = b.dst_stages, .access = b.dst_access}, legal);
        src |= s.stages;
        dst |= d.stages;
        bufs.push_back(
            {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
             .pNext = nullptr,
             .srcAccessMask = s.access,
             .dstAccessMask = d.access,
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
        const StageAccess s = narrow_src({.stages = b.src_stages, .access = b.src_access}, legal);
        const StageAccess d = narrow_dst({.stages = b.dst_stages, .access = b.dst_access}, legal);
        src |= s.stages;
        dst |= d.stages;
        // The subresources the fold decided (0.30): the whole image for an
        // ordinary use, one (layer, mip) where a descriptor narrowed it. The
        // aspect comes from the FORMAT rather than being COLOR — a depth
        // image reaches this path as soon as a pass samples the depth another
        // pass rendered, and naming COLOR on it is a barrier the layers
        // reject outright.
        const bool whole = b.range.layer_count == 0;
        imgs.push_back(
            {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
             .pNext = nullptr,
             .srcAccessMask = s.access,
             .dstAccessMask = d.access,
             .oldLayout = b.old_layout,
             .newLayout = b.new_layout,
             .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .image = image->vk_image(),
             .subresourceRange = {
                 .aspectMask = image->aspect(),
                 .baseMipLevel = whole ? 0 : b.range.base_mip,
                 .levelCount = whole ? image->mip_levels() : b.range.mip_count,
                 .baseArrayLayer = whole ? 0 : b.range.base_layer,
                 .layerCount = image->barrier_layers(whole ? image->array_layers() : b.range.layer_count)}});
    }

    if (bufs.empty() && imgs.empty())
    {
        return;
    }
    // BOTTOM_OF_PIPE for an empty destination, not TOP: an empty second scope
    // is the one that waits for nothing, and TOP there would block everything.
    frame.vk->vkCmdPipelineBarrier(
        cmd,
        src != 0 ? src : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        dst != 0 ? dst : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
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

void Graph::execute_batch(const Batch& batch, VkCommandBuffer vkCmd, const FrameContext& frame)
{
    // Query-pool resets are illegal inside a render pass, so this batch's
    // pools reset here, before anything opens. Per batch rather than per
    // graph: a reset has to run on the queue that writes the queries, and each
    // batch is its own submit.
    for (std::size_t i = batch.first; i < batch.last; ++i)
    {
        compiled_[i].pass->recorder().reset_query_pools(vkCmd, frame);
    }

    // Replay wrap-around, verbatim from the inline recorder: in-graph barriers
    // order uses within one replay, but the same graph ran last frame and may
    // still be in flight — its trailing reads/writes race with this replay's
    // first write. Emitted only when some enabled pass writes a tracked
    // resource: read-only graphs race with nothing.
    if (tracked_writes_)
    {
        const VkPipelineStageFlags wide = context_->all_shader_stages() | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
                                          VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
        const StageAccess s = narrow_src({.stages = wide, .access = VK_ACCESS_SHADER_WRITE_BIT}, frame.legal_stages);
        const StageAccess d = narrow_dst(
            {.stages = wide,
             .access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_UNIFORM_READ_BIT |
                       VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT |
                       VK_ACCESS_INDIRECT_COMMAND_READ_BIT},
            frame.legal_stages);
        VkMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = s.access,
            .dstAccessMask = d.access};
        frame.vk->vkCmdPipelineBarrier(vkCmd, s.stages, d.stages, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    }

    for (std::size_t i = batch.first; i < batch.last; ++i)
    {
        const CompiledPass& cp = compiled_[i];
        Pass& pass = *cp.pass;
        CommandBuffer& rec = pass.recorder();
        // Every named pass replays inside a debug label — entry barriers
        // included, so a validation message about one names the pass that
        // needed it (0.30). The pass's own begin_label/end_label pairs replay
        // strictly inside this one, and end_label drops unbalanced ends, so
        // the outer label cannot be closed from within.
        const bool labelled = !pass.name().empty();
        if (labelled)
        {
            begin_debug_label(vkCmd, pass.name());
        }
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
            for (const auto& [position, barriers] : cp.mid)
            {
                rec.replay_range(vkCmd, frame, at, position);
                barriers.record(vkCmd, frame);
                at = position;
            }
            rec.replay_range(vkCmd, frame, at, rec.command_count());
        }
        if (labelled)
        {
            end_debug_label(vkCmd);
        }
    }
}
