#pragma once
#include <volk.h>
#include <array>
#include <algorithm>
#include <cstddef>
#include <optional>
#include <unordered_map>
#include <vector>

#include "Queue.hpp"

class Buffer;
class Image;

// What a recorded command does to a buffer, named from the caller's point of
// view. This is the vocabulary of cmd.barrier() in manual mode; the automatic
// tracker speaks raw (stage, access) pairs directly for better precision.
enum class Access
{
    SHADER_READ,
    SHADER_WRITE,
    VERTEX_READ,
    INDEX_READ,
    UNIFORM_READ,
    // The draw or dispatch arguments themselves, read by the command processor
    // rather than by a shader. Appended, not inserted, because pybind enum values
    // are API. One entry covers both draw and dispatch: DRAW_INDIRECT is the stage
    // the spec names for indirect *and* dispatch-indirect data.
    INDIRECT_READ,
    // What cmd.fill_buffer, cmd.copy_buffer and a staging upload do (0.26).
    //
    // The automatic tracker has always known about these — it puts a
    // TRANSFER_WRITE floor under the first reader of a buffer, a few dozen
    // lines below. The manual vocabulary could not SAY them, which did not
    // matter while every reader the tracker missed was rare. buffer.address
    // made it matter: a buffer reached by a pointer is invisible to the
    // tracker, so zeroing a counter with fill_buffer and reading it from a
    // dispatch was a hazard nobody could express — not automatically, because
    // the read is unseen, and not by hand, because the write had no name.
    TRANSFER_WRITE,
    TRANSFER_READ
};

struct StageAccess
{
    VkPipelineStageFlags stages;
    VkAccessFlags access;
};

// A rectangle of subresources: which layers and mips a use touches (0.30, for
// set_image(layer=, mip=)). layer_count == 0 means "the whole image", which is
// what every whole-image call site says by default.
struct ImageRange
{
    std::uint32_t base_layer = 0;
    std::uint32_t layer_count = 0;
    std::uint32_t base_mip = 0;
    std::uint32_t mip_count = 0;
};

// Core-1.0 pairs on purpose: the whole codebase rides vkCmdPipelineBarrier,
// not synchronization2, and mixing models would be a second way to say the
// same thing.
//
// `all_shader_stages` is the caller's Context::all_shader_stages() and has no
// default. There used to be a constexpr kAllShaderStages here, and it could not
// survive tessellation: wide enough to cover a tessellation read, it is illegal
// on a Context without the feature (VUID-vkCmdPipelineBarrier-srcStageMask-04090
// /-04091); narrow enough to always be legal, it drops that read. Deleting the
// constant rather than defaulting the parameter is deliberate — a call site that
// was missed is a compile error, which is the only referee a mask has.
inline constexpr StageAccess to_vk(Access access, VkPipelineStageFlags all_shader_stages)
{
    switch (access)
    {
        case Access::SHADER_READ:
            return {all_shader_stages, VK_ACCESS_SHADER_READ_BIT};
        case Access::SHADER_WRITE:
            return {all_shader_stages, VK_ACCESS_SHADER_WRITE_BIT};
        case Access::VERTEX_READ:
            return {VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT};
        case Access::INDEX_READ:
            return {VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_ACCESS_INDEX_READ_BIT};
        case Access::UNIFORM_READ:
            return {all_shader_stages, VK_ACCESS_UNIFORM_READ_BIT};
        case Access::INDIRECT_READ:
            return {VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT};
        case Access::TRANSFER_WRITE:
            return {VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT};
        case Access::TRANSFER_READ:
            return {VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT};
    }
    // Not std::unreachable(): pybind enums accept arbitrary ints.
    return {all_shader_stages, VK_ACCESS_SHADER_READ_BIT};
}

// Which access bits a stage mask is allowed to carry (the spec's "Supported
// access types" table, in the shape the narrowing below needs). Dropping a
// stage from a mask must drop the accesses only that stage could perform, or
// the barrier names an access no stage in its mask supports — which is a
// validation error rather than a conservative one.
inline constexpr VkAccessFlags access_supported_by(VkPipelineStageFlags stages)
{
    // TOP_OF_PIPE and BOTTOM_OF_PIPE support no access at all; ALL_COMMANDS
    // supports every one, and the two ends of the pipe are what an emptied
    // mask falls back to, so an empty result there is correct.
    VkAccessFlags access = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    if (stages & VK_PIPELINE_STAGE_ALL_COMMANDS_BIT)
    {
        return ~VkAccessFlags{0};
    }
    if (stages & VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT)
    {
        access |= VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    }
    if (stages & VK_PIPELINE_STAGE_VERTEX_INPUT_BIT)
    {
        access |= VK_ACCESS_INDEX_READ_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
    }
    constexpr VkPipelineStageFlags kShaderStages =
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT |
        VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT |
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    if (stages & kShaderStages)
    {
        access |= VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    }
    if (stages & VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)
    {
        access |= VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;
    }
    if (stages & VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)
    {
        access |= VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    }
    if (stages & (VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT))
    {
        access |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    }
    if (stages & VK_PIPELINE_STAGE_TRANSFER_BIT)
    {
        access |= VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    }
    if (stages & VK_PIPELINE_STAGE_HOST_BIT)
    {
        access |= VK_ACCESS_HOST_READ_BIT | VK_ACCESS_HOST_WRITE_BIT;
    }
    return access;
}

// Narrow one half of a barrier to what the replaying queue family supports.
//
// Every stage bit in a vkCmdPipelineBarrier must be one the pool's family
// supports, and a compute-only family supports a handful. The tracker computes
// in graphics vocabulary because the fold does not know which queue will replay
// a pass, so the narrowing happens where the barrier is EMITTED.
//
// An emptied mask does not mean "no dependency": it means the dependency is
// carried by something else — the semaphore wait that got this batch its work
// in the first place. TOP_OF_PIPE as a source and BOTTOM_OF_PIPE as a
// destination are the empty scopes that say so, and both carry no access.
inline constexpr StageAccess narrow(StageAccess sa, VkPipelineStageFlags legal, VkPipelineStageFlags when_empty)
{
    const VkPipelineStageFlags stages = sa.stages & legal;
    if (stages == 0)
    {
        return {when_empty, 0};
    }
    return {stages, sa.access & access_supported_by(stages)};
}

inline constexpr StageAccess narrow_src(StageAccess sa, VkPipelineStageFlags legal)
{
    return narrow(sa, legal, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
}

inline constexpr StageAccess narrow_dst(StageAccess sa, VkPipelineStageFlags legal)
{
    return narrow(sa, legal, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
}

// The image layout each shader access implies: a storage image written by a
// shader lives in GENERAL, a sampled image in SHADER_READ_ONLY. Only these two
// shader accesses name an image layout; the rest are buffer-only. Backs the
// manual cmd.barrier(image, ...) — the caller names accesses, not raw layouts.
inline std::optional<VkImageLayout> image_layout_for(Access access)
{
    switch (access)
    {
        case Access::SHADER_WRITE:
            return VK_IMAGE_LAYOUT_GENERAL;
        case Access::SHADER_READ:
            return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        default:
            return std::nullopt;
    }
}

// Computes buffer barriers at RECORD time. Deferred recording fixes the usage
// sequence constructively — record once, replay every submit — so a barrier
// computed here is correct for every replay and nothing runs per frame.
//
// Keys on Buffer* (object identity), never VkBuffer: a DynamicBuffer has one
// handle per frame in flight, but it is one resource with one usage history.
//
// Scope: buffers when this was written in 0.6, images since, and graphics SSBO
// writes since 0.19 gave it SPIR-V reflection to ask. What it still cannot see
// is a write from OUTSIDE the recording it is tracking — another CommandBuffer,
// or the previous replay of this one — because record-time state is per
// recording by construction. That is what the first-use floors answer, one on
// each of use() and use_image(): the first touch of a resource assumes the
// worst about who wrote it last, and every touch after names its real
// predecessor.
class ResourceTracker
{
public:
    // The mask the first-use floor below synchronizes against. Set once from the
    // owning Context when the CommandBuffer is created, not per recording: it is a
    // property of the device, not of what is being recorded.
    void set_all_shader_stages(VkPipelineStageFlags stages)
    {
        all_shader_stages_ = stages;
    }

    // No batch at all — the value every "who touched this" field starts at.
    static constexpr std::size_t kNoBatch = static_cast<std::size_t>(-1);
    // No pass — the provenance twin of kNoBatch (0.30, for graph.explain()).
    static constexpr std::size_t kNoPass = static_cast<std::size_t>(-1);

    // Which batch the fold is currently folding, on which queue, and which
    // pass (an index the caller understands — the graph's compiled_ order).
    // The caller sets it before each pass; everything below records it, so a
    // use whose producer ran on the other queue can name the batch to wait
    // for, and explain() can name the pass that produced a dependency.
    void set_batch(std::size_t batch, QueueKind queue, std::size_t pass = kNoPass)
    {
        batch_ = batch;
        queue_ = queue;
        queue_slot_ = queue_index(queue);
        pass_ = pass;
    }

    // The hazard state, public since 0.30 so graph.explain() and the manual
    // lint can read what the fold knew before a use. The maps stay private;
    // state()/image_state() hand out const pointers.
    struct BufferState
    {
        bool written = false;
        VkPipelineStageFlags write_stages = 0;
        VkAccessFlags write_access = 0;
        // Which batch wrote it last, and on which queue. kNoBatch until
        // something in this fold writes it.
        std::size_t write_batch = kNoBatch;
        QueueKind write_queue = QueueKind::Graphics;
        // Stages/accesses already synchronized against the last write, PER
        // QUEUE. One shared pair would let a read on the compute queue skip
        // its wait because a read on the graphics queue had already made the
        // write visible there — visibility established on one queue says
        // nothing about the other.
        std::array<VkPipelineStageFlags, kQueueCount> visible_stages{};
        std::array<VkAccessFlags, kQueueCount> visible_access{};
        // The latest batch that read it since the last write, per queue. Only
        // the latest is needed: a timeline signal covers everything submitted
        // earlier on its queue, and a wait is ">=".
        std::array<std::size_t, kQueueCount> read_batch = per_queue(kNoBatch);
        // Reads since the last write (what a future write must wait for).
        VkPipelineStageFlags read_stages = 0;
        VkAccessFlags read_access = 0;
        // Provenance for explain() and the lint: which pass (in the caller's
        // compiled order) wrote last, and which read last per queue. Carried
        // beside the batch fields rather than derived from them, because a
        // batch holds several passes.
        std::size_t write_pass = kNoPass;
        std::array<std::size_t, kQueueCount> read_pass = per_queue(kNoPass);
    };

    // BufferState plus the layout the recording has left the image in so far.
    // Defaulted comparison: the split/collapse asks "does every subresource
    // agree" and a field-by-field answer is exactly that question.
    struct ImageState
    {
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        bool written = false;
        VkPipelineStageFlags write_stages = 0;
        VkAccessFlags write_access = 0;
        std::size_t write_batch = kNoBatch;
        QueueKind write_queue = QueueKind::Graphics;
        std::array<VkPipelineStageFlags, kQueueCount> visible_stages{};
        std::array<VkAccessFlags, kQueueCount> visible_access{};
        std::array<std::size_t, kQueueCount> read_batch = per_queue(kNoBatch);
        VkPipelineStageFlags read_stages = 0;
        VkAccessFlags read_access = 0;
        std::size_t write_pass = kNoPass;
        std::array<std::size_t, kQueueCount> read_pass = per_queue(kNoPass);
        bool operator==(const ImageState&) const = default;
    };

    // One state while the image is uniform, a per-(layer, mip) split while it
    // is not — the SubresourceLayouts shape applied to hazard state (0.30).
    struct ImageStates
    {
        ImageState whole;
        // Layer-major, layers x mips entries; empty = uniform.
        std::vector<ImageState> split;
        std::uint32_t layers = 1;
        std::uint32_t mips = 1;
    };

    static bool touched_(const ImageState& st)
    {
        return st.written || std::ranges::any_of(st.read_batch, [](std::size_t b) { return b != kNoBatch; });
    }

    // The manual-pass lint's question (0.30): would the automatic path emit a
    // barrier or a cross-queue wait for this use that the pass's own notes do
    // not already cover? Answered on a COPY, so it cannot drift from use() /
    // use_image(); a manual use is peeked, never committed. An untracked
    // resource answers no — the first-use floor covers writers outside the
    // graph, and a warning with no producer to name is noise.
    //
    // The covered test is applied to writes too, which the automatic path
    // does not do (it always barriers a WAW): the lint asks whether the
    // caller ordered the use, not whether the fold would emit its
    // belt-and-braces barrier.
    // ponytail: each peek copies both state maps; manual passes are rare and
    // a compile runs once — split use() into compute/commit halves if a
    // profile ever cares.
    struct Peek
    {
        bool barrier = false;
        bool cross_queue = false;
    };
    Peek peek_use(Buffer* buffer, VkPipelineStageFlags stages, VkAccessFlags access, bool writes, bool shader_writable)
        const
    {
        const auto it = states_.find(buffer);
        if (it == states_.end() || covered_(it->second, stages, access))
        {
            return {};
        }
        ResourceTracker copy(*this);
        std::vector<std::size_t> waits;
        const auto b = copy.use(buffer, stages, access, writes, shader_writable, waits);
        return {.barrier = b.has_value(), .cross_queue = !waits.empty()};
    }
    Peek peek_use_image(
        Image* image,
        VkImageLayout layout,
        VkPipelineStageFlags stages,
        VkAccessFlags access,
        bool writes,
        const ImageRange& range = {},
        std::uint32_t layers = 1,
        std::uint32_t mips = 1) const
    {
        const auto it = image_states_.find(image);
        if (it == image_states_.end())
        {
            return {};
        }
        if (it->second.split.empty() && it->second.whole.layout == layout && covered_(it->second.whole, stages, access))
        {
            return {};
        }
        ResourceTracker copy(*this);
        std::vector<std::size_t> waits;
        const auto barriers = copy.use_image(image, layout, stages, access, writes, waits, range, layers, mips);
        return {.barrier = !barriers.empty(), .cross_queue = !waits.empty()};
    }

    struct Barrier
    {
        VkPipelineStageFlags src_stages;
        VkPipelineStageFlags dst_stages;
        VkAccessFlags src_access;
        VkAccessFlags dst_access;
    };

    // Registers a use and returns the barrier that must precede it, if any.
    //
    // `shader_writable` says whether a shader can write this buffer at all, which
    // is what narrows the first-use floor below. The caller answers it from the
    // BufferUsage, because this file knows Buffer by identity only, on purpose.
    std::optional<Barrier> use(
        Buffer* buffer,
        VkPipelineStageFlags stages,
        VkAccessFlags access,
        bool writes,
        bool shader_writable,
        std::vector<std::size_t>& waits)
    {
        auto [it, inserted] = states_.try_emplace(buffer);
        BufferState& st = it->second;
        std::optional<Barrier> result;

        // The first READ of a buffer in this recording has no predecessor here to
        // name, and the writer may be outside the recording entirely: another
        // CommandBuffer that shares the buffer, or the previous replay of this
        // one. So it synchronizes against everything that could have written it.
        // Same argument and the same shape as the image floor below.
        //
        // Reads only, and that is the whole precision of it. A recording that
        // WRITES anything tracked already emits the replay wrap-around barrier at
        // the top of execute() — so its own first write is ordered against every
        // prior submit, and a floor here would be a second barrier for a
        // dependency that already exists. `tracked_writes_` is per RECORDING
        // though, not per buffer, which is why the read side was never covered:
        // a recording whose only writes are attachments — any ordinary draw —
        // sets it false and emitted nothing at all. That is the gap
        // examples/28_gpu_culling papered over with a manual barrier.
        //
        // The mask is narrowed by what can reach this buffer, not the floor
        // itself. Every type carries TRANSFER_DST (buffer_usage_for), so
        // cmd.copy_buffer and cmd.fill_buffer reach any of them; only STORAGE
        // carries VK_BUFFER_USAGE_STORAGE_BUFFER_BIT and can be written by a
        // shader. A vertex, index or uniform buffer therefore waits on TRANSFER
        // alone — idle in almost every frame — and only a storage buffer pays for
        // the shader stages.
        if (inserted && !writes)
        {
            VkPipelineStageFlags floor_stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkAccessFlags floor_access = VK_ACCESS_TRANSFER_WRITE_BIT;
            if (shader_writable)
            {
                floor_stages |= all_shader_stages_;
                floor_access |= VK_ACCESS_SHADER_WRITE_BIT;
            }
            result = Barrier{floor_stages, stages, floor_access, access};
        }

        if (writes)
        {
            // WAW / WAR: everything that touched the buffer must drain first.
            // Whatever touched it on ANOTHER queue becomes a wait; a barrier is
            // still due for the half that ran on this one.
            const bool local = predecessors_(st, /*include_reads=*/true, waits);
            if (local && (st.written || st.read_stages != 0))
            {
                result = Barrier{st.write_stages | st.read_stages, stages, st.write_access | st.read_access, access};
            }
            st = {};
            st.written = true;
            st.write_stages = stages;
            st.write_access = access;
            st.write_batch = batch_;
            st.write_queue = queue_;
            st.write_pass = pass_;
        }
        else
        {
            // RAW — only if the write isn't already visible to these stages ON
            // THIS QUEUE. (Two draws reading the same SSBO emit one barrier,
            // not two.)
            if (st.written &&
                ((stages & ~st.visible_stages[queue_slot_]) != 0 || (access & ~st.visible_access[queue_slot_]) != 0))
            {
                if (st.write_queue == queue_)
                {
                    result = Barrier{st.write_stages, stages, st.write_access, access};
                    st.visible_stages[queue_slot_] |= stages;
                    st.visible_access[queue_slot_] |= access;
                }
                else
                {
                    // The wait carries it, and a semaphore wait's second scope
                    // covers every access of every command later in submission
                    // order on this queue — so the write is visible here to
                    // everything from now on, not only to these stages.
                    waits.push_back(st.write_batch);
                    st.visible_stages[queue_slot_] = ~VkPipelineStageFlags{0};
                    st.visible_access[queue_slot_] = ~VkAccessFlags{0};
                }
            }
            st.read_stages |= stages;
            st.read_access |= access;
            st.read_batch[queue_slot_] = batch_;
            st.read_pass[queue_slot_] = pass_;
        }
        return result;
    }

    // Same hazard logic as a buffer, plus a layout: a storage image must be in
    // GENERAL to be read/written in a shader, SHADER_READ_ONLY to be sampled, so
    // every use may need a layout transition on top of the memory barrier.
    struct ImageBarrier
    {
        VkImageLayout old_layout;
        VkImageLayout new_layout;
        VkPipelineStageFlags src_stages;
        VkPipelineStageFlags dst_stages;
        VkAccessFlags src_access;
        VkAccessFlags dst_access;
        // Which subresources the barrier covers; layer_count == 0 = all of
        // them (the pre-0.30 behaviour, and still the common case).
        ImageRange range{};
    };

    // Registers an image use in `layout` and returns the barriers that must
    // precede it. Keyed on Image* (object identity), like buffers.
    //
    // The image's layout at the START of each replay is taken to be UNDEFINED:
    // the recording replays every submit, and a discard on entry is legal from
    // any real layout, so a storage image is re-established from scratch each
    // frame. Consequence — the documented ceiling — is that contents are NOT
    // carried between submits through the tracker; a dispatch that wants last
    // frame's image must overwrite it (post-processing does) or use cmd.barrier.
    //
    // Per (layer, mip) since 0.30, in the SubresourceLayouts shape: one state
    // for a uniform image, a split on the first narrowed use, a collapse back
    // when every entry agrees again. An image nobody narrows costs what it
    // cost before, and a pyramid pass can hold mip N-1 sampled while it
    // writes mip N — two layouts on one image at once, which one state per
    // image could never say.
    // ponytail: one barrier per subresource on a split image, no range
    // coalescing; merge adjacent equal barriers if a profile ever shows it.
    std::vector<ImageBarrier> use_image(
        Image* image,
        VkImageLayout layout,
        VkPipelineStageFlags stages,
        VkAccessFlags access,
        bool writes,
        std::vector<std::size_t>& waits,
        const ImageRange& range = {},
        std::uint32_t layers = 1,
        std::uint32_t mips = 1)
    {
        auto [it, inserted] = image_states_.try_emplace(image);
        ImageStates& states = it->second;
        states.layers = (std::max)(states.layers, layers);
        states.mips = (std::max)(states.mips, mips);
        const bool whole = range.layer_count == 0 || (range.base_layer == 0 && range.layer_count == states.layers &&
                                                      range.base_mip == 0 && range.mip_count == states.mips);
        std::vector<ImageBarrier> out;
        if (whole && states.split.empty())
        {
            if (auto b = use_one_(states.whole, layout, stages, access, writes, waits))
            {
                out.push_back(*b);
            }
            return out;
        }
        if (states.split.empty())
        {
            states.split.assign(static_cast<std::size_t>(states.layers) * states.mips, states.whole);
        }
        const std::uint32_t l0 = whole ? 0 : range.base_layer;
        const std::uint32_t lc = whole ? states.layers : range.layer_count;
        const std::uint32_t m0 = whole ? 0 : range.base_mip;
        const std::uint32_t mc = whole ? states.mips : range.mip_count;
        for (std::uint32_t l = l0; l < l0 + lc; ++l)
        {
            for (std::uint32_t m = m0; m < m0 + mc; ++m)
            {
                if (auto b = use_one_(states.split[l * states.mips + m], layout, stages, access, writes, waits))
                {
                    b->range = {.base_layer = l, .layer_count = 1, .base_mip = m, .mip_count = 1};
                    out.push_back(*b);
                }
            }
        }
        // A whole write resets every entry to one value, so the split ends
        // there; a partially-diverged image keeps paying per subresource
        // until then.
        if (std::ranges::all_of(states.split, [&](const ImageState& s) { return s == states.split.front(); }))
        {
            states.whole = states.split.front();
            states.split.clear();
        }
        return out;
    }

private:
    // The hazard logic for ONE state — the pre-0.30 use_image body, verbatim
    // in behaviour. Same logic as a buffer, plus a layout: a storage image
    // must be in GENERAL to be read/written in a shader, SHADER_READ_ONLY to
    // be sampled, so every use may need a layout transition on top of the
    // memory barrier.
    std::optional<ImageBarrier> use_one_(
        ImageState& st,
        VkImageLayout layout,
        VkPipelineStageFlags stages,
        VkAccessFlags access,
        bool writes,
        std::vector<std::size_t>& waits)
    {
        const VkImageLayout old = st.layout;
        const bool layout_change = (old != layout);
        std::optional<ImageBarrier> result;

        // The very first use across the whole recording synchronizes against
        // every shader stage: a previous frame's replay may still be sampling
        // this image (WAR), and there is no earlier use in THIS recording to
        // name as the source. Later uses name their real predecessor.
        auto with_first_use_floor = [&](VkPipelineStageFlags s,
                                        VkAccessFlags a) -> std::pair<VkPipelineStageFlags, VkAccessFlags>
        {
            if (s == 0)
            {
                return {all_shader_stages_, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
            }
            return {s, a};
        };

        // Whether anything anywhere in this fold has touched the state yet:
        // what tells a genuine first use (the floor applies) from one whose
        // predecessor simply ran on the other queue (the semaphore applies).
        const bool touched = touched_(st);

        if (writes)
        {
            const bool local = predecessors_(st, /*include_reads=*/true, waits);
            if (local || !touched)
            {
                if (st.written || st.read_stages != 0 || layout_change)
                {
                    auto [ss, sa] =
                        with_first_use_floor(st.write_stages | st.read_stages, st.write_access | st.read_access);
                    result = ImageBarrier{
                        .old_layout = old,
                        .new_layout = layout,
                        .src_stages = ss,
                        .dst_stages = stages,
                        .src_access = sa,
                        .dst_access = access};
                }
            }
            else if (layout_change)
            {
                // The other queue's work is already complete and visible by the
                // time this batch runs (the semaphore wait says so), so the
                // transition needs no source scope — only the layout move.
                result = ImageBarrier{
                    .old_layout = old,
                    .new_layout = layout,
                    .src_stages = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    .dst_stages = stages,
                    .dst_access = access};
            }
            st = {};
            st.layout = layout;
            st.written = true;
            st.write_stages = stages;
            st.write_access = access;
            st.write_batch = batch_;
            st.write_queue = queue_;
            st.write_pass = pass_;
        }
        else
        {
            const bool needs = layout_change || (st.written && ((stages & ~st.visible_stages[queue_slot_]) != 0 ||
                                                                (access & ~st.visible_access[queue_slot_]) != 0));
            if (needs)
            {
                // A layout transition is itself a write, so it has to wait for
                // every reader on the other queue, not only for the writer.
                const bool local = predecessors_(st, /*include_reads=*/layout_change, waits);
                if (local || !touched)
                {
                    auto [ss, sa] = with_first_use_floor(
                        st.written ? st.write_stages : st.read_stages, st.written ? st.write_access : st.read_access);
                    result = ImageBarrier{
                        .old_layout = old,
                        .new_layout = layout,
                        .src_stages = ss,
                        .dst_stages = stages,
                        .src_access = sa,
                        .dst_access = access};
                }
                else if (layout_change)
                {
                    result = ImageBarrier{
                        .old_layout = old,
                        .new_layout = layout,
                        .src_stages = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        .dst_stages = stages,
                        .dst_access = access};
                }
                if (st.written && st.write_queue == queue_)
                {
                    st.visible_stages[queue_slot_] |= stages;
                    st.visible_access[queue_slot_] |= access;
                }
                else if (st.written)
                {
                    st.visible_stages[queue_slot_] = ~VkPipelineStageFlags{0};
                    st.visible_access[queue_slot_] = ~VkAccessFlags{0};
                }
            }
            st.layout = layout;
            st.read_stages |= stages;
            st.read_access |= access;
            st.read_batch[queue_slot_] = batch_;
            st.read_pass[queue_slot_] = pass_;
        }
        return result;
    }

public:
    // Every batch on another queue that has touched this image, without
    // changing anything. A render pass's attachment transitions are recorded
    // by the RenderTarget rather than through use_image, so the fold asks this
    // before one runs: a pass on the compute queue that sampled the attachment
    // has to finish before the drawing overwrites it.
    void cross_queue_touches(Image* image, std::vector<std::size_t>& waits) const
    {
        const auto it = image_states_.find(image);
        if (it == image_states_.end())
        {
            return;
        }
        static_cast<void>(predecessors_(it->second.whole, /*include_reads=*/true, waits));
        for (const ImageState& st : it->second.split)
        {
            static_cast<void>(predecessors_(st, /*include_reads=*/true, waits));
        }
    }

    // Has this image been touched in the current recording? track_draw_ uses
    // this to leave uploaded textures (never seen by the tracker) alone while
    // still transitioning a compute-written image before it is sampled.
    bool tracks(Image* image) const
    {
        return image_states_.contains(image);
    }

    // The ranged twin (0.30), and it is load-bearing for the pyramid: the
    // storage write of mip N makes the image "tracked", and without the range
    // the sample of the uploaded mip 0 in the same pass would be transitioned
    // from the fold's UNDEFINED — a discard of the source.
    bool tracks(Image* image, const ImageRange& range) const
    {
        const auto it = image_states_.find(image);
        if (it == image_states_.end())
        {
            return false;
        }
        const ImageStates& states = it->second;
        if (states.split.empty())
        {
            return touched_(states.whole);
        }
        if (range.layer_count == 0)
        {
            return std::ranges::any_of(states.split, [](const ImageState& s) { return touched_(s); });
        }
        for (std::uint32_t l = range.base_layer; l < range.base_layer + range.layer_count; ++l)
        {
            for (std::uint32_t m = range.base_mip; m < range.base_mip + range.mip_count; ++m)
            {
                const std::size_t index = static_cast<std::size_t>(l) * states.mips + m;
                if (index < states.split.size() && touched_(states.split[index]))
                {
                    return true;
                }
            }
        }
        return false;
    }

    // Whether the fold has left ANY subresource of the image somewhere other
    // than `layout`. A render pass derives its entry barrier from the
    // RenderTarget rather than from any state, so when it PRESERVES an
    // attachment something else moved, the graph asks here and corrects the
    // difference first — per subresource where the image is split.
    bool layouts_differ(Image* image, VkImageLayout layout) const
    {
        const auto it = image_states_.find(image);
        if (it == image_states_.end())
        {
            return false;
        }
        if (it->second.split.empty())
        {
            return it->second.whole.layout != layout;
        }
        return std::ranges::any_of(it->second.split, [&](const ImageState& s) { return s.layout != layout; });
    }

    // A manual cmd.barrier(image) / generate_mipmaps already recorded a real
    // transition to `layout`, making prior work available to (dst_stages,
    // dst_access). Seed the tracker so a following automatic use of the same
    // image in this recording sees the real layout and does NOT re-transition
    // with a stale oldLayout (a validation error) — and WAR/WAW-orders correctly
    // against these consumers. Modelling dst as a completed read is right for
    // both the READ case (a later sample needs no barrier) and the WRITE case (a
    // later write waits on dst before overwriting).
    // The buffer counterpart, and it exists for the first of those two reasons.
    // A manual cmd.barrier(buffer, ...) records a real barrier and used to tell
    // the tracker nothing, which cost nothing while buffers had no floor. With
    // the floor, the automatic use right after a manual barrier would emit a
    // second barrier for a dependency the caller just expressed. Seeding the
    // state as a completed read removes it, and orders a later write in this
    // recording against the consumers the caller named.
    void note_buffer_access(
        Buffer* buffer,
        VkPipelineStageFlags dst_stages,
        VkAccessFlags dst_access,
        std::vector<std::size_t>& waits)
    {
        BufferState& st = states_[buffer];
        // The caller wrote the local half of this dependency by hand. The half
        // a pipeline barrier cannot express — a producer on the other queue —
        // is still the fold's to add.
        static_cast<void>(predecessors_(st, /*include_reads=*/true, waits));
        // Two p.barrier() calls in one pass both made memory available, so
        // same-pass notes accumulate their visible masks (0.30). Keeping only
        // the last dst mask made correct manual code trip the lint.
        VkPipelineStageFlags carried_stages = 0;
        VkAccessFlags carried_access = 0;
        if (st.written && st.write_pass == pass_)
        {
            carried_stages = st.visible_stages[queue_slot_];
            carried_access = st.visible_access[queue_slot_];
        }
        st = {};
        // Recorded as a WRITE this queue has already made available, not as a
        // bare read. The visible mask is what stops a redundant barrier on this
        // queue — a use covered by the caller's own barrier asks for nothing
        // more — while `written` is what a use on the OTHER queue trips on, so
        // it gets the semaphore wait a pipeline barrier could never give it.
        //
        // Leaving `written` false modelled this as "somebody read it", and a
        // read is not something a later reader has to wait for: a manual
        // barrier, a copy_image or a clear_image on one queue then left the
        // other queue's reader completely unordered.
        st.written = true;
        st.write_stages = dst_stages;
        st.write_access = dst_access;
        st.write_batch = batch_;
        st.write_queue = queue_;
        st.write_pass = pass_;
        st.read_stages = dst_stages;
        st.read_access = dst_access;
        st.visible_stages[queue_slot_] = dst_stages | carried_stages;
        st.visible_access[queue_slot_] = dst_access | carried_access;
        st.read_batch[queue_slot_] = batch_;
        st.read_pass[queue_slot_] = pass_;
    }

    // A render pass WROTE this image as an attachment, and left it in `layout`.
    // The graph reports this after folding a render pass, because an attachment
    // write is not a descriptor use and nothing else would report it — without
    // it a later pass sampling the image sees no predecessor and gets no
    // barrier, which is a READ_AFTER_WRITE hazard on the most ordinary thing a
    // frame does.
    //
    // Distinct from note_image_layout, which models a COMPLETED read: this one
    // has to leave the write PENDING so the next read is ordered against it.
    // The layout is already correct on the device (the pass's own exit
    // transition put it there), so nothing here asks for a transition — only
    // for the memory dependency.
    //
    // `visible_*` is what that exit transition already made available. A
    // colour attachment retires naming the fragment shader, because sampling
    // the result is what an offscreen target is for, so a fragment read after
    // it needs nothing more and gets no second barrier. A read from any OTHER
    // stage — a compute pass consuming a rendered texture — is outside what
    // the retire named, and that one does get the barrier.
    void note_image_write(
        Image* image,
        VkImageLayout layout,
        VkPipelineStageFlags src_stages,
        VkAccessFlags src_access,
        VkPipelineStageFlags visible_stages,
        VkAccessFlags visible_access)
    {
        // Whole-image on purpose: an attachment write covers every layer the
        // target wrote, and the note supersedes any split. Accepted ceiling —
        // rendering into target.layer(i) still claims the whole image here.
        ImageStates& states = image_states_[image];
        states.split.clear();
        ImageState& st = states.whole;
        st = {};
        st.layout = layout;
        st.written = true;
        st.write_stages = src_stages;
        st.write_access = src_access;
        st.write_batch = batch_;
        st.write_queue = queue_;
        st.write_pass = pass_;
        st.visible_stages[queue_slot_] = visible_stages;
        st.visible_access[queue_slot_] = visible_access;
    }

    // An image this fold will not touch, read by this batch. Records the read
    // and nothing else: no layout is claimed and no barrier is implied, because
    // the image is already in the layout the read names — it is only here so a
    // later pass that WRITES the image (a render pass drawing into it) can be
    // ordered against the read, which across queues has no other way to happen.
    void note_image_read(
        Image* image,
        VkImageLayout layout,
        VkPipelineStageFlags stages,
        VkAccessFlags access,
        const ImageRange& range = {},
        std::uint32_t layers = 1,
        std::uint32_t mips = 1)
    {
        ImageStates& states = image_states_[image];
        states.layers = (std::max)(states.layers, layers);
        states.mips = (std::max)(states.mips, mips);
        const auto note_one = [&](ImageState& st)
        {
            st.layout = layout;
            st.read_stages |= stages;
            st.read_access |= access;
            st.visible_stages[queue_slot_] |= stages;
            st.visible_access[queue_slot_] |= access;
            st.read_batch[queue_slot_] = batch_;
            st.read_pass[queue_slot_] = pass_;
        };
        const bool whole = range.layer_count == 0 || (range.base_layer == 0 && range.layer_count == states.layers &&
                                                      range.base_mip == 0 && range.mip_count == states.mips);
        if (whole && states.split.empty())
        {
            note_one(states.whole);
            return;
        }
        if (states.split.empty())
        {
            states.split.assign(static_cast<std::size_t>(states.layers) * states.mips, states.whole);
        }
        const std::uint32_t l0 = whole ? 0 : range.base_layer;
        const std::uint32_t lc = whole ? states.layers : range.layer_count;
        const std::uint32_t m0 = whole ? 0 : range.base_mip;
        const std::uint32_t mc = whole ? states.mips : range.mip_count;
        for (std::uint32_t l = l0; l < l0 + lc; ++l)
        {
            for (std::uint32_t m = m0; m < m0 + mc; ++m)
            {
                note_one(states.split[l * states.mips + m]);
            }
        }
    }

    void note_image_layout(
        Image* image,
        VkImageLayout layout,
        VkPipelineStageFlags dst_stages,
        VkAccessFlags dst_access,
        std::vector<std::size_t>* waits = nullptr)
    {
        // Whole-image on purpose (a manual cmd.barrier names the whole image —
        // accepted ceiling), so a split collapses to what the note says.
        ImageStates& states = image_states_[image];
        if (waits != nullptr)
        {
            // Same argument as note_buffer_access: a manual transition covers
            // this queue, and a producer on the other one still needs a wait.
            // The parameter is optional because the fold also calls this to
            // record a transition it just emitted itself, where there is
            // nothing to add.
            static_cast<void>(predecessors_(states.whole, /*include_reads=*/true, *waits));
            for (const ImageState& split_state : states.split)
            {
                static_cast<void>(predecessors_(split_state, /*include_reads=*/true, *waits));
            }
        }
        states.split.clear();
        ImageState& st = states.whole;
        // Same-pass accumulation, as note_buffer_access — the layout has to
        // agree, because a note that MOVED the image did supersede the last.
        VkPipelineStageFlags carried_stages = 0;
        VkAccessFlags carried_access = 0;
        if (st.written && st.write_pass == pass_ && st.layout == layout)
        {
            carried_stages = st.visible_stages[queue_slot_];
            carried_access = st.visible_access[queue_slot_];
        }
        st = {};
        st.layout = layout;
        // A write this queue has already made available — see
        // note_buffer_access for why it is not modelled as a read.
        st.written = true;
        st.write_stages = dst_stages;
        st.write_access = dst_access;
        st.write_batch = batch_;
        st.write_queue = queue_;
        st.write_pass = pass_;
        st.read_stages = dst_stages;
        st.read_access = dst_access;
        st.visible_stages[queue_slot_] = dst_stages | carried_stages;
        st.visible_access[queue_slot_] = dst_access | carried_access;
        st.read_batch[queue_slot_] = batch_;
        st.read_pass[queue_slot_] = pass_;
    }

    void reset()
    {
        states_.clear();
        image_states_.clear();
    }

    const BufferState* state(Buffer* buffer) const
    {
        const auto it = states_.find(buffer);
        return it != states_.end() ? &it->second : nullptr;
    }
    // The whole-image state for explain() and the lint. Where the image is
    // split this answers with the FIRST touched subresource — a debugging aid
    // needs a representative previous state, not all of them.
    const ImageState* image_state(Image* image) const
    {
        const auto it = image_states_.find(image);
        if (it == image_states_.end())
        {
            return nullptr;
        }
        if (it->second.split.empty())
        {
            return &it->second.whole;
        }
        for (const ImageState& st : it->second.split)
        {
            if (touched_(st))
            {
                return &st;
            }
        }
        return &it->second.whole;
    }

private:
    // What a p.barrier() note in this pass established: the write is on this
    // queue and the requested scopes sit inside the visible masks.
    template <typename State>
    bool covered_(const State& st, VkPipelineStageFlags stages, VkAccessFlags access) const
    {
        return st.written && st.write_queue == queue_ && (stages & ~st.visible_stages[queue_slot_]) == 0 &&
               (access & ~st.visible_access[queue_slot_]) == 0;
    }

    // Whichever touches of `st` happened on ANOTHER queue become semaphore
    // waits; returns whether anything touched it on THIS one, which is what
    // still needs a pipeline barrier.
    template <typename State>
    bool predecessors_(const State& st, bool include_reads, std::vector<std::size_t>& waits) const
    {
        bool local = false;
        if (st.write_batch != kNoBatch)
        {
            if (st.write_queue == queue_)
            {
                local = true;
            }
            else
            {
                waits.push_back(st.write_batch);
            }
        }
        if (include_reads)
        {
            for (std::size_t q = 0; q < kQueueCount; ++q)
            {
                if (st.read_batch[q] == kNoBatch)
                {
                    continue;
                }
                if (q == queue_slot_)
                {
                    local = true;
                }
                else
                {
                    waits.push_back(st.read_batch[q]);
                }
            }
        }
        return local;
    }

    // Defaults to the mask every conformant device has, so a tracker nobody told
    // is narrow rather than illegal. reset() must NOT clear it: it survives every
    // recording, like the device it describes.
    VkPipelineStageFlags all_shader_stages_ = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    // Which batch is being folded, and on which queue. Set per pass by the
    // graph; a tracker nobody told folds everything as batch 0 on the graphics
    // queue, which is what a single-queue graph is.
    std::size_t batch_ = 0;
    QueueKind queue_ = QueueKind::Graphics;
    std::size_t queue_slot_ = 0;
    std::size_t pass_ = kNoPass;

    std::unordered_map<Buffer*, BufferState> states_;
    std::unordered_map<Image*, ImageStates> image_states_;
};
