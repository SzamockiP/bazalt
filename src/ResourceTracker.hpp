#pragma once
#include <volk.h>
#include <array>
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

    // Which batch the fold is currently folding, and on which queue. The
    // caller sets it before each pass; everything below records it, so a use
    // whose producer ran on the other queue can name the batch to wait for
    // instead of emitting a barrier that cannot reach it.
    void set_batch(std::size_t batch, QueueKind queue)
    {
        batch_ = batch;
        queue_ = queue;
        queue_slot_ = queue_index(queue);
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
    // BufferType, because this file knows Buffer by identity only, on purpose.
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
    };

    // Registers an image use in `layout` and returns the barrier that must
    // precede it, if any. Keyed on Image* (object identity), like buffers.
    //
    // The image's layout at the START of each replay is taken to be UNDEFINED:
    // the recording replays every submit, and a discard on entry is legal from
    // any real layout, so a storage image is re-established from scratch each
    // frame. Consequence — the documented ceiling — is that contents are NOT
    // carried between submits through the tracker; a dispatch that wants last
    // frame's image must overwrite it (post-processing does) or use cmd.barrier.
    std::optional<ImageBarrier> use_image(
        Image* image,
        VkImageLayout layout,
        VkPipelineStageFlags stages,
        VkAccessFlags access,
        bool writes,
        std::vector<std::size_t>& waits)
    {
        auto [it, inserted] = image_states_.try_emplace(image);
        ImageState& st = it->second;
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

        // Whether anything anywhere in this fold has touched the image yet:
        // what tells a genuine first use (the floor applies) from one whose
        // predecessor simply ran on the other queue (the semaphore applies).
        const bool touched = st.written || st.read_batch[0] != kNoBatch || st.read_batch[1] != kNoBatch;

        if (writes)
        {
            const bool local = predecessors_(st, /*include_reads=*/true, waits);
            if (local || !touched)
            {
                if (st.written || st.read_stages != 0 || layout_change)
                {
                    auto [ss, sa] =
                        with_first_use_floor(st.write_stages | st.read_stages, st.write_access | st.read_access);
                    result = ImageBarrier{old, layout, ss, stages, sa, access};
                }
            }
            else if (layout_change)
            {
                // The other queue's work is already complete and visible by the
                // time this batch runs (the semaphore wait says so), so the
                // transition needs no source scope — only the layout move.
                result = ImageBarrier{old, layout, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, stages, 0, access};
            }
            st = {};
            st.layout = layout;
            st.written = true;
            st.write_stages = stages;
            st.write_access = access;
            st.write_batch = batch_;
            st.write_queue = queue_;
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
                    result = ImageBarrier{old, layout, ss, stages, sa, access};
                }
                else if (layout_change)
                {
                    result = ImageBarrier{old, layout, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, stages, 0, access};
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
        }
        return result;
    }

    // Every batch on another queue that has touched this image, without
    // changing anything. A render pass's attachment transitions are recorded
    // by the RenderTarget rather than through use_image, so the fold asks this
    // before one runs: a pass on the compute queue that sampled the attachment
    // has to finish before the drawing overwrites it.
    void cross_queue_touches(Image* image, std::vector<std::size_t>& waits) const
    {
        const auto it = image_states_.find(image);
        if (it != image_states_.end())
        {
            static_cast<void>(predecessors_(it->second, /*include_reads=*/true, waits));
        }
    }

    // Has this image been touched in the current recording? track_draw_ uses
    // this to leave uploaded textures (never seen by the tracker) alone while
    // still transitioning a compute-written image before it is sampled.
    bool tracks(Image* image) const
    {
        return image_states_.contains(image);
    }

    // The layout this fold has left the image in, or nullopt where it has never
    // seen it. A render pass derives its entry barrier from the RenderTarget
    // rather than from any state, so when it PRESERVES an attachment something
    // else moved, the graph asks here and corrects the difference first.
    std::optional<VkImageLayout> layout_of(Image* image) const
    {
        const auto it = image_states_.find(image);
        return it == image_states_.end() ? std::nullopt : std::optional{it->second.layout};
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
        st = {};
        st.read_stages = dst_stages;
        st.read_access = dst_access;
        st.visible_stages[queue_slot_] = dst_stages;
        st.visible_access[queue_slot_] = dst_access;
        st.read_batch[queue_slot_] = batch_;
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
        ImageState& st = image_states_[image];
        st = {};
        st.layout = layout;
        st.written = true;
        st.write_stages = src_stages;
        st.write_access = src_access;
        st.write_batch = batch_;
        st.write_queue = queue_;
        st.visible_stages[queue_slot_] = visible_stages;
        st.visible_access[queue_slot_] = visible_access;
    }

    void note_image_layout(
        Image* image,
        VkImageLayout layout,
        VkPipelineStageFlags dst_stages,
        VkAccessFlags dst_access,
        std::vector<std::size_t>* waits = nullptr)
    {
        ImageState& st = image_states_[image];
        // Same argument as note_buffer_access: a manual transition covers this
        // queue, and a producer on the other one still needs a wait. The
        // parameter is optional because the fold also calls this to record a
        // transition it just emitted itself, where there is nothing to add.
        if (waits != nullptr)
        {
            static_cast<void>(predecessors_(st, /*include_reads=*/true, *waits));
        }
        st = {};
        st.layout = layout;
        st.read_stages = dst_stages;
        st.read_access = dst_access;
        st.visible_stages[queue_slot_] = dst_stages;
        st.visible_access[queue_slot_] = dst_access;
        st.read_batch[queue_slot_] = batch_;
    }

    void reset()
    {
        states_.clear();
        image_states_.clear();
    }

private:
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
        std::array<std::size_t, kQueueCount> read_batch{kNoBatch, kNoBatch};
        // Reads since the last write (what a future write must wait for).
        VkPipelineStageFlags read_stages = 0;
        VkAccessFlags read_access = 0;
    };

    // BufferState plus the layout the recording has left the image in so far.
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
        std::array<std::size_t, kQueueCount> read_batch{kNoBatch, kNoBatch};
        VkPipelineStageFlags read_stages = 0;
        VkAccessFlags read_access = 0;
    };

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

    std::unordered_map<Buffer*, BufferState> states_;
    std::unordered_map<Image*, ImageState> image_states_;
};
