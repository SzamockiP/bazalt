#pragma once
#include <volk.h>
#include <optional>
#include <unordered_map>

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
    INDIRECT_READ
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
    }
    // Not std::unreachable(): pybind enums accept arbitrary ints.
    return {all_shader_stages, VK_ACCESS_SHADER_READ_BIT};
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
        bool shader_writable)
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
            if (st.written || st.read_stages != 0)
            {
                result = Barrier{st.write_stages | st.read_stages, stages, st.write_access | st.read_access, access};
            }
            st = {};
            st.written = true;
            st.write_stages = stages;
            st.write_access = access;
        }
        else
        {
            // RAW — only if the write isn't already visible to these stages.
            // (Two draws reading the same SSBO emit one barrier, not two.)
            if (st.written && ((stages & ~st.visible_stages) != 0 || (access & ~st.visible_access) != 0))
            {
                result = Barrier{st.write_stages, stages, st.write_access, access};
                st.visible_stages |= stages;
                st.visible_access |= access;
            }
            st.read_stages |= stages;
            st.read_access |= access;
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
        bool writes)
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

        if (writes)
        {
            if (st.written || st.read_stages != 0 || layout_change)
            {
                auto [ss, sa] =
                    with_first_use_floor(st.write_stages | st.read_stages, st.write_access | st.read_access);
                result = ImageBarrier{old, layout, ss, stages, sa, access};
            }
            st = {};
            st.layout = layout;
            st.written = true;
            st.write_stages = stages;
            st.write_access = access;
        }
        else
        {
            const bool needs = layout_change || (st.written && ((stages & ~st.visible_stages) != 0 ||
                                                                (access & ~st.visible_access) != 0));
            if (needs)
            {
                auto [ss, sa] = with_first_use_floor(
                    st.written ? st.write_stages : st.read_stages, st.written ? st.write_access : st.read_access);
                result = ImageBarrier{old, layout, ss, stages, sa, access};
                st.visible_stages |= stages;
                st.visible_access |= access;
            }
            st.layout = layout;
            st.read_stages |= stages;
            st.read_access |= access;
        }
        return result;
    }

    // Has this image been touched in the current recording? track_draw_ uses
    // this to leave uploaded textures (never seen by the tracker) alone while
    // still transitioning a compute-written image before it is sampled.
    bool tracks(Image* image) const
    {
        return image_states_.contains(image);
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
    void note_buffer_access(Buffer* buffer, VkPipelineStageFlags dst_stages, VkAccessFlags dst_access)
    {
        BufferState& st = states_[buffer];
        st = {};
        st.read_stages = dst_stages;
        st.read_access = dst_access;
        st.visible_stages = dst_stages;
        st.visible_access = dst_access;
    }

    void note_image_layout(
        Image* image,
        VkImageLayout layout,
        VkPipelineStageFlags dst_stages,
        VkAccessFlags dst_access)
    {
        ImageState& st = image_states_[image];
        st = {};
        st.layout = layout;
        st.read_stages = dst_stages;
        st.read_access = dst_access;
        st.visible_stages = dst_stages;
        st.visible_access = dst_access;
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
        // Stages/accesses already synchronized against the last write.
        VkPipelineStageFlags visible_stages = 0;
        VkAccessFlags visible_access = 0;
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
        VkPipelineStageFlags visible_stages = 0;
        VkAccessFlags visible_access = 0;
        VkPipelineStageFlags read_stages = 0;
        VkAccessFlags read_access = 0;
    };

    // Defaults to the mask every conformant device has, so a tracker nobody told
    // is narrow rather than illegal. reset() must NOT clear it: it survives every
    // recording, like the device it describes.
    VkPipelineStageFlags all_shader_stages_ = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    std::unordered_map<Buffer*, BufferState> states_;
    std::unordered_map<Image*, ImageState> image_states_;
};
