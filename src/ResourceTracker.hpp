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
    UNIFORM_READ
};

struct StageAccess
{
    VkPipelineStageFlags stages;
    VkAccessFlags access;
};

// Core-1.0 pairs on purpose: the whole codebase rides vkCmdPipelineBarrier,
// not synchronization2, and mixing models would be a second way to say the
// same thing.
inline constexpr VkPipelineStageFlags kAllShaderStages =
    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

inline constexpr StageAccess to_vk(Access access)
{
    switch (access)
    {
        case Access::SHADER_READ:
            return {kAllShaderStages, VK_ACCESS_SHADER_READ_BIT};
        case Access::SHADER_WRITE:
            return {kAllShaderStages, VK_ACCESS_SHADER_WRITE_BIT};
        case Access::VERTEX_READ:
            return {VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT};
        case Access::INDEX_READ:
            return {VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_ACCESS_INDEX_READ_BIT};
        case Access::UNIFORM_READ:
            return {kAllShaderStages, VK_ACCESS_UNIFORM_READ_BIT};
    }
    // Not std::unreachable(): pybind enums accept arbitrary ints.
    return {kAllShaderStages, VK_ACCESS_SHADER_READ_BIT};
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
// Scope (0.6): buffers only. The compute API has no storage images, and
// sampled images are covered by the RenderTarget final_layout contract, so a
// buffer tracker covers everything the Python API can express. Known limit:
// SSBO *writes* from graphics shaders are invisible here (no reflection) —
// cmd.barrier() in manual mode is the ceiling for that.
class ResourceTracker
{
public:
    struct Barrier
    {
        VkPipelineStageFlags src_stages;
        VkPipelineStageFlags dst_stages;
        VkAccessFlags src_access;
        VkAccessFlags dst_access;
    };

    // Registers a use and returns the barrier that must precede it, if any.
    std::optional<Barrier> use(Buffer* buffer, VkPipelineStageFlags stages, VkAccessFlags access, bool writes)
    {
        BufferState& st = states_[buffer];
        std::optional<Barrier> result;

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
                return {kAllShaderStages, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
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

    std::unordered_map<Buffer*, BufferState> states_;
    std::unordered_map<Image*, ImageState> image_states_;
};
