#pragma once
#include <volk.h>

#include <array>
#include <cstddef>
#include <cstdint>

// Which queue a pass runs on, and the vocabulary the three timelines need.
//
// Its own header because both ends need it and neither may include the other:
// Context.hpp owns the runtimes, ResourceTracker.hpp has to know which queue a
// use was folded on, and the tracker deliberately includes nothing but volk.
enum class QueueKind : std::uint8_t
{
    Graphics = 0,
    Compute = 1,
    Transfer = 2
};

// Three, and three for a reason rather than for now: of Vulkan's eight queue
// capability bits only GRAPHICS, COMPUTE and TRANSFER describe work a pass
// records (DESIGN.md, "The queues after 0.29, and where they end").
inline constexpr std::size_t kQueueCount = 3;

// A chain of ternaries rather than a cast: a pybind enum accepts any int, so
// Queue(7) must index something that exists rather than run off the array.
inline constexpr std::size_t queue_index(QueueKind kind)
{
    return kind == QueueKind::Compute ? 1 : kind == QueueKind::Transfer ? 2 : 0;
}

// The Python spelling, for messages that name a queue: "Queue.COMPUTE".
inline constexpr const char* queue_name(QueueKind kind)
{
    switch (kind)
    {
        case QueueKind::Graphics:
            return "GRAPHICS";
        case QueueKind::Compute:
            return "COMPUTE";
        case QueueKind::Transfer:
            return "TRANSFER";
    }
    return "GRAPHICS";
}

// One timeline value per queue. 0 means "nothing to wait for there", which is
// also the value a timeline wait satisfies trivially, so no call site branches
// on it.
using QueueSerials = std::array<std::uint64_t, kQueueCount>;

// An array with every slot set, for the tracker's "no batch yet" sentinels. A
// brace list shorter than kQueueCount value-initializes the rest to 0, which is
// a VALID batch index — the third queue made that a silent bug in waiting.
template <typename T>
inline constexpr std::array<T, kQueueCount> per_queue(T value)
{
    std::array<T, kQueueCount> out{};
    out.fill(value);
    return out;
}

inline constexpr void max_merge(QueueSerials& into, const QueueSerials& from)
{
    for (std::size_t i = 0; i < kQueueCount; ++i)
    {
        into[i] = into[i] > from[i] ? into[i] : from[i];
    }
}

// The stage bits a vkCmdPipelineBarrier recorded into a pool of this family may
// carry. Every bit in srcStageMask/dstStageMask must be supported by the family
// the command buffer's pool was created on, and a compute-only family supports
// far fewer of them than the graphics one — the same class of mistake the
// per-Context all_shader_stages() mask exists for, one dimension over.
//
// TRANSFER is supported by a graphics, compute or transfer family; DRAW_INDIRECT
// by a graphics or compute one (it is the stage that reads dispatch-indirect
// arguments too). HOST and the two ends of the pipe are universal.
inline constexpr VkPipelineStageFlags legal_stages_for(VkQueueFlags family)
{
    if ((family & VK_QUEUE_GRAPHICS_BIT) != 0)
    {
        return ~VkPipelineStageFlags{0};
    }
    if ((family & VK_QUEUE_COMPUTE_BIT) != 0)
    {
        return VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
               VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT | VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }
    return VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT |
           VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
}
