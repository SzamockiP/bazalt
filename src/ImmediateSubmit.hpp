#pragma once
#include <volk.h>

#include <cstdint>
#include <cstring>
#include <expected>
#include <mutex>
#include <utility>

#include "Context.hpp"
#include "Error.hpp"

enum class Staging
{
    Upload,  // TRANSFER_SRC, write-combined — the CPU fills it, the GPU reads it
    Readback // TRANSFER_DST, host-random — the GPU fills it, the CPU maps it
};

// A host-visible buffer used to carry bytes across the CPU/GPU boundary, which
// is the other half of a one-shot submit: every upload copies out of one and
// every readback copies into one. The two directions differ only in the usage
// and VMA flags, which is why one function covers every staging buffer bazalt
// makes.
//
// `data` fills an Upload buffer on the spot. Leaving it null is legal and means
// "allocate, do not fill" — create_buffer(size_in_bytes) has no bytes yet.
//
// The caller owns the result and must vmaDestroyBuffer it — through
// Context::defer_destroy when a submit still references it.
inline std::expected<std::pair<VkBuffer, VmaAllocation>, Error> create_staging_buffer(
    Context& context,
    VkDeviceSize size,
    Staging direction,
    const void* data = nullptr)
{
    const bool upload = direction == Staging::Upload;

    VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = size,
        .usage = static_cast<VkBufferUsageFlags>(
            upload ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT : VK_BUFFER_USAGE_TRANSFER_DST_BIT),
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr};

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = upload ? VMA_MEMORY_USAGE_CPU_ONLY : VMA_MEMORY_USAGE_GPU_TO_CPU;
    allocInfo.flags = upload ? VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                             : VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

    VkBuffer staging = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    if (auto e = check(
            vmaCreateBuffer(context.allocator(), &bufferInfo, &allocInfo, &staging, &allocation, nullptr),
            upload ? "create upload staging buffer" : "create readback staging buffer",
            ErrorCode::Resource))
    {
        return std::unexpected(*e);
    }

    if (upload && data != nullptr && size > 0)
    {
        void* mapped = nullptr;
        if (auto e = check(
                vmaMapMemory(context.allocator(), allocation, &mapped),
                "map upload staging buffer",
                ErrorCode::Resource))
        {
            vmaDestroyBuffer(context.allocator(), staging, allocation);
            return std::unexpected(*e);
        }
        std::memcpy(mapped, data, static_cast<std::size_t>(size));
        vmaUnmapMemory(context.allocator(), allocation);
    }

    return std::pair{staging, allocation};
}

// Records `record(VkCommandBuffer)` into a freshly allocated one-shot command
// buffer and submits it to the graphics queue, signalling the Context's
// submission timeline. Returns the serial that submit will signal, and does
// NOT wait for it.
//
// This is the transport under every "the resource is its own future" upload
// (Image::upload_pixels, StaticBuffer::create). The caller keeps the serial:
// a submit that reads the resource waits on it GPU-side, and `.wait()` waits
// on it CPU-side. Nothing here needs the upload worker — a numpy array or a
// Python list has no decode to move off the main thread, so the only cost an
// upload used to pay was the vkQueueWaitIdle, and this is what removes it.
//
// The command buffer retires through the deletion queue, which is keyed on
// exactly this serial. Whatever else the recording borrowed (a staging
// buffer) is the caller's to defer the same way.
template <typename F>
std::expected<std::uint64_t, Error> deferred_submit(Context& context, F&& record)
{
    const VolkDeviceTable& vk = context.vk();

    VkCommandBufferAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = context.command_pool(),
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1};

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (auto e = check(
            vk.vkAllocateCommandBuffers(context.device(), &allocInfo, &cmd),
            "allocate one-shot command buffer",
            ErrorCode::Resource))
    {
        return std::unexpected(*e);
    }

    // Before the submit nothing references the command buffer, so a failure
    // here frees it inline rather than through the deletion queue.
    const auto fail = [&](Error error) -> std::expected<std::uint64_t, Error>
    {
        vk.vkFreeCommandBuffers(context.device(), context.command_pool(), 1, &cmd);
        return std::unexpected(std::move(error));
    };

    VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr};
    if (auto e = check(vk.vkBeginCommandBuffer(cmd, &beginInfo), "begin one-shot command buffer", ErrorCode::Resource))
    {
        return fail(std::move(*e));
    }

    std::forward<F>(record)(cmd);

    if (auto e = check(vk.vkEndCommandBuffer(cmd), "record one-shot command buffer", ErrorCode::Resource))
    {
        return fail(std::move(*e));
    }

    // One-shot submits count on the submission timeline too, so the deletion
    // queue keeps draining even on frame-less workloads.
    auto submitted = context.submit_one_shot(cmd);
    if (!submitted)
    {
        return fail(std::move(submitted.error()));
    }
    const std::uint64_t serial = *submitted;

    // The GPU still holds it, so this cannot be an inline free. The deletion
    // queue keys the entry on the serial above, which is exactly when the
    // command buffer stops being referenced.
    // Raw handles plus a pointer to the dispatch table, never a
    // shared_ptr<Context> — the table lives in the Context, which outlives
    // every drain of its own deletion queue.
    context.defer_destroy([device = context.device(), pool = context.command_pool(), table = &vk, cmd]
                          { table->vkFreeCommandBuffers(device, pool, 1, &cmd); });
    return serial;
}

// The blocking half: one deferred_submit plus a wait for its own serial.
//
// This is THE way to do a synchronous GPU round trip. It used to be uploads as
// well as readbacks, and the per-upload queue drain was the whole cost that
// made a Context with many buffers or procedural textures slow to start; since
// 0.18.0 an upload keeps its serial instead (see deferred_submit) and only a
// readback, which has to hand bytes back to Python on the next line, still
// stands still for the GPU.
template <typename F>
std::expected<void, Error> immediate_submit(Context& context, F&& record)
{
    auto serial = deferred_submit(context, std::forward<F>(record));
    if (!serial)
    {
        return std::unexpected(serial.error());
    }

    // Waits for this submit and nothing else. A vkQueueWaitIdle here would also
    // stall on whatever the upload worker happens to have in flight, which a
    // readback has no reason to care about.
    if (auto r = context.wait_for_serial(*serial); !r)
    {
        return std::unexpected(r.error());
    }

    // Something has provably completed, so this is a free chance to reclaim
    // deferred handles — this submit's own command buffer among them.
    context.flush_deletion_queue();
    return {};
}
