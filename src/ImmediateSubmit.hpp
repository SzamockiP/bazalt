#pragma once
#include <volk.h>

#include <cstdint>
#include <expected>
#include <mutex>
#include <utility>

#include "Context.hpp"
#include "Error.hpp"

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

    VkSemaphore timeline = context.submit_timeline();
    std::uint64_t serial = 0;
    {
        std::lock_guard lock(context.queue_mutex());

        // One-shot submits count on the submission timeline too, so the
        // deletion queue keeps draining even on frame-less workloads.
        serial = context.advance_submit_serial();
        VkTimelineSemaphoreSubmitInfo timelineInfo{
            .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .waitSemaphoreValueCount = 0,
            .pWaitSemaphoreValues = nullptr,
            .signalSemaphoreValueCount = 1,
            .pSignalSemaphoreValues = &serial};

        VkSubmitInfo submitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = &timelineInfo,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = nullptr,
            .pWaitDstStageMask = nullptr,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmd,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &timeline};
        if (auto e = check(
                vk.vkQueueSubmit(context.graphics_queue(), 1, &submitInfo, VK_NULL_HANDLE),
                "submit one-shot command buffer",
                ErrorCode::Resource))
        {
            return fail(std::move(*e));
        }
    }

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

// The blocking half: one deferred_submit plus a wait for the queue to drain.
//
// This is THE way to do a synchronous GPU round trip. It used to be uploads as
// well as readbacks, and the vkQueueWaitIdle per upload was the whole cost that
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

    {
        std::lock_guard lock(context.queue_mutex());
        if (auto e = check(
                context.vk().vkQueueWaitIdle(context.graphics_queue()),
                "wait for one-shot submit",
                ErrorCode::Resource))
        {
            return std::unexpected(*e);
        }
    }

    // The wait-idle above proves everything submitted so far has completed —
    // a free chance to reclaim deferred handles, this submit's own command
    // buffer among them.
    context.flush_deletion_queue();
    return {};
}
