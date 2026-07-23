#pragma once
#include <volk.h>

#include <expected>
#include <mutex>
#include <utility>

#include "Context.hpp"
#include "Error.hpp"

// Runs `record(VkCommandBuffer)` in a freshly allocated one-shot command
// buffer, submits it to the graphics queue and blocks until it completes.
//
// This is THE way to do a synchronous GPU round trip (staging uploads,
// readbacks). It used to be hand-rolled in three places, each ignoring the
// VkResult of allocate/submit/wait — so a lost device during an upload
// surfaced later as inexplicable garbage instead of a DeviceLostError here.
//
// Blocking by design: 0.5's UploadManager (transfer queue + timeline
// semaphores) is what makes uploads asynchronous. Callers should treat this
// as the slow, correct path.
template <typename F>
std::expected<void, Error> immediate_submit(Context& context, F&& record)
{
    VkCommandBufferAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = context.command_pool(),
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1};

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (auto e = check(
            vkAllocateCommandBuffers(context.device(), &allocInfo, &cmd),
            "allocate one-shot command buffer",
            ErrorCode::Resource))
    {
        return std::unexpected(*e);
    }

    const auto fail = [&](Error error) -> std::expected<void, Error>
    {
        vkFreeCommandBuffers(context.device(), context.command_pool(), 1, &cmd);
        return std::unexpected(std::move(error));
    };

    VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr};
    if (auto e = check(vkBeginCommandBuffer(cmd, &beginInfo), "begin one-shot command buffer", ErrorCode::Resource))
    {
        return fail(std::move(*e));
    }

    std::forward<F>(record)(cmd);

    if (auto e = check(vkEndCommandBuffer(cmd), "record one-shot command buffer", ErrorCode::Resource))
    {
        return fail(std::move(*e));
    }

    VkSemaphore timeline = context.submit_timeline();
    {
        std::lock_guard lock(context.queue_mutex());

        // One-shot submits count on the submission timeline too, so the
        // deletion queue keeps draining even on frame-less workloads.
        const std::uint64_t serial = context.advance_submit_serial();
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
                vkQueueSubmit(context.graphics_queue(), 1, &submitInfo, VK_NULL_HANDLE),
                "submit one-shot command buffer",
                ErrorCode::Resource))
        {
            return fail(std::move(*e));
        }
        if (auto e = check(vkQueueWaitIdle(context.graphics_queue()), "wait for one-shot submit", ErrorCode::Resource))
        {
            return fail(std::move(*e));
        }
    }

    vkFreeCommandBuffers(context.device(), context.command_pool(), 1, &cmd);

    // The wait-idle above proves everything submitted so far has completed —
    // a free chance to reclaim deferred handles.
    context.flush_deletion_queue();
    return {};
}
