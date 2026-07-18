#pragma once
#include <volk.h>
#include <memory>
#include <vector>
#include <array>
#include <expected>
#include <functional>
#include "Context.hpp"
#include "RenderTarget.hpp"
#include "Pipeline.hpp"
#include "Buffer.hpp"
#include "DescriptorSet.hpp"

// Records commands once and replays them every submit.
//
// The recorded lambdas take a FrameContext rather than a SwapchainRenderer&.
// That single change is what lets the same command buffer be replayed against a
// window, an offscreen image, or (later) a compute-only submit: this file no
// longer knows that swapchains exist.
class CommandBuffer {
public:
    // Takes a Context, not a renderer: command buffers are a device resource and
    // have nothing to do with presentation. This is what lets a headless Context
    // with no renderer at all record commands.
    static std::expected<std::shared_ptr<CommandBuffer>, Error> create(Context& context) {
        auto ctx = context.shared_from_this();
        auto cmd = std::shared_ptr<CommandBuffer>(new CommandBuffer(ctx));
        cmd->command_buffers_.resize(ctx->frames_in_flight(), VK_NULL_HANDLE);

        VkCommandBufferAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr,
            .commandPool = ctx->command_pool(),
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = ctx->frames_in_flight()
        };

        if (auto e = check(vkAllocateCommandBuffers(ctx->device(), &allocInfo, cmd->command_buffers_.data()),
                           "allocate command buffers", ErrorCode::Resource)) {
            return std::unexpected(*e);
        }

        return cmd;
    }

    // Deferred: a per-frame VkCommandBuffer may still be executing when the
    // Python object is dropped.
    ~CommandBuffer() {
        if (context_) {
            context_->defer_destroy(
                [device = context_->device(), pool = context_->command_pool(),
                 buffers = std::move(command_buffers_)] {
                    vkFreeCommandBuffers(device, pool,
                                         static_cast<uint32_t>(buffers.size()),
                                         buffers.data());
                });
        }
    }

    CommandBuffer(const CommandBuffer&) = delete;
    CommandBuffer& operator=(const CommandBuffer&) = delete;

    CommandBuffer& begin() {
        commands_.clear();
        used_sets_.clear();
        return *this;
    }

    // The target is explicit. It used to default to "the swapchain" implicitly,
    // which made presentation a special case dressed up as the default and left
    // no way to name anything else. Naming what you draw into costs one token
    // and buys one rule that holds everywhere.
    CommandBuffer& begin_rendering(std::shared_ptr<RenderTarget> target, const std::vector<float>& clear_color) {
        std::array<float, 4> cc = {0.0f, 0.0f, 0.0f, 1.0f};
        if (clear_color.size() >= 4) {
            cc[0] = clear_color[0]; cc[1] = clear_color[1]; cc[2] = clear_color[2]; cc[3] = clear_color[3];
        }

        commands_.push_back([cc, target](VkCommandBuffer cmd, const FrameContext& frame) {
            RenderTarget* rt = target.get();

            // Every colour attachment enters COLOR_ATTACHMENT_OPTIMAL. UNDEFINED
            // as the source: contents are cleared each pass anyway.
            for (uint32_t i = 0; i < rt->color_count(); ++i) {
                VkImageMemoryBarrier barrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = 0,
                    .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                    .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = rt->color_image(i),
                    .subresourceRange = {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1
                    }
                };

                vkCmdPipelineBarrier(
                    cmd,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    0, 0, nullptr, 0, nullptr, 1, &barrier
                );
            }

            if (rt->depth_image() != VK_NULL_HANDLE) {
                VkImageMemoryBarrier depthBarrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = 0,
                    .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                    .newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = rt->depth_image(),
                    .subresourceRange = {
                        .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1
                    }
                };

                vkCmdPipelineBarrier(
                    cmd,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                    0, 0, nullptr, 0, nullptr, 1, &depthBarrier
                );
            }

            std::vector<VkRenderingAttachmentInfo> colorAttachments;
            colorAttachments.reserve(rt->color_count());
            for (uint32_t i = 0; i < rt->color_count(); ++i) {
                colorAttachments.push_back({
                    .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                    .pNext = nullptr,
                    .imageView = rt->color_view(i),
                    .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    .resolveMode = VK_RESOLVE_MODE_NONE,
                    .resolveImageView = VK_NULL_HANDLE,
                    .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                    .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                    // One clear colour for every attachment; per-attachment
                    // clears can arrive additively if something needs them.
                    .clearValue = { .color = { { cc[0], cc[1], cc[2], cc[3] } } }
                });
            }

            VkRenderingAttachmentInfo depthAttachment{
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .pNext = nullptr,
                .imageView = rt->depth_view(),
                .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                .resolveMode = VK_RESOLVE_MODE_NONE,
                .resolveImageView = VK_NULL_HANDLE,
                .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                // A depth that will be consumed (shadow maps) must be stored;
                // the swapchain's scratch depth keeps DONT_CARE.
                .storeOp = rt->depth_final_layout() == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL
                    ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE,
                .clearValue = { .depthStencil = { 1.0f, 0 } }
            };

            VkRenderingInfo renderingInfo{
                .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                .pNext = nullptr,
                .flags = 0,
                .renderArea = { {0, 0}, rt->extent() },
                .layerCount = 1,
                .viewMask = 0,
                .colorAttachmentCount = static_cast<uint32_t>(colorAttachments.size()),
                .pColorAttachments = colorAttachments.empty() ? nullptr : colorAttachments.data(),
                .pDepthAttachment = rt->depth_view() != VK_NULL_HANDLE ? &depthAttachment : nullptr,
                .pStencilAttachment = nullptr
            };

            vkCmdBeginRendering(cmd, &renderingInfo);

            // Emitted automatically: set_viewport()/set_scissor() took no arguments
            // and silently read the swapchain, which is magic â€” just less legible
            // than doing it here. set_viewport(x, y, w, h) remains for the cases
            // that genuinely want something other than the whole target.
            VkViewport viewport{
                .x = 0.0f,
                .y = 0.0f,
                .width = static_cast<float>(rt->extent().width),
                .height = static_cast<float>(rt->extent().height),
                .minDepth = 0.0f,
                .maxDepth = 1.0f
            };
            vkCmdSetViewport(cmd, 0, 1, &viewport);

            VkRect2D scissor{ .offset = {0, 0}, .extent = rt->extent() };
            vkCmdSetScissor(cmd, 0, 1, &scissor);
        });
        return *this;
    }

    CommandBuffer& end_rendering(std::shared_ptr<RenderTarget> target) {
        commands_.push_back([target](VkCommandBuffer cmd, const FrameContext& frame) {
            vkCmdEndRendering(cmd);

            // Every colour attachment retires to the target's final layout.
            // (Was VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, unconditionally, on colour 0
            // only â€” that one constant is why nothing but a swapchain could ever
            // be drawn into.)
            for (uint32_t i = 0; i < target->color_count(); ++i) {
                VkImageMemoryBarrier barrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    .dstAccessMask = 0,
                    .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    .newLayout = target->final_layout(),
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = target->color_image(i),
                    .subresourceRange = {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1
                    }
                };

                vkCmdPipelineBarrier(
                    cmd,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                    0, 0, nullptr, 0, nullptr, 1, &barrier
                );
            }

            // Depth retires to its own final layout when it will be consumed
            // (offscreen: SHADER_READ_ONLY, which is what makes `target.depth`
            // sampleable). The swapchain's depth stays put â€” no barrier.
            if (target->depth_image() != VK_NULL_HANDLE &&
                target->depth_final_layout() != VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL) {
                VkImageMemoryBarrier depthBarrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                    .newLayout = target->depth_final_layout(),
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = target->depth_image(),
                    .subresourceRange = {
                        .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1
                    }
                };

                vkCmdPipelineBarrier(
                    cmd,
                    VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                    0, 0, nullptr, 0, nullptr, 1, &depthBarrier
                );
            }

            // Runs at execute() time, inside a real submit â€” so the target learns
            // its images have left UNDEFINED exactly when that becomes true, and a
            // recorded-but-never-submitted command buffer marks nothing.
            target->on_rendering_recorded();
        });
        return *this;
    }

    // Explicit override for split-screen and similar. The no-argument version is
    // gone: begin_rendering already covers the whole-target case.
    CommandBuffer& set_viewport(float x, float y, float width, float height) {
        commands_.push_back([x, y, width, height](VkCommandBuffer cmd, const FrameContext&) {
            VkViewport viewport{
                .x = x, .y = y, .width = width, .height = height,
                .minDepth = 0.0f, .maxDepth = 1.0f
            };
            vkCmdSetViewport(cmd, 0, 1, &viewport);
        });
        return *this;
    }

    CommandBuffer& set_scissor(std::int32_t x, std::int32_t y, std::uint32_t width, std::uint32_t height) {
        commands_.push_back([x, y, width, height](VkCommandBuffer cmd, const FrameContext&) {
            VkRect2D scissor{ .offset = {x, y}, .extent = {width, height} };
            vkCmdSetScissor(cmd, 0, 1, &scissor);
        });
        return *this;
    }

    CommandBuffer& bind_pipeline(std::shared_ptr<Pipeline> pipeline) {
        commands_.push_back([pipeline](VkCommandBuffer cmd, const FrameContext&) {
            vkCmdBindPipeline(cmd, pipeline->bind_point(), pipeline->get());
        });
        return *this;
    }

    CommandBuffer& bind_vertex_buffer(std::shared_ptr<Buffer> buffer) {
        commands_.push_back([buffer](VkCommandBuffer cmd, const FrameContext&) {
            VkBuffer vertexBuffers[] = {buffer->get()};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
        });
        return *this;
    }

    CommandBuffer& bind_index_buffer(std::shared_ptr<Buffer> buffer) {
        commands_.push_back([buffer](VkCommandBuffer cmd, const FrameContext&) {
            // Derived from the buffer rather than hardcoded to UINT32: create_buffer
            // accepts UINT16 indices, which used to be read back at half count.
            vkCmdBindIndexBuffer(cmd, buffer->get(), 0, buffer->index_type());
        });
        return *this;
    }

    CommandBuffer& draw(uint32_t vertexCount) {
        commands_.push_back([vertexCount](VkCommandBuffer cmd, const FrameContext&) {
            vkCmdDraw(cmd, vertexCount, 1, 0, 0);
        });
        return *this;
    }

    CommandBuffer& draw_indexed(uint32_t indexCount, uint32_t firstIndex = 0, int32_t vertexOffset = 0) {
        commands_.push_back([indexCount, firstIndex, vertexOffset](VkCommandBuffer cmd, const FrameContext&) {
            vkCmdDrawIndexed(cmd, indexCount, 1, firstIndex, vertexOffset, 0);
        });
        return *this;
    }

    CommandBuffer& draw_indexed_instanced(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex = 0, int32_t vertexOffset = 0) {
        commands_.push_back([indexCount, instanceCount, firstIndex, vertexOffset](VkCommandBuffer cmd, const FrameContext&) {
            vkCmdDrawIndexed(cmd, indexCount, instanceCount, firstIndex, vertexOffset, 0);
        });
        return *this;
    }

    // No stage argument: the Pipeline already knows which stages its push constant
    // range covers, so passing a mismatched one was a validation error for no gain.
    CommandBuffer& push_constants(std::shared_ptr<Pipeline> pipeline, uint32_t offset, uint32_t size, const void* data) {
        std::vector<uint8_t> buffer(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + size);
        commands_.push_back([pipeline, offset, size, buffer](VkCommandBuffer cmd, const FrameContext&) {
            vkCmdPushConstants(cmd, pipeline->layout(), pipeline->push_constant_stages(),
                               offset, size, buffer.data());
        });
        return *this;
    }

    CommandBuffer& bind_descriptor_set(std::shared_ptr<DescriptorSet> descSet,
                           std::shared_ptr<Pipeline> pipeline,
                           uint32_t setIndex) {
        // Remembered so submit paths can walk the images this recording
        // references and wait for their (async) uploads — residency is a
        // per-command-buffer question, not a global one, or a loading screen
        // would serialize behind its own cargo.
        used_sets_.push_back(descSet);
        commands_.push_back([descSet, pipeline, setIndex](VkCommandBuffer cmd, const FrameContext& frame) {
            VkDescriptorSet set = descSet->get(frame.frame_index);
            vkCmdBindDescriptorSets(cmd, pipeline->bind_point(),
                pipeline->layout(), setIndex, 1, &set, 0, nullptr);
        });
        return *this;
    }

    const std::vector<std::shared_ptr<DescriptorSet>>& used_sets() const {
        return used_sets_;
    }

    VkCommandBuffer get(std::uint32_t frame_index) const {
        return command_buffers_[frame_index];
    }

    void execute(VkCommandBuffer vkCmd, const FrameContext& frame) {
        for (auto& cmd_func : commands_) {
            cmd_func(vkCmd, frame);
        }
    }

private:
    CommandBuffer(std::shared_ptr<Context> context) : context_(context) {}

    std::shared_ptr<Context> context_;
    std::vector<VkCommandBuffer> command_buffers_;
    std::vector<std::function<void(VkCommandBuffer, const FrameContext&)>> commands_;
    std::vector<std::shared_ptr<DescriptorSet>> used_sets_;
};
