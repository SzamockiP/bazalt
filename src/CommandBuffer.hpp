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
#include "Texture.hpp"
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

    ~CommandBuffer() {
        if (context_) {
            vkFreeCommandBuffers(context_->device(), context_->command_pool(),
                                 static_cast<uint32_t>(command_buffers_.size()),
                                 command_buffers_.data());
        }
    }

    CommandBuffer(const CommandBuffer&) = delete;
    CommandBuffer& operator=(const CommandBuffer&) = delete;

    void begin() {
        commands_.clear();
    }

    // The target is explicit. It used to default to "the swapchain" implicitly,
    // which made presentation a special case dressed up as the default and left
    // no way to name anything else. Naming what you draw into costs one token
    // and buys one rule that holds everywhere.
    void begin_rendering(std::shared_ptr<RenderTarget> target, const std::vector<float>& clear_color) {
        std::array<float, 4> cc = {0.0f, 0.0f, 0.0f, 1.0f};
        if (clear_color.size() >= 4) {
            cc[0] = clear_color[0]; cc[1] = clear_color[1]; cc[2] = clear_color[2]; cc[3] = clear_color[3];
        }

        commands_.push_back([cc, target](VkCommandBuffer cmd, const FrameContext& frame) {
            RenderTarget* rt = target.get();

            VkImageMemoryBarrier barrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = 0,
                .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = rt->color_image(0),
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

            VkRenderingAttachmentInfo colorAttachment{
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .pNext = nullptr,
                .imageView = rt->color_view(0),
                .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .resolveMode = VK_RESOLVE_MODE_NONE,
                .resolveImageView = VK_NULL_HANDLE,
                .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .clearValue = { .color = { { cc[0], cc[1], cc[2], cc[3] } } }
            };

            VkRenderingAttachmentInfo depthAttachment{
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .pNext = nullptr,
                .imageView = rt->depth_view(),
                .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                .resolveMode = VK_RESOLVE_MODE_NONE,
                .resolveImageView = VK_NULL_HANDLE,
                .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                .clearValue = { .depthStencil = { 1.0f, 0 } }
            };

            VkRenderingInfo renderingInfo{
                .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                .pNext = nullptr,
                .flags = 0,
                .renderArea = { {0, 0}, rt->extent() },
                .layerCount = 1,
                .viewMask = 0,
                .colorAttachmentCount = 1,
                .pColorAttachments = &colorAttachment,
                .pDepthAttachment = rt->depth_view() != VK_NULL_HANDLE ? &depthAttachment : nullptr,
                .pStencilAttachment = nullptr
            };

            vkCmdBeginRendering(cmd, &renderingInfo);

            // Emitted automatically: set_viewport()/set_scissor() took no arguments
            // and silently read the swapchain, which is magic — just less legible
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
    }

    void end_rendering(std::shared_ptr<RenderTarget> target) {
        commands_.push_back([target](VkCommandBuffer cmd, const FrameContext& frame) {
            vkCmdEndRendering(cmd);

            VkImageMemoryBarrier barrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                .dstAccessMask = 0,
                .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                // Was VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, unconditionally. That one
                // constant is why nothing but a swapchain could ever be drawn into.
                .newLayout = target->final_layout(),
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = target->color_image(0),
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

            // Runs at execute() time, inside a real submit — so the target learns
            // its image has left UNDEFINED exactly when that becomes true, and a
            // recorded-but-never-submitted command buffer marks nothing.
            target->on_rendering_recorded();
        });
    }

    // Explicit override for split-screen and similar. The no-argument version is
    // gone: begin_rendering already covers the whole-target case.
    void set_viewport(float x, float y, float width, float height) {
        commands_.push_back([x, y, width, height](VkCommandBuffer cmd, const FrameContext&) {
            VkViewport viewport{
                .x = x, .y = y, .width = width, .height = height,
                .minDepth = 0.0f, .maxDepth = 1.0f
            };
            vkCmdSetViewport(cmd, 0, 1, &viewport);
        });
    }

    void set_scissor(std::int32_t x, std::int32_t y, std::uint32_t width, std::uint32_t height) {
        commands_.push_back([x, y, width, height](VkCommandBuffer cmd, const FrameContext&) {
            VkRect2D scissor{ .offset = {x, y}, .extent = {width, height} };
            vkCmdSetScissor(cmd, 0, 1, &scissor);
        });
    }

    void bind_pipeline(std::shared_ptr<Pipeline> pipeline) {
        commands_.push_back([pipeline](VkCommandBuffer cmd, const FrameContext&) {
            vkCmdBindPipeline(cmd, pipeline->bind_point(), pipeline->get());
        });
    }

    void bind_vertex_buffer(std::shared_ptr<Buffer> buffer) {
        commands_.push_back([buffer](VkCommandBuffer cmd, const FrameContext&) {
            VkBuffer vertexBuffers[] = {buffer->get()};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
        });
    }

    void bind_index_buffer(std::shared_ptr<Buffer> buffer) {
        commands_.push_back([buffer](VkCommandBuffer cmd, const FrameContext&) {
            // Derived from the buffer rather than hardcoded to UINT32: create_buffer
            // accepts UINT16 indices, which used to be read back at half count.
            vkCmdBindIndexBuffer(cmd, buffer->get(), 0, buffer->index_type());
        });
    }

    void draw(uint32_t vertexCount) {
        commands_.push_back([vertexCount](VkCommandBuffer cmd, const FrameContext&) {
            vkCmdDraw(cmd, vertexCount, 1, 0, 0);
        });
    }

    void draw_indexed(uint32_t indexCount, uint32_t firstIndex = 0, int32_t vertexOffset = 0) {
        commands_.push_back([indexCount, firstIndex, vertexOffset](VkCommandBuffer cmd, const FrameContext&) {
            vkCmdDrawIndexed(cmd, indexCount, 1, firstIndex, vertexOffset, 0);
        });
    }

    void draw_indexed_instanced(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex = 0, int32_t vertexOffset = 0) {
        commands_.push_back([indexCount, instanceCount, firstIndex, vertexOffset](VkCommandBuffer cmd, const FrameContext&) {
            vkCmdDrawIndexed(cmd, indexCount, instanceCount, firstIndex, vertexOffset, 0);
        });
    }

    // No stage argument: the Pipeline already knows which stages its push constant
    // range covers, so passing a mismatched one was a validation error for no gain.
    void push_constants(std::shared_ptr<Pipeline> pipeline, uint32_t offset, uint32_t size, const void* data) {
        std::vector<uint8_t> buffer(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + size);
        commands_.push_back([pipeline, offset, size, buffer](VkCommandBuffer cmd, const FrameContext&) {
            vkCmdPushConstants(cmd, pipeline->layout(), pipeline->push_constant_stages(),
                               offset, size, buffer.data());
        });
    }

    void bind_descriptor_set(std::shared_ptr<DescriptorSet> descSet,
                           std::shared_ptr<Pipeline> pipeline,
                           uint32_t setIndex) {
        commands_.push_back([descSet, pipeline, setIndex](VkCommandBuffer cmd, const FrameContext& frame) {
            VkDescriptorSet set = descSet->get(frame.frame_index);
            vkCmdBindDescriptorSets(cmd, pipeline->bind_point(),
                pipeline->layout(), setIndex, 1, &set, 0, nullptr);
        });
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
};
