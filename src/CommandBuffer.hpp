#pragma once
#include <volk.h>
#include <memory>
#include <vector>
#include <array>
#include <expected>
#include <functional>
#include "Renderer.hpp"
#include "Pipeline.hpp"
#include "Buffer.hpp"
#include "Texture.hpp"
#include "DescriptorSet.hpp"

class CommandBuffer {
public:
    static std::expected<std::shared_ptr<CommandBuffer>, std::string> create(Renderer& renderer) {
        auto cmd = std::shared_ptr<CommandBuffer>(new CommandBuffer(renderer));
        
        VkCommandBufferAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr,
            .commandPool = renderer.command_pool(),
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = Renderer::MAX_FRAMES_IN_FLIGHT
        };

        if (vkAllocateCommandBuffers(renderer.device(), &allocInfo, cmd->command_buffers_.data()) != VK_SUCCESS) {
            return std::unexpected("Failed to allocate command buffers!");
        }
        
        return cmd;
    }

    ~CommandBuffer() {
        if (renderer_.device() != VK_NULL_HANDLE && renderer_.command_pool() != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(renderer_.device(), renderer_.command_pool(), 
                                 Renderer::MAX_FRAMES_IN_FLIGHT, command_buffers_.data());
        }
    }

    CommandBuffer(const CommandBuffer&) = delete;
    CommandBuffer& operator=(const CommandBuffer&) = delete;

    void begin() {
        commands_.clear();
    }

    void beginRendering(const std::vector<float>& clear_color) {
        std::array<float, 4> cc = {0.0f, 0.0f, 0.0f, 1.0f};
        if (clear_color.size() >= 4) {
            cc[0] = clear_color[0]; cc[1] = clear_color[1]; cc[2] = clear_color[2]; cc[3] = clear_color[3];
        }
        commands_.push_back([cc](VkCommandBuffer cmd, Renderer& renderer) {
            VkImageMemoryBarrier barrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = 0,
                .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = renderer.swapchain_images()[renderer.current_image_index()],
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
                0,
                0, nullptr,
                0, nullptr,
                1, &barrier
            );

            if (renderer.depth_image() != VK_NULL_HANDLE) {
                VkImageMemoryBarrier depthBarrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = 0,
                    .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                    .newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = renderer.depth_image(),
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
                    0,
                    0, nullptr,
                    0, nullptr,
                    1, &depthBarrier
                );
            }

            VkRenderingAttachmentInfo colorAttachment{
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .pNext = nullptr,
                .imageView = renderer.swapchain_image_views()[renderer.current_image_index()],
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
                .imageView = renderer.depth_image_view(),
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
                .renderArea = { {0, 0}, renderer.swapchain_extent() },
                .layerCount = 1,
                .viewMask = 0,
                .colorAttachmentCount = 1,
                .pColorAttachments = &colorAttachment,
                .pDepthAttachment = renderer.depth_image_view() != VK_NULL_HANDLE ? &depthAttachment : nullptr,
                .pStencilAttachment = nullptr
            };

            vkCmdBeginRendering(cmd, &renderingInfo);
        });
    }

    void endRendering() {
        commands_.push_back([](VkCommandBuffer cmd, Renderer& renderer) {
            vkCmdEndRendering(cmd);

            VkImageMemoryBarrier barrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                .dstAccessMask = 0,
                .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = renderer.swapchain_images()[renderer.current_image_index()],
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
                0,
                0, nullptr,
                0, nullptr,
                1, &barrier
            );
        });
    }

    void setViewport() {
        commands_.push_back([](VkCommandBuffer cmd, Renderer& renderer) {
            VkViewport viewport{
                .x = 0.0f,
                .y = 0.0f,
                .width = static_cast<float>(renderer.swapchain_extent().width),
                .height = static_cast<float>(renderer.swapchain_extent().height),
                .minDepth = 0.0f,
                .maxDepth = 1.0f
            };
            vkCmdSetViewport(cmd, 0, 1, &viewport);
        });
    }

    void setScissor() {
        commands_.push_back([](VkCommandBuffer cmd, Renderer& renderer) {
            VkRect2D scissor{
                .offset = {0, 0},
                .extent = renderer.swapchain_extent()
            };
            vkCmdSetScissor(cmd, 0, 1, &scissor);
        });
    }

    void bindPipeline(std::shared_ptr<Pipeline> pipeline) {
        commands_.push_back([pipeline](VkCommandBuffer cmd, Renderer& renderer) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->get());
        });
    }

    void bindVertexBuffer(std::shared_ptr<Buffer> buffer) {
        commands_.push_back([buffer](VkCommandBuffer cmd, Renderer& renderer) {
            VkBuffer vertexBuffers[] = {buffer->get()};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
        });
    }

    void bindIndexBuffer(std::shared_ptr<Buffer> buffer) {
        commands_.push_back([buffer](VkCommandBuffer cmd, Renderer& renderer) {
            vkCmdBindIndexBuffer(cmd, buffer->get(), 0, VK_INDEX_TYPE_UINT32);
        });
    }

    void draw(uint32_t vertexCount) {
        commands_.push_back([vertexCount](VkCommandBuffer cmd, Renderer& renderer) {
            vkCmdDraw(cmd, vertexCount, 1, 0, 0);
        });
    }

    void drawIndexed(uint32_t indexCount, uint32_t firstIndex = 0, int32_t vertexOffset = 0) {
        commands_.push_back([indexCount, firstIndex, vertexOffset](VkCommandBuffer cmd, Renderer& renderer) {
            vkCmdDrawIndexed(cmd, indexCount, 1, firstIndex, vertexOffset, 0);
        });
    }

    void drawIndexedInstanced(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex = 0, int32_t vertexOffset = 0) {
        commands_.push_back([indexCount, instanceCount, firstIndex, vertexOffset](VkCommandBuffer cmd, Renderer& renderer) {
            vkCmdDrawIndexed(cmd, indexCount, instanceCount, firstIndex, vertexOffset, 0);
        });
    }

    void pushConstants(std::shared_ptr<Pipeline> pipeline, ShaderStage stage, uint32_t offset, uint32_t size, const void* data) {
        std::vector<uint8_t> buffer(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + size);
        commands_.push_back([pipeline, stage, offset, size, buffer](VkCommandBuffer cmd, Renderer& renderer) {
            VkShaderStageFlags stageFlags = (stage == ShaderStage::VERTEX) ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
            vkCmdPushConstants(cmd, pipeline->layout(), stageFlags, offset, size, buffer.data());
        });
    }

    void bindDescriptorSet(std::shared_ptr<DescriptorSet> descSet,
                           std::shared_ptr<Pipeline> pipeline,
                           uint32_t setIndex) {
        commands_.push_back([descSet, pipeline, setIndex](VkCommandBuffer cmd, Renderer& renderer) {
            VkDescriptorSet set = descSet->get(renderer.current_frame());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipeline->layout(), setIndex, 1, &set, 0, nullptr);
        });
    }

    VkCommandBuffer get() const { 
        return command_buffers_[renderer_.current_frame()]; 
    }

    void execute(VkCommandBuffer vkCmd) {
        for (auto& cmd_func : commands_) {
            cmd_func(vkCmd, renderer_);
        }
    }

private:
    CommandBuffer(Renderer& renderer) : renderer_(renderer) {}

    Renderer& renderer_;
    std::array<VkCommandBuffer, Renderer::MAX_FRAMES_IN_FLIGHT> command_buffers_{};
    std::vector<std::function<void(VkCommandBuffer, Renderer&)>> commands_;
};
