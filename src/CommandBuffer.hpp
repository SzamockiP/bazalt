#pragma once
#include <volk.h>
#include <memory>
#include <vector>
#include <array>
#include <expected>
#include "Renderer.hpp"
#include "Pipeline.hpp"
#include "Buffer.hpp"

class CommandBuffer {
public:
    static std::expected<std::shared_ptr<CommandBuffer>, std::string> create(Renderer& renderer) {
        auto cmd = std::shared_ptr<CommandBuffer>(new CommandBuffer(renderer));
        
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = renderer.command_pool();
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = Renderer::MAX_FRAMES_IN_FLIGHT;

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
        current_cmd_ = command_buffers_[renderer_.current_frame()];
        
        vkResetCommandBuffer(current_cmd_, 0);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        
        if (vkBeginCommandBuffer(current_cmd_, &beginInfo) != VK_SUCCESS) {
            // It's a method returning void right now, usually we just log or throw, 
            // but since it's a runtime command buffer recording issue, we can keep exception or change it.
            // I'll leave the throw here because it's void begin(). Changing all methods to expected is too much.
            throw std::runtime_error("Failed to begin recording command buffer!");
        }
    }

    void beginRendering(const std::vector<float>& clear_color) {
        // Transition swapchain image to color attachment optimal
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = renderer_.swapchain_images()[renderer_.current_image_index()];
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        vkCmdPipelineBarrier(
            current_cmd_,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );

        if (renderer_.depth_image() != VK_NULL_HANDLE) {
            VkImageMemoryBarrier depthBarrier{};
            depthBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            depthBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            depthBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            depthBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            depthBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            depthBarrier.image = renderer_.depth_image();
            depthBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            depthBarrier.subresourceRange.baseMipLevel = 0;
            depthBarrier.subresourceRange.levelCount = 1;
            depthBarrier.subresourceRange.baseArrayLayer = 0;
            depthBarrier.subresourceRange.layerCount = 1;
            depthBarrier.srcAccessMask = 0;
            depthBarrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

            vkCmdPipelineBarrier(
                current_cmd_,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                0,
                0, nullptr,
                0, nullptr,
                1, &depthBarrier
            );
        }

        VkRenderingAttachmentInfo colorAttachment{};
        colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttachment.imageView = renderer_.swapchain_image_views()[renderer_.current_image_index()];
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        
        if (clear_color.size() >= 4) {
            colorAttachment.clearValue.color = {clear_color[0], clear_color[1], clear_color[2], clear_color[3]};
        } else {
            colorAttachment.clearValue.color = {0.0f, 0.0f, 0.0f, 1.0f};
        }

        VkRenderingAttachmentInfo depthAttachment{};
        if (renderer_.depth_image_view() != VK_NULL_HANDLE) {
            depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depthAttachment.imageView = renderer_.depth_image_view();
            depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAttachment.clearValue.depthStencil = {1.0f, 0};
        }

        VkRenderingInfo renderingInfo{};
        renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderingInfo.renderArea.offset = {0, 0};
        renderingInfo.renderArea.extent = renderer_.swapchain_extent();
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachment;
        if (renderer_.depth_image_view() != VK_NULL_HANDLE) {
            renderingInfo.pDepthAttachment = &depthAttachment;
        }

        vkCmdBeginRendering(current_cmd_, &renderingInfo);
    }

    void endRendering() {
        vkCmdEndRendering(current_cmd_);

        // Transition swapchain image to present optimal
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = renderer_.swapchain_images()[renderer_.current_image_index()];
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = 0;

        vkCmdPipelineBarrier(
            current_cmd_,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );
    }

    void setViewport() {
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(renderer_.swapchain_extent().width);
        viewport.height = static_cast<float>(renderer_.swapchain_extent().height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(current_cmd_, 0, 1, &viewport);
    }

    void setScissor() {
        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = renderer_.swapchain_extent();
        vkCmdSetScissor(current_cmd_, 0, 1, &scissor);
    }

    void bindPipeline(std::shared_ptr<Pipeline> pipeline) {
        vkCmdBindPipeline(current_cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->get());
    }

    void bindVertexBuffer(std::shared_ptr<Buffer> buffer) {
        VkBuffer vertexBuffers[] = {buffer->get()};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(current_cmd_, 0, 1, vertexBuffers, offsets);
    }

    void bindIndexBuffer(std::shared_ptr<Buffer> buffer) {
        vkCmdBindIndexBuffer(current_cmd_, buffer->get(), 0, VK_INDEX_TYPE_UINT32);
    }

    void draw(uint32_t vertexCount) {
        vkCmdDraw(current_cmd_, vertexCount, 1, 0, 0);
    }

    void drawIndexed(uint32_t indexCount) {
        vkCmdDrawIndexed(current_cmd_, indexCount, 1, 0, 0, 0);
    }

    void pushConstants(std::shared_ptr<Pipeline> pipeline, ShaderStage stage, uint32_t offset, uint32_t size, const void* data) {
        VkShaderStageFlags stageFlags = (stage == ShaderStage::VERTEX) ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
        vkCmdPushConstants(current_cmd_, pipeline->layout(), stageFlags, offset, size, data);
    }

    void bindUniformBuffer(uint32_t binding, std::shared_ptr<Buffer> buffer, std::shared_ptr<Pipeline> pipeline) {
        VkDescriptorSet descSet = pipeline->descriptor_set();
        if (descSet == VK_NULL_HANDLE) {
            throw std::runtime_error("Pipeline has no descriptor set allocated");
        }

        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = buffer->get();
        bufferInfo.offset = 0;
        bufferInfo.range = buffer->size();

        VkWriteDescriptorSet descriptorWrite{};
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = descSet;
        descriptorWrite.dstBinding = binding;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pBufferInfo = &bufferInfo;

        vkUpdateDescriptorSets(renderer_.device(), 1, &descriptorWrite, 0, nullptr);

        vkCmdBindDescriptorSets(current_cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->layout(), 0, 1, &descSet, 0, nullptr);
    }

    VkCommandBuffer get() const { return current_cmd_; }

private:
    CommandBuffer(Renderer& renderer) : renderer_(renderer) {}

    Renderer& renderer_;
    std::array<VkCommandBuffer, Renderer::MAX_FRAMES_IN_FLIGHT> command_buffers_{};
    VkCommandBuffer current_cmd_ = VK_NULL_HANDLE;
};
