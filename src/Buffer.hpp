#pragma once
#include <volk.h>
#include <vk_mem_alloc.h>
#include <span>
#include <stdexcept>
#include <memory>
#include <expected>
#include <cstring>
#include <array>
#include "Renderer.hpp"

enum class BufferType {
    VERTEX,
    INDEX,
    UNIFORM,
    STORAGE
};

enum class DataType {
    FLOAT,
    UINT32,
    UINT16,
    INT32
};

class Buffer {
public:
    virtual ~Buffer() = default;

    virtual VkBuffer get() const = 0;
    virtual size_t size() const = 0;
    virtual bool is_dynamic() const { return false; }
    virtual VkBuffer get_for_frame(uint32_t /*frame*/) const { return get(); }
    
    virtual void update(const void* data, size_t size) {
        throw std::runtime_error("Update not supported for this buffer type");
    }

    static std::expected<std::shared_ptr<Buffer>, std::string> create(Renderer& renderer, const void* data, size_t data_size, BufferType type);
};

class StaticBuffer : public Buffer {
public:
    StaticBuffer(VmaAllocator allocator, VkBuffer buffer, VmaAllocation allocation, size_t size)
        : allocator_(allocator), buffer_(buffer), allocation_(allocation), size_(size) {}

    ~StaticBuffer() override {
        if (buffer_ != VK_NULL_HANDLE) {
            vmaDestroyBuffer(allocator_, buffer_, allocation_);
        }
    }

    StaticBuffer(const StaticBuffer&) = delete;
    StaticBuffer& operator=(const StaticBuffer&) = delete;

    VkBuffer get() const override { return buffer_; }
    size_t size() const override { return size_; }

    static std::expected<std::shared_ptr<StaticBuffer>, std::string> create(Renderer& renderer, const void* data, size_t data_size, BufferType type) {
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        switch (type) {
            case BufferType::VERTEX: usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT; break;
            case BufferType::INDEX: usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT; break;
            case BufferType::STORAGE: usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; break;
            default: break;
        }

        VkBufferCreateInfo stagingInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = data_size,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr
        };

        VmaAllocationCreateInfo stagingAllocInfo{};
        stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
        stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

        VkBuffer stagingBuffer;
        VmaAllocation stagingAllocation;
        VmaAllocationInfo stagingAllocInfoOut;

        if (vmaCreateBuffer(renderer.allocator(), &stagingInfo, &stagingAllocInfo, &stagingBuffer, &stagingAllocation, &stagingAllocInfoOut) != VK_SUCCESS) {
            return std::unexpected("Failed to create staging buffer");
        }

        if (data != nullptr && data_size > 0) {
            void* mappedData;
            vmaMapMemory(renderer.allocator(), stagingAllocation, &mappedData);
            std::memcpy(mappedData, data, data_size);
            vmaUnmapMemory(renderer.allocator(), stagingAllocation);
        }

        VkBufferCreateInfo bufferInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = data_size,
            .usage = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr
        };

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        VkBuffer buffer;
        VmaAllocation allocation;
        if (vmaCreateBuffer(renderer.allocator(), &bufferInfo, &allocInfo, &buffer, &allocation, nullptr) != VK_SUCCESS) {
            vmaDestroyBuffer(renderer.allocator(), stagingBuffer, stagingAllocation);
            return std::unexpected("Failed to create device local buffer");
        }

        VkCommandBufferAllocateInfo allocCmdInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr,
            .commandPool = renderer.command_pool(),
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1
        };

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(renderer.device(), &allocCmdInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr
        };

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        VkBufferCopy copyRegion{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = data_size
        };
        vkCmdCopyBuffer(commandBuffer, stagingBuffer, buffer, 1, &copyRegion);

        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = nullptr,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = nullptr,
            .pWaitDstStageMask = nullptr,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffer,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = nullptr
        };

        vkQueueSubmit(renderer.graphics_queue(), 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(renderer.graphics_queue());

        vkFreeCommandBuffers(renderer.device(), renderer.command_pool(), 1, &commandBuffer);
        vmaDestroyBuffer(renderer.allocator(), stagingBuffer, stagingAllocation);

        return std::make_shared<StaticBuffer>(renderer.allocator(), buffer, allocation, data_size);
    }

private:
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    size_t size_ = 0;
};

class DynamicBuffer : public Buffer {
public:
    DynamicBuffer(Renderer& renderer, 
                  std::array<VkBuffer, Renderer::MAX_FRAMES_IN_FLIGHT> buffers, 
                  std::array<VmaAllocation, Renderer::MAX_FRAMES_IN_FLIGHT> allocations, 
                  size_t size,
                  BufferType type)
        : renderer_(renderer), buffers_(buffers), allocations_(allocations), size_(size), type_(type) {}

    ~DynamicBuffer() override {
        for (size_t i = 0; i < Renderer::MAX_FRAMES_IN_FLIGHT; ++i) {
            if (buffers_[i] != VK_NULL_HANDLE) {
                vmaDestroyBuffer(renderer_.allocator(), buffers_[i], allocations_[i]);
            }
        }
    }

    DynamicBuffer(const DynamicBuffer&) = delete;
    DynamicBuffer& operator=(const DynamicBuffer&) = delete;

    VkBuffer get() const override { 
        return buffers_[renderer_.current_frame()]; 
    }
    
    size_t size() const override { return size_; }
    bool is_dynamic() const override { return true; }
    VkBuffer get_for_frame(uint32_t frame) const override { return buffers_[frame]; }
    BufferType buffer_type() const { return type_; }

    void update(const void* data, size_t size) override {
        if (size > size_) {
            throw std::runtime_error("Update size exceeds buffer size");
        }
        uint32_t frame = renderer_.current_frame();
        void* mappedData;
        if (vmaMapMemory(renderer_.allocator(), allocations_[frame], &mappedData) == VK_SUCCESS) {
            std::memcpy(mappedData, data, size);
            vmaUnmapMemory(renderer_.allocator(), allocations_[frame]);
        } else {
            throw std::runtime_error("Failed to map memory for buffer update");
        }
    }

    static std::expected<std::shared_ptr<DynamicBuffer>, std::string> create(Renderer& renderer, const void* data, size_t data_size, BufferType type) {
        VkBufferUsageFlags usage = (type == BufferType::STORAGE) 
            ? VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 
            : VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
        allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

        VkBufferCreateInfo bufferInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = data_size,
            .usage = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr
        };

        std::array<VkBuffer, Renderer::MAX_FRAMES_IN_FLIGHT> buffers{};
        std::array<VmaAllocation, Renderer::MAX_FRAMES_IN_FLIGHT> allocations{};

        for (size_t i = 0; i < Renderer::MAX_FRAMES_IN_FLIGHT; ++i) {
            if (vmaCreateBuffer(renderer.allocator(), &bufferInfo, &allocInfo, &buffers[i], &allocations[i], nullptr) != VK_SUCCESS) {
                return std::unexpected(std::string("Failed to create ") + (type == BufferType::STORAGE ? "storage" : "uniform") + " buffer");
            }

            if (data != nullptr && data_size > 0) {
                void* mappedData;
                vmaMapMemory(renderer.allocator(), allocations[i], &mappedData);
                std::memcpy(mappedData, data, data_size);
                vmaUnmapMemory(renderer.allocator(), allocations[i]);
            }
        }
        return std::make_shared<DynamicBuffer>(renderer, buffers, allocations, data_size, type);
    }

private:
    Renderer& renderer_;
    std::array<VkBuffer, Renderer::MAX_FRAMES_IN_FLIGHT> buffers_ = {VK_NULL_HANDLE};
    std::array<VmaAllocation, Renderer::MAX_FRAMES_IN_FLIGHT> allocations_ = {VK_NULL_HANDLE};
    size_t size_ = 0;
    BufferType type_ = BufferType::UNIFORM;
};

// Keep backward-compatible alias
using UniformBuffer = DynamicBuffer;

inline std::expected<std::shared_ptr<Buffer>, std::string> Buffer::create(Renderer& renderer, const void* data, size_t data_size, BufferType type) {
    if (type == BufferType::UNIFORM || type == BufferType::STORAGE) {
        return DynamicBuffer::create(renderer, data, data_size, type);
    } else {
        return StaticBuffer::create(renderer, data, data_size, type);
    }
}
