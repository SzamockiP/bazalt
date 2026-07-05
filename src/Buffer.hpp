#pragma once
#include <volk.h>
#include <vk_mem_alloc.h>
#include <span>
#include <stdexcept>
#include <memory>
#include <expected>
#include <cstring>
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
    Buffer() = default;

    Buffer(VmaAllocator allocator, VkBuffer buffer, VmaAllocation allocation, size_t size)
        : allocator_(allocator), buffer_(buffer), allocation_(allocation), size_(size) {}

    ~Buffer() {
        if (buffer_ != VK_NULL_HANDLE) {
            vmaDestroyBuffer(allocator_, buffer_, allocation_);
        }
    }

    Buffer(Buffer&& other) noexcept 
        : allocator_(other.allocator_), buffer_(other.buffer_), allocation_(other.allocation_), size_(other.size_) {
        other.buffer_ = VK_NULL_HANDLE;
    }

    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            if (buffer_ != VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator_, buffer_, allocation_);
            }
            allocator_ = other.allocator_;
            buffer_ = other.buffer_;
            allocation_ = other.allocation_;
            size_ = other.size_;
            other.buffer_ = VK_NULL_HANDLE;
        }
        return *this;
    }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    VkBuffer get() const { return buffer_; }
    size_t size() const { return size_; }

    void update(const void* update_data, size_t update_size) {
        if (update_size > size_) {
            throw std::runtime_error("Update size exceeds buffer size");
        }
        void* mappedData;
        if (vmaMapMemory(allocator_, allocation_, &mappedData) == VK_SUCCESS) {
            std::memcpy(mappedData, update_data, update_size);
            vmaUnmapMemory(allocator_, allocation_);
        } else {
            throw std::runtime_error("Failed to map memory for buffer update");
        }
    }

    static std::expected<std::shared_ptr<Buffer>, std::string> create(Renderer& renderer, const void* data, size_t data_size, BufferType type) {
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bool use_staging = true;
        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        switch (type) {
            case BufferType::VERTEX: usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT; break;
            case BufferType::INDEX: usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT; break;
            case BufferType::UNIFORM: 
                usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT; 
                allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
                allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
                use_staging = false;
                break;
            case BufferType::STORAGE: usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; break;
        }

        if (!use_staging) {
            VkBufferCreateInfo bufferInfo{};
            bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferInfo.size = data_size;
            bufferInfo.usage = usage;
            bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VkBuffer buffer;
            VmaAllocation allocation;
            if (vmaCreateBuffer(renderer.allocator(), &bufferInfo, &allocInfo, &buffer, &allocation, nullptr) != VK_SUCCESS) {
                return std::unexpected("Failed to create host-visible buffer");
            }

            if (data != nullptr && data_size > 0) {
                void* mappedData;
                vmaMapMemory(renderer.allocator(), allocation, &mappedData);
                std::memcpy(mappedData, data, data_size);
                vmaUnmapMemory(renderer.allocator(), allocation);
            }
            return std::make_shared<Buffer>(renderer.allocator(), buffer, allocation, data_size);
        }

        // Create staging buffer
        VkBufferCreateInfo stagingInfo{};
        stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        stagingInfo.size = data_size;
        stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo stagingAllocInfo{};
        stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
        stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

        VkBuffer stagingBuffer;
        VmaAllocation stagingAllocation;
        VmaAllocationInfo stagingAllocInfoOut;

        if (vmaCreateBuffer(renderer.allocator(), &stagingInfo, &stagingAllocInfo, &stagingBuffer, &stagingAllocation, &stagingAllocInfoOut) != VK_SUCCESS) {
            return std::unexpected("Failed to create staging buffer");
        }

        // Copy data to staging buffer
        if (data != nullptr && data_size > 0) {
            void* mappedData;
            vmaMapMemory(renderer.allocator(), stagingAllocation, &mappedData);
            std::memcpy(mappedData, data, data_size);
            vmaUnmapMemory(renderer.allocator(), stagingAllocation);
        }

        // Create device-local buffer
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = data_size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer buffer;
        VmaAllocation allocation;
        if (vmaCreateBuffer(renderer.allocator(), &bufferInfo, &allocInfo, &buffer, &allocation, nullptr) != VK_SUCCESS) {
            vmaDestroyBuffer(renderer.allocator(), stagingBuffer, stagingAllocation);
            return std::unexpected("Failed to create device local buffer");
        }

        // Copy staging to device local
        VkCommandBufferAllocateInfo allocCmdInfo{};
        allocCmdInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocCmdInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocCmdInfo.commandPool = renderer.command_pool();
        allocCmdInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(renderer.device(), &allocCmdInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        VkBufferCopy copyRegion{};
        copyRegion.size = data_size;
        vkCmdCopyBuffer(commandBuffer, stagingBuffer, buffer, 1, &copyRegion);

        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(renderer.graphics_queue(), 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(renderer.graphics_queue());

        vkFreeCommandBuffers(renderer.device(), renderer.command_pool(), 1, &commandBuffer);
        vmaDestroyBuffer(renderer.allocator(), stagingBuffer, stagingAllocation);

        return std::make_shared<Buffer>(renderer.allocator(), buffer, allocation, data_size);
    }

private:
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    size_t size_ = 0;
};
