#pragma once
#include <volk.h>
#include <vk_mem_alloc.h>
#include <span>
#include <string>
#include <memory>
#include <expected>
#include <cstring>
#include <array>
#include "Context.hpp"
#include "ImmediateSubmit.hpp"

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

enum class MemoryUsage {
    STATIC,
    DYNAMIC
};

class Buffer {
public:
    virtual ~Buffer() = default;

    virtual VkBuffer get() const = 0;
    virtual size_t size() const = 0;
    virtual bool is_dynamic() const { return false; }
    virtual VkBuffer get_for_frame(uint32_t /*frame*/) const { return get(); }

    // Fails through the unified Error channel, not a raw exception: at the
    // pybind boundary this surfaces as bz.ResourceError, so `except BazaltError`
    // actually catches it.
    virtual std::expected<void, Error> update(const void* /*data*/, size_t /*size*/) {
        return std::unexpected(err_resource(
            "update() is only supported on DYNAMIC buffers; "
            "create the buffer with MemoryUsage.DYNAMIC instead"));
    }

    // Remembered so bindIndexBuffer doesn't have to assume. It used to hardcode
    // VK_INDEX_TYPE_UINT32 while create_buffer happily accepted UINT16 indices,
    // which were then read back at half the count with no error.
    DataType data_type() const { return data_type_; }
    void set_data_type(DataType type) { data_type_ = type; }

    VkIndexType index_type() const {
        return data_type_ == DataType::UINT16 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
    }

    static std::expected<std::shared_ptr<Buffer>, Error> create(Context& context, const void* data, size_t data_size, BufferType type, MemoryUsage usage);

protected:
    DataType data_type_ = DataType::FLOAT;
};

class StaticBuffer : public Buffer {
public:
    StaticBuffer(std::shared_ptr<Context> context, VkBuffer buffer, VmaAllocation allocation, size_t size)
        : context_(context), buffer_(buffer), allocation_(allocation), size_(size) {}

    ~StaticBuffer() override {
        if (buffer_ != VK_NULL_HANDLE && context_) {
            vmaDestroyBuffer(context_->allocator(), buffer_, allocation_);
        }
    }

    StaticBuffer(const StaticBuffer&) = delete;
    StaticBuffer& operator=(const StaticBuffer&) = delete;

    VkBuffer get() const override { return buffer_; }
    size_t size() const override { return size_; }

    static std::expected<std::shared_ptr<StaticBuffer>, Error> create(Context& context, const void* data, size_t data_size, BufferType type) {
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

        if (auto e = check(vmaCreateBuffer(context.allocator(), &stagingInfo, &stagingAllocInfo, &stagingBuffer, &stagingAllocation, &stagingAllocInfoOut),
                           "create staging buffer", ErrorCode::Resource)) {
            return std::unexpected(*e);
        }

        if (data != nullptr && data_size > 0) {
            void* mappedData;
            if (auto e = check(vmaMapMemory(context.allocator(), stagingAllocation, &mappedData),
                               "map staging buffer memory", ErrorCode::Resource)) {
                vmaDestroyBuffer(context.allocator(), stagingBuffer, stagingAllocation);
                return std::unexpected(*e);
            }
            std::memcpy(mappedData, data, data_size);
            vmaUnmapMemory(context.allocator(), stagingAllocation);
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
        if (auto e = check(vmaCreateBuffer(context.allocator(), &bufferInfo, &allocInfo, &buffer, &allocation, nullptr),
                           "create device local buffer", ErrorCode::Resource)) {
            vmaDestroyBuffer(context.allocator(), stagingBuffer, stagingAllocation);
            return std::unexpected(*e);
        }

        auto submitted = immediate_submit(context, [&](VkCommandBuffer cmd) {
            VkBufferCopy copyRegion{
                .srcOffset = 0,
                .dstOffset = 0,
                .size = data_size
            };
            vkCmdCopyBuffer(cmd, stagingBuffer, buffer, 1, &copyRegion);
        });

        vmaDestroyBuffer(context.allocator(), stagingBuffer, stagingAllocation);
        if (!submitted) {
            vmaDestroyBuffer(context.allocator(), buffer, allocation);
            return std::unexpected(submitted.error());
        }

        return std::make_shared<StaticBuffer>(context.shared_from_this(), buffer, allocation, data_size);
    }

private:
    std::shared_ptr<Context> context_;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    size_t size_ = 0;
};

class DynamicBuffer : public Buffer {
public:
    DynamicBuffer(std::shared_ptr<Context> context, 
                  std::array<VkBuffer, Context::MAX_FRAMES_IN_FLIGHT> buffers, 
                  std::array<VmaAllocation, Context::MAX_FRAMES_IN_FLIGHT> allocations, 
                  size_t size,
                  BufferType type)
        : context_(context), buffers_(buffers), allocations_(allocations), size_(size), type_(type) {}

    ~DynamicBuffer() override {
        if (context_) {
            for (size_t i = 0; i < Context::MAX_FRAMES_IN_FLIGHT; ++i) {
                if (buffers_[i] != VK_NULL_HANDLE) {
                    vmaDestroyBuffer(context_->allocator(), buffers_[i], allocations_[i]);
                }
            }
        }
    }

    DynamicBuffer(const DynamicBuffer&) = delete;
    DynamicBuffer& operator=(const DynamicBuffer&) = delete;

    VkBuffer get() const override { 
        return buffers_[context_->current_frame()]; 
    }
    
    size_t size() const override { return size_; }
    bool is_dynamic() const override { return true; }
    VkBuffer get_for_frame(uint32_t frame) const override { return buffers_[frame]; }
    BufferType buffer_type() const { return type_; }

    std::expected<void, Error> update(const void* data, size_t size) override {
        if (size > size_) {
            return std::unexpected(err_resource(
                "Update of " + std::to_string(size) + " bytes exceeds the buffer size of " +
                std::to_string(size_) + " bytes"));
        }
        uint32_t frame = context_->current_frame();
        void* mappedData;
        if (auto e = check(vmaMapMemory(context_->allocator(), allocations_[frame], &mappedData),
                           "map dynamic buffer memory for update", ErrorCode::Resource)) {
            return std::unexpected(*e);
        }
        std::memcpy(mappedData, data, size);
        vmaUnmapMemory(context_->allocator(), allocations_[frame]);
        return {};
    }

    static std::expected<std::shared_ptr<DynamicBuffer>, Error> create(Context& context, const void* data, size_t data_size, BufferType type) {
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

        std::array<VkBuffer, Context::MAX_FRAMES_IN_FLIGHT> buffers{};
        std::array<VmaAllocation, Context::MAX_FRAMES_IN_FLIGHT> allocations{};

        for (size_t i = 0; i < Context::MAX_FRAMES_IN_FLIGHT; ++i) {
            if (auto e = check(vmaCreateBuffer(context.allocator(), &bufferInfo, &allocInfo, &buffers[i], &allocations[i], nullptr),
                               std::string("create ") + (type == BufferType::STORAGE ? "storage" : "uniform") + " buffer",
                               ErrorCode::Resource)) {
                for (size_t j = 0; j < i; ++j) {
                    vmaDestroyBuffer(context.allocator(), buffers[j], allocations[j]);
                }
                return std::unexpected(*e);
            }

            if (data != nullptr && data_size > 0) {
                void* mappedData;
                vmaMapMemory(context.allocator(), allocations[i], &mappedData);
                std::memcpy(mappedData, data, data_size);
                vmaUnmapMemory(context.allocator(), allocations[i]);
            }
        }
        return std::make_shared<DynamicBuffer>(context.shared_from_this(), buffers, allocations, data_size, type);
    }

private:
    std::shared_ptr<Context> context_;
    std::array<VkBuffer, Context::MAX_FRAMES_IN_FLIGHT> buffers_ = {VK_NULL_HANDLE};
    std::array<VmaAllocation, Context::MAX_FRAMES_IN_FLIGHT> allocations_ = {VK_NULL_HANDLE};
    size_t size_ = 0;
    BufferType type_ = BufferType::UNIFORM;
};

// Keep backward-compatible alias
using UniformBuffer = DynamicBuffer;

inline std::expected<std::shared_ptr<Buffer>, Error> Buffer::create(Context& context, const void* data, size_t data_size, BufferType type, MemoryUsage usage) {
    if (usage == MemoryUsage::DYNAMIC) {
        return DynamicBuffer::create(context, data, data_size, type);
    } else {
        return StaticBuffer::create(context, data, data_size, type);
    }
}
