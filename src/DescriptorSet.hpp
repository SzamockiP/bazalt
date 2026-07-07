#pragma once
#include <volk.h>
#include <vector>
#include <memory>
#include <expected>
#include <stdexcept>
#include <unordered_map>
#include "Renderer.hpp"
#include "Pipeline.hpp"
#include "Texture.hpp"
#include "Buffer.hpp"

class DescriptorSet {
public:
    // sets: 1 element (static) or MAX_FRAMES_IN_FLIGHT elements (frame)
    DescriptorSet(VkDevice device, std::vector<VkDescriptorSet> sets,
                  Pipeline::BindingTypeMap bindingTypes, bool isFrameSet)
        : device_(device), sets_(std::move(sets)),
          binding_types_(std::move(bindingTypes)), is_frame_set_(isFrameSet) {}

    // Write a texture to this descriptor set (all copies)
    void setTexture(uint32_t binding, std::shared_ptr<Texture> texture) {
        VkDescriptorImageInfo imageInfo{
            .sampler = texture->sampler(),
            .imageView = texture->image_view(),
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        };

        for (auto& set : sets_) {
            VkWriteDescriptorSet write{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = set,
                .dstBinding = binding,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &imageInfo,
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr
            };
            vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
        }
        textures_.push_back(texture);
    }

    // Write a buffer to this descriptor set
    // For frame descriptor sets + DynamicBuffer: writes per-frame buffer to each copy
    // For static descriptor sets + DynamicBuffer: throws RuntimeError
    void setBuffer(uint32_t binding, std::shared_ptr<Buffer> buffer) {
        if (!is_frame_set_ && buffer->is_dynamic()) {
            throw std::runtime_error(
                "Cannot bind DynamicBuffer to a static DescriptorSet. "
                "Use allocateFrameDescriptorSet() instead.");
        }

        auto it = binding_types_.find(binding);
        VkDescriptorType descType = (it != binding_types_.end()) 
            ? it->second 
            : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

        for (size_t i = 0; i < sets_.size(); i++) {
            VkBuffer vkBuf = buffer->get_for_frame(static_cast<uint32_t>(i));
            VkDescriptorBufferInfo bufferInfo{
                .buffer = vkBuf,
                .offset = 0,
                .range = buffer->size()
            };

            VkWriteDescriptorSet write{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = sets_[i],
                .dstBinding = binding,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = descType,
                .pImageInfo = nullptr,
                .pBufferInfo = &bufferInfo,
                .pTexelBufferView = nullptr
            };
            vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
        }
        buffers_.push_back(buffer);
    }

    // Get the VkDescriptorSet for the given frame
    VkDescriptorSet get(uint32_t currentFrame) const {
        if (is_frame_set_) {
            return sets_[currentFrame % sets_.size()];
        }
        return sets_[0];
    }

    bool is_frame_set() const { return is_frame_set_; }

private:
    VkDevice device_;
    std::vector<VkDescriptorSet> sets_;
    Pipeline::BindingTypeMap binding_types_;
    bool is_frame_set_;
    // Hold shared_ptrs to prevent resources from being freed
    std::vector<std::shared_ptr<Texture>> textures_;
    std::vector<std::shared_ptr<Buffer>> buffers_;
};

class DescriptorPool {
public:
    static std::expected<std::shared_ptr<DescriptorPool>, std::string> create(
        VkDevice device,
        uint32_t maxSets,
        uint32_t samplerCount,
        uint32_t uniformBufferCount,
        uint32_t storageBufferCount)
    {
        std::vector<VkDescriptorPoolSize> poolSizes;
        
        if (samplerCount > 0) {
            poolSizes.push_back({
                .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = samplerCount
            });
        }
        if (uniformBufferCount > 0) {
            poolSizes.push_back({
                .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = uniformBufferCount
            });
        }
        if (storageBufferCount > 0) {
            poolSizes.push_back({
                .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = storageBufferCount
            });
        }

        if (poolSizes.empty()) {
            return std::unexpected("DescriptorPool must have at least one non-zero descriptor count");
        }

        VkDescriptorPoolCreateInfo poolInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .maxSets = maxSets,
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data()
        };

        VkDescriptorPool pool;
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &pool) != VK_SUCCESS) {
            return std::unexpected("Failed to create descriptor pool");
        }

        return std::shared_ptr<DescriptorPool>(new DescriptorPool(device, pool));
    }

    ~DescriptorPool() {
        if (pool_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device_, pool_, nullptr);
        }
    }

    DescriptorPool(const DescriptorPool&) = delete;
    DescriptorPool& operator=(const DescriptorPool&) = delete;

    // Allocate a static descriptor set (1 VkDescriptorSet)
    std::expected<std::shared_ptr<DescriptorSet>, std::string>
    allocateDescriptorSet(std::shared_ptr<Pipeline> pipeline, uint32_t setIndex) {
        VkDescriptorSetLayout layout = pipeline->descriptor_set_layout(setIndex);
        if (layout == VK_NULL_HANDLE) {
            return std::unexpected("Pipeline has no descriptor set layout at set index " + std::to_string(setIndex));
        }

        VkDescriptorSetAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = pool_,
            .descriptorSetCount = 1,
            .pSetLayouts = &layout
        };

        VkDescriptorSet set;
        if (vkAllocateDescriptorSets(device_, &allocInfo, &set) != VK_SUCCESS) {
            return std::unexpected("Failed to allocate descriptor set from pool (pool may be full)");
        }

        return std::make_shared<DescriptorSet>(
            device_, std::vector<VkDescriptorSet>{set},
            pipeline->binding_types(setIndex), false);
    }

    // Allocate a frame descriptor set (MAX_FRAMES_IN_FLIGHT VkDescriptorSets)
    std::expected<std::shared_ptr<DescriptorSet>, std::string>
    allocateFrameDescriptorSet(std::shared_ptr<Pipeline> pipeline, uint32_t setIndex) {
        VkDescriptorSetLayout layout = pipeline->descriptor_set_layout(setIndex);
        if (layout == VK_NULL_HANDLE) {
            return std::unexpected("Pipeline has no descriptor set layout at set index " + std::to_string(setIndex));
        }

        std::vector<VkDescriptorSetLayout> layouts(Renderer::MAX_FRAMES_IN_FLIGHT, layout);
        VkDescriptorSetAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = pool_,
            .descriptorSetCount = Renderer::MAX_FRAMES_IN_FLIGHT,
            .pSetLayouts = layouts.data()
        };

        std::vector<VkDescriptorSet> sets(Renderer::MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(device_, &allocInfo, sets.data()) != VK_SUCCESS) {
            return std::unexpected("Failed to allocate frame descriptor sets from pool (pool may be full)");
        }

        return std::make_shared<DescriptorSet>(
            device_, std::move(sets),
            pipeline->binding_types(setIndex), true);
    }

private:
    DescriptorPool(VkDevice device, VkDescriptorPool pool)
        : device_(device), pool_(pool) {}

    VkDevice device_;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
};
