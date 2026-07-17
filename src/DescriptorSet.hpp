#pragma once
#include <volk.h>
#include <vector>
#include <memory>
#include <expected>
#include <unordered_map>
#include "Context.hpp"
#include "Pipeline.hpp"
#include "Texture.hpp"
#include "Buffer.hpp"

class DescriptorSet {
public:
    // sets: 1 element (static) or MAX_FRAMES_IN_FLIGHT elements (frame)
    DescriptorSet(std::shared_ptr<Context> context, std::vector<VkDescriptorSet> sets,
                  Pipeline::BindingTypeMap bindingTypes, bool isFrameSet)
        : context_(context), sets_(std::move(sets)),
          binding_types_(std::move(bindingTypes)), is_frame_set_(isFrameSet) {}

    // Write a texture to this descriptor set (all copies)
    std::expected<void, Error> setTexture(uint32_t binding, std::shared_ptr<Texture> texture) {
        if (!context_) return {};

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
            vkUpdateDescriptorSets(context_->device(), 1, &write, 0, nullptr);
        }
        textures_.push_back(texture);
        return {};
    }

    // Write a buffer to this descriptor set
    // For frame descriptor sets + DynamicBuffer: writes per-frame buffer to each copy
    // For static descriptor sets + DynamicBuffer: bz.ResourceError
    std::expected<void, Error> setBuffer(uint32_t binding, std::shared_ptr<Buffer> buffer) {
        if (!context_) return {};

        if (!is_frame_set_ && buffer->is_dynamic()) {
            return std::unexpected(err_resource(
                "Cannot bind a DYNAMIC buffer to a static DescriptorSet. "
                "Use allocate_frame_set() instead."));
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
            vkUpdateDescriptorSets(context_->device(), 1, &write, 0, nullptr);
        }
        buffers_.push_back(buffer);
        return {};
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
    std::shared_ptr<Context> context_;
    std::vector<VkDescriptorSet> sets_;
    Pipeline::BindingTypeMap binding_types_;
    bool is_frame_set_;
    // Hold shared_ptrs to prevent resources from being freed
    std::vector<std::shared_ptr<Texture>> textures_;
    std::vector<std::shared_ptr<Buffer>> buffers_;
};

class DescriptorPool {
public:
    static std::expected<std::shared_ptr<DescriptorPool>, Error> create(
        Context& context,
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
            return std::unexpected(err_resource("DescriptorPool must have at least one non-zero descriptor count"));
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
        if (auto e = check(vkCreateDescriptorPool(context.device(), &poolInfo, nullptr, &pool),
                           "create descriptor pool", ErrorCode::Resource)) {
            return std::unexpected(*e);
        }

        return std::shared_ptr<DescriptorPool>(new DescriptorPool(context.shared_from_this(), pool));
    }

    ~DescriptorPool() {
        if (pool_ != VK_NULL_HANDLE && context_) {
            vkDestroyDescriptorPool(context_->device(), pool_, nullptr);
        }
    }

    DescriptorPool(const DescriptorPool&) = delete;
    DescriptorPool& operator=(const DescriptorPool&) = delete;

    // Allocate a static descriptor set (1 VkDescriptorSet)
    std::expected<std::shared_ptr<DescriptorSet>, Error>
    allocateDescriptorSet(std::shared_ptr<Pipeline> pipeline, uint32_t setIndex) {
        if (!context_) return std::unexpected(err_init("Context destroyed"));

        VkDescriptorSetLayout layout = pipeline->descriptor_set_layout(setIndex);
        if (layout == VK_NULL_HANDLE) {
            return std::unexpected(err_resource("Pipeline has no descriptor set layout at set index " + std::to_string(setIndex)));
        }

        VkDescriptorSetAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = pool_,
            .descriptorSetCount = 1,
            .pSetLayouts = &layout
        };

        VkDescriptorSet set;
        if (auto e = check(vkAllocateDescriptorSets(context_->device(), &allocInfo, &set),
                           "allocate descriptor set from pool (pool may be full)", ErrorCode::Resource)) {
            return std::unexpected(*e);
        }

        return std::make_shared<DescriptorSet>(
            context_, std::vector<VkDescriptorSet>{set},
            pipeline->binding_types(setIndex), false);
    }

    // Allocate a frame descriptor set (MAX_FRAMES_IN_FLIGHT VkDescriptorSets)
    std::expected<std::shared_ptr<DescriptorSet>, Error>
    allocateFrameDescriptorSet(std::shared_ptr<Pipeline> pipeline, uint32_t setIndex) {
        if (!context_) return std::unexpected(err_init("Context destroyed"));

        VkDescriptorSetLayout layout = pipeline->descriptor_set_layout(setIndex);
        if (layout == VK_NULL_HANDLE) {
            return std::unexpected(err_resource("Pipeline has no descriptor set layout at set index " + std::to_string(setIndex)));
        }

        std::vector<VkDescriptorSetLayout> layouts(Context::MAX_FRAMES_IN_FLIGHT, layout);
        VkDescriptorSetAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = pool_,
            .descriptorSetCount = Context::MAX_FRAMES_IN_FLIGHT,
            .pSetLayouts = layouts.data()
        };

        std::vector<VkDescriptorSet> sets(Context::MAX_FRAMES_IN_FLIGHT);
        if (auto e = check(vkAllocateDescriptorSets(context_->device(), &allocInfo, sets.data()),
                           "allocate frame descriptor sets from pool (pool may be full)", ErrorCode::Resource)) {
            return std::unexpected(*e);
        }

        return std::make_shared<DescriptorSet>(
            context_, std::move(sets),
            pipeline->binding_types(setIndex), true);
    }

    std::shared_ptr<Logger> logger() const {
        return context_ ? context_->logger() : nullptr;
    }

private:
    DescriptorPool(std::shared_ptr<Context> context, VkDescriptorPool pool)
        : context_(context), pool_(pool) {}

    std::shared_ptr<Context> context_;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
};
