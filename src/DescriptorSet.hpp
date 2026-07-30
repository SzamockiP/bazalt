#pragma once
#include <volk.h>
#include <algorithm>
#include <expected>
#include <format>
#include <memory>
#include <unordered_map>
#include <vector>
#include "Context.hpp"
#include "Pipeline.hpp"
#include "Image.hpp"
#include "Sampler.hpp"
#include "Buffer.hpp"

class DescriptorPool;

class DescriptorSet
{
public:
    // The descriptor type rides along so the ResourceTracker can tell a
    // storage buffer (read-write in compute) from a uniform one at dispatch
    // and draw time.
    struct BoundBuffer
    {
        std::shared_ptr<Buffer> buffer;
        VkDescriptorType type;
        // Which binding it went to. Recorded since 0.19 because shader reflection
        // answers "is this written?" per (set, binding), and the tracker knows the
        // set from the map key but had no way to name the binding.
        std::uint32_t binding;
        // Which element of that binding, for a count>1 array. Together with
        // `binding` it is the identity of the descriptor, which is what lets a
        // rewrite replace the entry instead of appending a second one.
        std::uint32_t index;
    };

    // Same idea for images: STORAGE_IMAGE (compute read-write, GENERAL layout)
    // vs COMBINED_IMAGE_SAMPLER (sampled, SHADER_READ_ONLY). The type lets the
    // tracker transition a compute-written image before a later sample.
    struct BoundImage
    {
        std::shared_ptr<Image> image;
        VkDescriptorType type;
        std::uint32_t binding;
        std::uint32_t index;
        // Kept here rather than in a parallel vector: it belongs to this
        // descriptor, so it is replaced when the descriptor is.
        std::shared_ptr<Sampler> sampler;
    };

    // sets: 1 element (static) or frames_in_flight elements (frame)
    DescriptorSet(
        std::shared_ptr<Context> context,
        std::shared_ptr<DescriptorPool> pool,
        std::vector<VkDescriptorSet> sets,
        Pipeline::BindingTypeMap bindingTypes,
        bool isFrameSet)
        : context_(context),
          pool_(std::move(pool)),
          sets_(std::move(sets)),
          binding_types_(std::move(bindingTypes)),
          is_frame_set_(isFrameSet)
    {
    }

    // Frees the sets back to the pool, deferred (an in-flight frame may still
    // have them bound). The lambda captures the RAW pool handle, never the
    // shared_ptr — the pool holds the Context, and a shared_ptr sitting in the
    // Context's own deletion queue would keep the Context alive from its own
    // member. Ordering is safe without it: this object holds pool_ as a
    // member, so the pool's (also deferred) destruction is enqueued after this
    // free, and the queue runs in order.
    ~DescriptorSet();

    // Write an image + sampler to this descriptor set (all copies).
    // sampler == nullptr means "the default": linear, repeat, anisotropic —
    // resolved through the Context's cache, so it costs nothing.
    //
    // index selects the element of a count>1 array binding. It defaults to 0, so
    // a plain binding is the one-element case of the same call.
    std::expected<void, Error> set_image(
        uint32_t binding,
        std::shared_ptr<Image> image,
        std::shared_ptr<Sampler> sampler = nullptr,
        uint32_t index = 0)
    {
        if (!context_)
            return std::unexpected(err_init("Context destroyed"));
        if (!image)
            return std::unexpected(err_resource("set_image: image is null"));

        // A typo in the binding index used to surface only as a validation error
        // at submit time (or not at all with the layers off). Diagnose it here.
        auto decl = check_binding(binding, index, "set_image");
        if (!decl)
        {
            return std::unexpected(decl.error());
        }
        if (decl->type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
        {
            return std::unexpected(err_resource(
                std::format("Binding {} is not a sampler binding. Use set_buffer() for buffer bindings", binding)));
        }

        if (!sampler)
        {
            auto def = context_->get_sampler({});
            if (!def)
            {
                return std::unexpected(def.error());
            }
            sampler = std::move(*def);
        }

        VkDescriptorImageInfo imageInfo{
            .sampler = sampler->get(),
            .imageView = image->view(),
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

        for (auto& set : sets_)
        {
            VkWriteDescriptorSet write{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = set,
                .dstBinding = binding,
                .dstArrayElement = index,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &imageInfo,
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr};
            context_->vk().vkUpdateDescriptorSets(context_->device(), 1, &write, 0, nullptr);
        }
        record_image_(binding, index, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, std::move(image), std::move(sampler));
        return {};
    }

    // Write a storage image to this descriptor set (all copies). No sampler:
    // a storage image is read/written by coordinate (imageLoad/imageStore), and
    // its descriptor layout is GENERAL — the only layout a storage image may be
    // accessed in. The auto-barrier tracker transitions the image to GENERAL
    // before the dispatch, so the recorded layout here is always what the GPU
    // finds at execute time.
    std::expected<void, Error> set_storage_image(uint32_t binding, std::shared_ptr<Image> image, uint32_t index = 0)
    {
        if (!context_)
            return std::unexpected(err_init("Context destroyed"));
        if (!image)
            return std::unexpected(err_resource("set_storage_image: image is null"));

        auto decl = check_binding(binding, index, "set_storage_image");
        if (!decl)
        {
            return std::unexpected(decl.error());
        }
        if (decl->type != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
        {
            return std::unexpected(err_resource(
                std::format(
                    "Binding {} is not a storage-image binding. Declare it with "
                    ".storage_image({}) on the pipeline builder",
                    binding,
                    binding)));
        }

        VkDescriptorImageInfo imageInfo{
            .sampler = VK_NULL_HANDLE,
            // storage_view() is the 2D_ARRAY view for a cubemap (a CUBE view is
            // illegal as storage) and the plain view for everything else.
            .imageView = image->storage_view(),
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL};

        for (auto& set : sets_)
        {
            VkWriteDescriptorSet write{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = set,
                .dstBinding = binding,
                .dstArrayElement = index,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .pImageInfo = &imageInfo,
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr};
            context_->vk().vkUpdateDescriptorSets(context_->device(), 1, &write, 0, nullptr);
        }
        // A storage image is a compute output: after the dispatch it holds
        // contents, in GENERAL. Marking it here (record time) is the same
        // optimism as a storage buffer being readable — the headless submit
        // blocks before read(), so contents exist by then, and read()'s
        // transition needs the resting layout to be GENERAL, not UNDEFINED.
        image->mark_has_contents(VK_IMAGE_LAYOUT_GENERAL);
        record_image_(binding, index, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, std::move(image), nullptr);
        return {};
    }

    // Write a buffer to this descriptor set
    // For frame descriptor sets + DynamicBuffer: writes per-frame buffer to each copy
    // For static descriptor sets + DynamicBuffer: bz.ResourceError
    std::expected<void, Error> set_buffer(uint32_t binding, std::shared_ptr<Buffer> buffer, uint32_t index = 0)
    {
        if (!context_)
            return std::unexpected(err_init("Context destroyed"));
        if (!buffer)
            return std::unexpected(err_resource("set_buffer: buffer is null"));

        if (!is_frame_set_ && buffer->is_dynamic())
        {
            return std::unexpected(err_resource(
                "Cannot bind a DYNAMIC buffer to a static DescriptorSet. "
                "Use allocate_frame_set() instead."));
        }

        // No silent fallback: an unknown binding used to be *assumed* to be a
        // UNIFORM_BUFFER, so a typo'd index produced a descriptor write the
        // layout never declared — garbage diagnosed (at best) at submit time.
        // Either buffer type is accepted here, so the declared one is what gets
        // written and only a sampler binding is refused.
        auto decl = check_binding(binding, index, "set_buffer");
        if (!decl)
        {
            return std::unexpected(decl.error());
        }
        const VkDescriptorType descType = decl->type;
        if (descType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
        {
            return std::unexpected(err_resource(
                std::format("Binding {} is a sampler binding. Use set_image() for image bindings", binding)));
        }

        for (size_t i = 0; i < sets_.size(); i++)
        {
            VkBuffer vkBuf = buffer->get_for_frame(static_cast<uint32_t>(i));
            VkDescriptorBufferInfo bufferInfo{.buffer = vkBuf, .offset = 0, .range = buffer->size()};

            VkWriteDescriptorSet write{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = sets_[i],
                .dstBinding = binding,
                .dstArrayElement = index,
                .descriptorCount = 1,
                .descriptorType = descType,
                .pImageInfo = nullptr,
                .pBufferInfo = &bufferInfo,
                .pTexelBufferView = nullptr};
            context_->vk().vkUpdateDescriptorSets(context_->device(), 1, &write, 0, nullptr);
        }
        replace_or_append_(buffers_, {std::move(buffer), descType, binding, index});
        return {};
    }

    // Get the VkDescriptorSet for the given frame
    VkDescriptorSet get(uint32_t currentFrame) const
    {
        if (is_frame_set_)
        {
            return sets_[currentFrame % sets_.size()];
        }
        return sets_[0];
    }

    bool is_frame_set() const
    {
        return is_frame_set_;
    }

    // The images this set references — walked at submit time for upload
    // residency and at record time by the ResourceTracker (the type tells a
    // storage image from a sampled one).
    const std::vector<BoundImage>& images() const
    {
        return bound_images_;
    }

    // The buffers this set references — walked at record time by the
    // ResourceTracker to compute automatic barriers.
    const std::vector<BoundBuffer>& buffers() const
    {
        return buffers_;
    }

    const Context* owner() const
    {
        return context_.get();
    }

private:
    // What the three setters have to agree about: the binding exists and the array
    // element is inside the count the layout declared. One function, so a new
    // setter cannot invent a different answer. The type check stays at the call
    // sites, because each one has a different sentence to say about it.
    //
    // ResourceError rather than ValueError for the index too: the number is only
    // wrong relative to a layout, and deciding that means asking an object (the
    // 0.20 rule).
    std::expected<Pipeline::BindingInfo, Error> check_binding(uint32_t binding, uint32_t index, const char* what) const
    {
        auto it = binding_types_.find(binding);
        if (it == binding_types_.end())
        {
            return std::unexpected(
                err_resource(std::format("Binding {} does not exist in this descriptor set's layout", binding)));
        }
        if (index >= it->second.count)
        {
            return std::unexpected(err_resource(
                std::format(
                    "{}: index {} is outside binding {}, which was declared with count={}",
                    what,
                    index,
                    binding,
                    it->second.count)));
        }
        return it->second;
    }

    // Replace the entry for this (binding, index) or append a new one.
    //
    // Appending unconditionally is what these vectors used to do, and it was
    // invisible while a binding held one descriptor: writing the same binding twice
    // kept the first image alive for the set's whole life and made the record-time
    // tracker walk a list that only grows. A descriptor array rewritten per frame
    // turns that into an unbounded leak, so the identity of a descriptor —
    // (binding, index) — is what the list is keyed on.
    template <typename T>
    static void replace_or_append_(std::vector<T>& entries, T entry)
    {
        auto it = std::ranges::find_if(
            entries, [&](const T& e) { return e.binding == entry.binding && e.index == entry.index; });
        if (it != entries.end())
        {
            *it = std::move(entry);
            return;
        }
        entries.push_back(std::move(entry));
    }

    void record_image_(
        uint32_t binding,
        uint32_t index,
        VkDescriptorType type,
        std::shared_ptr<Image> image,
        std::shared_ptr<Sampler> sampler)
    {
        replace_or_append_(bound_images_, BoundImage{std::move(image), type, binding, index, std::move(sampler)});
    }

    std::shared_ptr<Context> context_;
    std::shared_ptr<DescriptorPool> pool_; // sets must not outlive their pool
    std::vector<VkDescriptorSet> sets_;
    Pipeline::BindingTypeMap binding_types_;
    bool is_frame_set_;
    // Hold shared_ptrs to prevent resources from being freed
    std::vector<BoundImage> bound_images_;
    std::vector<BoundBuffer> buffers_;
};

class DescriptorPool : public std::enable_shared_from_this<DescriptorPool>
{
public:
    static std::expected<std::shared_ptr<DescriptorPool>, Error> create(
        Context& context,
        uint32_t maxSets,
        uint32_t samplerCount,
        uint32_t uniformBufferCount,
        uint32_t storageBufferCount,
        uint32_t storageImageCount)
    {
        std::vector<VkDescriptorPoolSize> poolSizes;

        if (samplerCount > 0)
        {
            poolSizes.push_back({.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = samplerCount});
        }
        if (uniformBufferCount > 0)
        {
            poolSizes.push_back({.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = uniformBufferCount});
        }
        if (storageBufferCount > 0)
        {
            poolSizes.push_back({.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = storageBufferCount});
        }
        if (storageImageCount > 0)
        {
            poolSizes.push_back({.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = storageImageCount});
        }

        if (poolSizes.empty())
        {
            return std::unexpected(err_resource("DescriptorPool must have at least one non-zero descriptor count"));
        }

        // A set whose layout has an UPDATE_AFTER_BIND binding may only come from a
        // pool that says so, and the pool cannot know which layouts it will serve.
        // Gated on the feature rather than set always, so a Context that never asked
        // for BINDLESS keeps the descriptor limits it had: update-after-bind
        // descriptors are counted against a separate, sometimes smaller, budget.
        VkDescriptorPoolCreateFlags flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        if (context.supports(Feature::BINDLESS))
        {
            flags |= VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        }

        VkDescriptorPoolCreateInfo poolInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            // Sets return to the pool when their Python object is collected.
            // Without this flag a pool was strictly one-way: allocate enough
            // times and it fills up forever.
            .flags = flags,
            .maxSets = maxSets,
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data()};

        VkDescriptorPool pool;
        if (auto e = check(
                context.vk().vkCreateDescriptorPool(context.device(), &poolInfo, nullptr, &pool),
                "create descriptor pool",
                ErrorCode::Resource))
        {
            return std::unexpected(*e);
        }

        return std::shared_ptr<DescriptorPool>(new DescriptorPool(context.shared_from_this(), pool));
    }

    // Deferred, so it lands in the queue after every set's free (sets hold the
    // pool, so their destructors necessarily run first).
    ~DescriptorPool()
    {
        if (pool_ != VK_NULL_HANDLE && context_)
        {
            context_->defer_destroy([vk = &context_->vk(), device = context_->device(), pool = pool_]
                                    { vk->vkDestroyDescriptorPool(device, pool, nullptr); });
        }
    }

    VkDescriptorPool get() const
    {
        return pool_;
    }

    DescriptorPool(const DescriptorPool&) = delete;
    DescriptorPool& operator=(const DescriptorPool&) = delete;

    // Allocate a static descriptor set (1 VkDescriptorSet)
    std::expected<std::shared_ptr<DescriptorSet>, Error> allocate_descriptor_set(
        std::shared_ptr<Pipeline> pipeline,
        uint32_t setIndex)
    {
        if (!context_)
            return std::unexpected(err_init("Context destroyed"));

        VkDescriptorSetLayout layout = pipeline->descriptor_set_layout(setIndex);
        if (layout == VK_NULL_HANDLE)
        {
            return std::unexpected(
                err_resource("Pipeline has no descriptor set layout at set index " + std::to_string(setIndex)));
        }

        VkDescriptorSetAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = pool_,
            .descriptorSetCount = 1,
            .pSetLayouts = &layout};

        VkDescriptorSet set;
        if (auto e = check(
                context_->vk().vkAllocateDescriptorSets(context_->device(), &allocInfo, &set),
                "allocate descriptor set from pool (pool may be full)",
                ErrorCode::Resource))
        {
            return std::unexpected(*e);
        }

        return std::make_shared<DescriptorSet>(
            context_, shared_from_this(), std::vector<VkDescriptorSet>{set}, pipeline->binding_types(setIndex), false);
    }

    // Allocate a frame descriptor set (frames_in_flight VkDescriptorSets)
    std::expected<std::shared_ptr<DescriptorSet>, Error> allocate_frame_descriptor_set(
        std::shared_ptr<Pipeline> pipeline,
        uint32_t setIndex)
    {
        if (!context_)
            return std::unexpected(err_init("Context destroyed"));

        VkDescriptorSetLayout layout = pipeline->descriptor_set_layout(setIndex);
        if (layout == VK_NULL_HANDLE)
        {
            return std::unexpected(
                err_resource("Pipeline has no descriptor set layout at set index " + std::to_string(setIndex)));
        }

        const uint32_t frames = context_->frames_in_flight();
        std::vector<VkDescriptorSetLayout> layouts(frames, layout);
        VkDescriptorSetAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = pool_,
            .descriptorSetCount = frames,
            .pSetLayouts = layouts.data()};

        std::vector<VkDescriptorSet> sets(frames);
        if (auto e = check(
                context_->vk().vkAllocateDescriptorSets(context_->device(), &allocInfo, sets.data()),
                "allocate frame descriptor sets from pool (pool may be full)",
                ErrorCode::Resource))
        {
            return std::unexpected(*e);
        }

        return std::make_shared<DescriptorSet>(
            context_, shared_from_this(), std::move(sets), pipeline->binding_types(setIndex), true);
    }

    std::shared_ptr<Logger> logger() const
    {
        return context_ ? context_->logger() : nullptr;
    }

private:
    DescriptorPool(std::shared_ptr<Context> context, VkDescriptorPool pool)
        : context_(context),
          pool_(pool)
    {
    }

public:
    const Context* owner() const
    {
        return context_.get();
    }

private:
    std::shared_ptr<Context> context_;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
};

// Out of line: needs DescriptorPool to be complete for pool_->get().
inline DescriptorSet::~DescriptorSet()
{
    if (context_ && pool_ && !sets_.empty())
    {
        context_->defer_destroy(
            [vk = &context_->vk(), device = context_->device(), pool = pool_->get(), sets = std::move(sets_)]
            { vk->vkFreeDescriptorSets(device, pool, static_cast<uint32_t>(sets.size()), sets.data()); });
    }
}
