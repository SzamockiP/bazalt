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

    // sets: 1 element (static) or frames_in_flight elements (frame). `block` is
    // the VkDescriptorPool the sets came from: an auto pool owns several, and a
    // free must go back to the one that allocated — the shared_ptr alone cannot
    // say which.
    DescriptorSet(
        std::shared_ptr<Context> context,
        std::shared_ptr<DescriptorPool> pool,
        VkDescriptorPool block,
        std::vector<VkDescriptorSet> sets,
        Pipeline::BindingTypeMap bindingTypes,
        bool isFrameSet,
        std::uint32_t setIndex = 0,
        VkPipelineBindPoint bindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS)
        : context_(context),
          pool_(std::move(pool)),
          block_(block),
          sets_(std::move(sets)),
          binding_types_(std::move(bindingTypes)),
          is_frame_set_(isFrameSet),
          set_index_(setIndex),
          bind_point_(bindPoint)
    {
    }

    // Which set index this was allocated for, and at which bind point. Recorded
    // since 0.25 so cmd.bind_descriptor_set(set) can work them out instead of
    // making the caller repeat what the allocation already decided.
    std::uint32_t set_index() const
    {
        return set_index_;
    }
    VkPipelineBindPoint bind_point() const
    {
        return bind_point_;
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
    std::shared_ptr<DescriptorPool> pool_;    // sets must not outlive their pool
    VkDescriptorPool block_ = VK_NULL_HANDLE; // the block the sets free back into
    std::vector<VkDescriptorSet> sets_;
    Pipeline::BindingTypeMap binding_types_;
    bool is_frame_set_;
    std::uint32_t set_index_ = 0;
    VkPipelineBindPoint bind_point_ = VK_PIPELINE_BIND_POINT_GRAPHICS;
    // Hold shared_ptrs to prevent resources from being freed
    std::vector<BoundImage> bound_images_;
    std::vector<BoundBuffer> buffers_;
};

class DescriptorPool : public std::enable_shared_from_this<DescriptorPool>
{
public:
    // Fixed mode: one VkDescriptorPool of exactly these sizes, and exhaustion
    // is an error. The escape hatch for anyone who wants to budget descriptors
    // by hand; create_auto below is the default story.
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

        auto block = create_block_(context, maxSets, poolSizes);
        if (!block)
        {
            return std::unexpected(block.error());
        }
        auto pool = std::shared_ptr<DescriptorPool>(new DescriptorPool(context.shared_from_this(), false));
        pool->blocks_.push_back(*block);
        return pool;
    }

    // Auto mode (0.23): no sizes at all. Blocks are allocated as sets are, each
    // sized from the layout being served, so the caller stops doing Vulkan's
    // arithmetic — which depended on frames_in_flight, a number that never
    // appeared in the old call. A whole count=N array must fit its block
    // (MoltenVK refuses a partial fit, 0.22), and sizing from the layout is
    // what guarantees it.
    static std::expected<std::shared_ptr<DescriptorPool>, Error> create_auto(Context& context)
    {
        return std::shared_ptr<DescriptorPool>(new DescriptorPool(context.shared_from_this(), true));
    }

    // Deferred, so it lands in the queue after every set's free (sets hold the
    // pool, so their destructors necessarily run first).
    ~DescriptorPool()
    {
        if (!blocks_.empty() && context_)
        {
            context_->defer_destroy(
                [vk = &context_->vk(), device = context_->device(), blocks = std::move(blocks_)]
                {
                    for (VkDescriptorPool block : blocks)
                    {
                        vk->vkDestroyDescriptorPool(device, block, nullptr);
                    }
                });
        }
    }

    DescriptorPool(const DescriptorPool&) = delete;
    DescriptorPool& operator=(const DescriptorPool&) = delete;

    // Allocate a static descriptor set (1 VkDescriptorSet)
    std::expected<std::shared_ptr<DescriptorSet>, Error> allocate_descriptor_set(
        std::shared_ptr<Pipeline> pipeline,
        uint32_t setIndex)
    {
        return allocate_(std::move(pipeline), setIndex, /*frame_set=*/false);
    }

    // Allocate a frame descriptor set (frames_in_flight VkDescriptorSets)
    std::expected<std::shared_ptr<DescriptorSet>, Error> allocate_frame_descriptor_set(
        std::shared_ptr<Pipeline> pipeline,
        uint32_t setIndex)
    {
        return allocate_(std::move(pipeline), setIndex, /*frame_set=*/true);
    }

    std::shared_ptr<Logger> logger() const
    {
        return context_ ? context_->logger() : nullptr;
    }

private:
    DescriptorPool(std::shared_ptr<Context> context, bool grows)
        : context_(std::move(context)),
          grows_(grows)
    {
    }

    // One body for both set kinds: they differ only by how many sets one call
    // allocates, and the two used to disagree about nothing but their strings.
    std::expected<std::shared_ptr<DescriptorSet>, Error> allocate_(
        std::shared_ptr<Pipeline> pipeline,
        uint32_t setIndex,
        bool frame_set)
    {
        if (!context_)
            return std::unexpected(err_init("Context destroyed"));

        VkDescriptorSetLayout layout = pipeline->descriptor_set_layout(setIndex);
        if (layout == VK_NULL_HANDLE)
        {
            return std::unexpected(
                err_resource("Pipeline has no descriptor set layout at set index " + std::to_string(setIndex)));
        }

        const uint32_t count = frame_set ? context_->frames_in_flight() : 1;
        std::vector<VkDescriptorSetLayout> layouts(count, layout);

        std::vector<VkDescriptorSet> sets(count);
        VkResult result = VK_ERROR_OUT_OF_POOL_MEMORY; // "no block yet" allocates one
        VkDescriptorPool from = VK_NULL_HANDLE;
        if (!blocks_.empty())
        {
            from = blocks_.back();
            result = try_allocate_(from, layouts, sets);
        }
        // Growth, auto mode only: a full block is the expected state of a pool
        // that sizes itself, so a fresh block — sized from THIS request's
        // layout, so even a count=500 array fits — is allocated and the request
        // retried once. Everything else (device OOM, a fixed pool filling up)
        // stays an error.
        if (grows_ && (result == VK_ERROR_OUT_OF_POOL_MEMORY || result == VK_ERROR_FRAGMENTED_POOL))
        {
            auto grown = grow_for_(*pipeline, setIndex, count);
            if (!grown)
            {
                return std::unexpected(grown.error());
            }
            from = *grown;
            result = try_allocate_(from, layouts, sets);
        }
        if (auto e = check(result, "allocate descriptor set from pool (pool may be full)", ErrorCode::Resource))
        {
            return std::unexpected(*e);
        }

        return std::make_shared<DescriptorSet>(
            context_,
            shared_from_this(),
            from,
            std::move(sets),
            pipeline->binding_types(setIndex),
            frame_set,
            setIndex,
            pipeline->bind_point());
    }

    VkResult try_allocate_(
        VkDescriptorPool from,
        const std::vector<VkDescriptorSetLayout>& layouts,
        std::vector<VkDescriptorSet>& sets)
    {
        VkDescriptorSetAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = from,
            .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
            .pSetLayouts = layouts.data()};
        return context_->vk().vkAllocateDescriptorSets(context_->device(), &allocInfo, sets.data());
    }

    // A new auto-mode block. Sized max(default, this request) per descriptor
    // type: the default keeps small sets from each costing a VkDescriptorPool,
    // and the request half is what lets one bindless array larger than any
    // default land in a block of its own.
    std::expected<VkDescriptorPool, Error> grow_for_(const Pipeline& pipeline, uint32_t setIndex, uint32_t set_count)
    {
        constexpr uint32_t kDefaultSets = 64;
        constexpr uint32_t kDefaultDescriptors = 64;

        std::unordered_map<VkDescriptorType, uint32_t> needed;
        for (const auto& [binding, info] : pipeline.binding_types(setIndex))
        {
            needed[info.type] += info.count * set_count;
        }
        std::vector<VkDescriptorPoolSize> sizes;
        for (VkDescriptorType type :
             {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE})
        {
            const auto it = needed.find(type);
            const uint32_t requested = it == needed.end() ? 0 : it->second;
            sizes.push_back({.type = type, .descriptorCount = (std::ranges::max)(kDefaultDescriptors, requested)});
        }
        auto block = create_block_(*context_, (std::ranges::max)(kDefaultSets, set_count), sizes);
        if (!block)
        {
            return std::unexpected(block.error());
        }
        blocks_.push_back(*block);
        return *block;
    }

    static std::expected<VkDescriptorPool, Error> create_block_(
        Context& context,
        uint32_t maxSets,
        const std::vector<VkDescriptorPoolSize>& poolSizes)
    {
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
        return pool;
    }

public:
    const Context* owner() const
    {
        return context_.get();
    }

private:
    std::shared_ptr<Context> context_;
    // Every VkDescriptorPool this object owns. Fixed mode holds exactly one and
    // never grows; auto mode appends. Sets free back into the block that
    // allocated them (DescriptorSet::block_), so a block is never emptied by a
    // free aimed at a sibling.
    std::vector<VkDescriptorPool> blocks_;
    bool grows_ = false;
};

// Out of line only for symmetry with its history; the free names block_, the
// exact VkDescriptorPool that allocated these sets — an auto pool owns several.
inline DescriptorSet::~DescriptorSet()
{
    if (context_ && pool_ && block_ != VK_NULL_HANDLE && !sets_.empty())
    {
        context_->defer_destroy(
            [vk = &context_->vk(), device = context_->device(), pool = block_, sets = std::move(sets_)]
            { vk->vkFreeDescriptorSets(device, pool, static_cast<uint32_t>(sets.size()), sets.data()); });
    }
}
