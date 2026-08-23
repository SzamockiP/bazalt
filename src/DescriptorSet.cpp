#include "DescriptorSet.hpp"

#include <algorithm>
#include <format>
#include <string>

// Out of line only for symmetry with its history; the free names block_, the
// exact VkDescriptorPool that allocated these sets — an auto pool owns several.
DescriptorSet::~DescriptorSet()
{
    if (context_ && pool_ && block_ != VK_NULL_HANDLE && !sets_.empty())
    {
        context_->defer_destroy(
            [vk = &context_->vk(), device = context_->device(), pool = block_, sets = std::move(sets_)]
            { vk->vkFreeDescriptorSets(device, pool, static_cast<uint32_t>(sets.size()), sets.data()); });
    }
}

std::expected<void, Error> DescriptorSet::set_image(
    uint32_t binding,
    std::shared_ptr<Image> image,
    std::shared_ptr<Sampler> sampler,
    uint32_t index)
{
    if (!context_)
    {
        return std::unexpected(err_init("Context destroyed"));
    }
    if (!image)
    {
        return std::unexpected(err_resource("set_image: image is null"));
    }

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
        .sampler = sampler->get(), .imageView = image->view(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

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

std::expected<void, Error> DescriptorSet::set_storage_image(
    uint32_t binding,
    std::shared_ptr<Image> image,
    uint32_t index)
{
    if (!context_)
    {
        return std::unexpected(err_init("Context destroyed"));
    }
    if (!image)
    {
        return std::unexpected(err_resource("set_storage_image: image is null"));
    }

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

std::expected<void, Error> DescriptorSet::set_buffer(uint32_t binding, std::shared_ptr<Buffer> buffer, uint32_t index)
{
    if (!context_)
    {
        return std::unexpected(err_init("Context destroyed"));
    }
    if (!buffer)
    {
        return std::unexpected(err_resource("set_buffer: buffer is null"));
    }

    if (!is_frame_set_ && buffer->is_dynamic())
    {
        return std::unexpected(err_resource(
            "Cannot bind a DYNAMIC buffer to a static DescriptorSet. "
            "Use allocate_frame_set() instead."));
    }

    // No silent fallback: an unknown binding used to be *assumed* to be a
    // UNIFORM_BUFFER, so a typo'd index produced a descriptor write the
    // layout never declared — garbage diagnosed (at best) at submit time.
    auto decl = check_binding(binding, index, "set_buffer");
    if (!decl)
    {
        return std::unexpected(decl.error());
    }
    const VkDescriptorType descType = decl->type;
    if (descType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
    {
        return std::unexpected(
            err_resource(std::format("Binding {} is a sampler binding. Use set_image() for image bindings", binding)));
    }
    // The buffer must carry the bit the binding declares (0.30). Before usages
    // were bits "the declared type is what gets written" was the whole rule,
    // and a VERTEX buffer on a uniform binding reached the layers as
    // VUID-VkWriteDescriptorSet-descriptorType-00327. With unions the rule has
    // to be per bit, or UNIFORM|STORAGE would be the first buffer for which the
    // old rule is silently wrong.
    const bool storageBinding = descType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    const BufferUsage needed = storageBinding ? BufferUsage::STORAGE : BufferUsage::UNIFORM;
    if (!has(buffer->buffer_usage(), needed))
    {
        return std::unexpected(err_resource(
            std::format(
                "set_buffer: binding {} is a {} binding and this buffer was created with usage={}. "
                "Create the buffer with BufferUsage.{} in its usage.",
                binding,
                storageBinding ? "storage" : "uniform",
                buffer_usage_name(buffer->buffer_usage()),
                buffer_usage_name(needed))));
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
    replace_or_append_(buffers_, {.buffer = std::move(buffer), .type = descType, .binding = binding, .index = index});
    return {};
}

VkDescriptorSet DescriptorSet::get(uint32_t currentFrame) const
{
    if (is_frame_set_)
    {
        return sets_[currentFrame % sets_.size()];
    }
    return sets_[0];
}

std::expected<Pipeline::BindingInfo, Error> DescriptorSet::check_binding(
    uint32_t binding,
    uint32_t index,
    const char* what) const
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

void DescriptorSet::record_image_(
    uint32_t binding,
    uint32_t index,
    VkDescriptorType type,
    std::shared_ptr<Image> image,
    std::shared_ptr<Sampler> sampler)
{
    replace_or_append_(
        bound_images_,
        BoundImage{
            .image = std::move(image),
            .type = type,
            .binding = binding,
            .index = index,
            .sampler = std::move(sampler)});
}

std::expected<std::shared_ptr<DescriptorPool>, Error> DescriptorPool::create(
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
    Block record{.pool = *block, .capacity = {}, .max_sets = maxSets};
    for (const auto& size : poolSizes)
    {
        record.capacity[size.type] = size.descriptorCount;
    }
    pool->blocks_.push_back(record);
    return pool;
}

std::expected<std::shared_ptr<DescriptorPool>, Error> DescriptorPool::create_auto(Context& context)
{
    return std::shared_ptr<DescriptorPool>(new DescriptorPool(context.shared_from_this(), true));
}

DescriptorPool::~DescriptorPool()
{
    if (!blocks_.empty() && context_)
    {
        context_->defer_destroy(
            [vk = &context_->vk(), device = context_->device(), blocks = std::move(blocks_)]
            {
                for (const auto& block : blocks)
                {
                    vk->vkDestroyDescriptorPool(device, block.pool, nullptr);
                }
            });
    }
}

std::expected<std::shared_ptr<DescriptorSet>, Error> DescriptorPool::allocate_(
    const std::shared_ptr<Pipeline>& pipeline,
    uint32_t setIndex,
    bool frame_set)
{
    if (!context_)
    {
        return std::unexpected(err_init("Context destroyed"));
    }

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
    // Ask before trying, since 0.26. A self-sizing pool used to discover a
    // request it could not fit by making it and reading the error, so the
    // documented default printed a validation warning on its way to working
    // — which is the one thing a pool that sizes itself must not do. The
    // check is against what the block was declared to hold, so a request
    // larger than any block skips straight to growth.
    if (!blocks_.empty() && !(grows_ && exceeds_(blocks_.back(), needed_(*pipeline, setIndex, count), count)))
    {
        from = blocks_.back().pool;
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

VkResult DescriptorPool::try_allocate_(
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

std::expected<VkDescriptorPool, Error> DescriptorPool::grow_for_(
    const Pipeline& pipeline,
    uint32_t setIndex,
    uint32_t set_count)
{
    constexpr uint32_t kDefaultSets = 64;
    constexpr uint32_t kDefaultDescriptors = 64;

    const auto needed = needed_(pipeline, setIndex, set_count);
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
    const uint32_t max_sets = (std::ranges::max)(kDefaultSets, set_count);
    auto block = create_block_(*context_, max_sets, sizes);
    if (!block)
    {
        return std::unexpected(block.error());
    }
    Block record{.pool = *block, .capacity = {}, .max_sets = max_sets};
    for (const auto& size : sizes)
    {
        record.capacity[size.type] = size.descriptorCount;
    }
    blocks_.push_back(record);
    return *block;
}

std::expected<VkDescriptorPool, Error> DescriptorPool::create_block_(
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

    VkDescriptorPool pool = nullptr;
    if (auto e = check(
            context.vk().vkCreateDescriptorPool(context.device(), &poolInfo, nullptr, &pool),
            "create descriptor pool",
            ErrorCode::Resource))
    {
        return std::unexpected(*e);
    }
    return pool;
}

std::unordered_map<VkDescriptorType, uint32_t> DescriptorPool::needed_(
    const Pipeline& pipeline,
    uint32_t setIndex,
    uint32_t set_count)
{
    std::unordered_map<VkDescriptorType, uint32_t> needed;
    for (const auto& [binding, info] : pipeline.binding_types(setIndex))
    {
        needed[info.type] += info.count * set_count;
    }
    return needed;
}

bool DescriptorPool::exceeds_(
    const Block& block,
    const std::unordered_map<VkDescriptorType, uint32_t>& needed,
    uint32_t set_count)
{
    if (set_count > block.max_sets)
    {
        return true;
    }
    return std::ranges::any_of(
        needed,
        [&block](const auto& entry)
        {
            const auto& [type, count] = entry;
            const auto it = block.capacity.find(type);
            return it == block.capacity.end() || it->second < count;
        });
}
