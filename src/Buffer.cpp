#include "Buffer.hpp"

#include <utility>

// ── Buffer ────────────────────────────────────────────────────────────────────

std::expected<void, Error> Buffer::update(std::span<const std::byte> /*data*/, size_t /*offset*/)
{
    return std::unexpected(err_resource(
        "update() is only supported on DYNAMIC buffers. "
        "Create the buffer with MemoryUsage.DYNAMIC instead"));
}

std::expected<VkDeviceAddress, Error> Buffer::address()
{
    wait();
    const Context* context = owner();
    if (context == nullptr)
    {
        return std::unexpected(err_state("this buffer has no Context"));
    }
    if (!context->supports(Feature::BUFFER_ADDRESS))
    {
        return std::unexpected(err_unsupported(
            "buffer.address needs Feature.BUFFER_ADDRESS, which this Context did not enable. "
            "Pass it in Context(features=[bz.Feature.BUFFER_ADDRESS]) — the flag has to be set "
            "when the buffer is created, so enabling it later would not help this buffer."));
    }
    VkBufferDeviceAddressInfo info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .pNext = nullptr, .buffer = get()};
    return context->vk().vkGetBufferDeviceAddress(context->device(), &info);
}

// ── StaticBuffer ──────────────────────────────────────────────────────────────

StaticBuffer::StaticBuffer(std::shared_ptr<Context> context, VkBuffer buffer, VmaAllocation allocation, size_t size)
    : context_(std::move(context)),
      buffer_(buffer),
      allocation_(allocation),
      size_(size)
{
}

StaticBuffer::~StaticBuffer()
{
    if (buffer_ != VK_NULL_HANDLE && context_)
    {
        context_->defer_destroy([allocator = context_->allocator(), buffer = buffer_, allocation = allocation_]
                                { vmaDestroyBuffer(allocator, buffer, allocation); });
    }
}

std::expected<std::vector<std::byte>, Error> StaticBuffer::read_bytes()
{
    // The fill is a separate submit with no barrier against this one, and
    // two submits on one queue are not ordered by anything but a semaphore.
    // Without this the first read after create_buffer is a race that
    // returns uninitialized memory on exactly the drivers that overlap.
    wait();

    auto staging_pair = create_staging_buffer(*context_, size_, Staging::Readback);
    if (!staging_pair)
    {
        return std::unexpected(staging_pair.error());
    }
    auto [staging, staging_alloc] = *staging_pair;

    auto submitted = immediate_submit(
        *context_,
        [&](VkCommandBuffer cmd)
        {
            VkBufferCopy region{.srcOffset = 0, .dstOffset = 0, .size = size_};
            context_->vk().vkCmdCopyBuffer(cmd, buffer_, staging, 1, &region);
        });
    if (!submitted)
    {
        vmaDestroyBuffer(context_->allocator(), staging, staging_alloc);
        return std::unexpected(submitted.error());
    }

    std::vector<std::byte> out(size_);
    void* mapped = nullptr;
    if (auto e = check(
            vmaMapMemory(context_->allocator(), staging_alloc, &mapped),
            "map buffer readback staging",
            ErrorCode::Resource))
    {
        vmaDestroyBuffer(context_->allocator(), staging, staging_alloc);
        return std::unexpected(*e);
    }
    std::memcpy(out.data(), mapped, size_);
    vmaUnmapMemory(context_->allocator(), staging_alloc);
    vmaDestroyBuffer(context_->allocator(), staging, staging_alloc);
    return out;
}

std::expected<std::shared_ptr<StaticBuffer>, Error> StaticBuffer::create(
    Context& context,
    const void* data,
    size_t data_size,
    BufferUsage type)
{
    const VkBufferUsageFlags usage = buffer_usage_for(type, context.supports(Feature::BUFFER_ADDRESS));

    // The device buffer comes first since 0.26: it is the allocation that
    // can fail on size, and failing before any host memory is reserved for
    // it is the difference between an error and a machine that swaps.
    VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = data_size,
        .usage = usage,
        .sharingMode = context.sharing().mode,
        .queueFamilyIndexCount = context.sharing().family_count,
        .pQueueFamilyIndices = context.sharing().families};

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    VkBuffer buffer = nullptr;
    VmaAllocation allocation = nullptr;
    if (auto e = check(
            vmaCreateBuffer(context.allocator(), &bufferInfo, &allocInfo, &buffer, &allocation, nullptr),
            "create device local buffer",
            ErrorCode::Resource))
    {
        return std::unexpected(*e);
    }

    auto serial = fill_from_host(context, buffer, data, data_size);
    if (!serial)
    {
        vmaDestroyBuffer(context.allocator(), buffer, allocation);
        return std::unexpected(serial.error());
    }

    auto result = std::make_shared<StaticBuffer>(context.shared_from_this(), buffer, allocation, data_size);
    result->set_upload_serial(*serial);
    if (*serial != 0)
    {
        context.note_upload_serial(*serial);
    }
    return result;
}

std::expected<std::uint64_t, Error> StaticBuffer::fill_from_host(
    Context& context,
    VkBuffer buffer,
    const void* data,
    VkDeviceSize data_size)
{
    if (data == nullptr || data_size == 0)
    {
        return 0; // A sized buffer with no contents yet.
    }

    std::uint64_t serial = 0;
    for (VkDeviceSize offset = 0; offset < data_size; offset += kStagingChunk)
    {
        const VkDeviceSize span = (std::min)(kStagingChunk, data_size - offset);
        const bool last = offset + span >= data_size;

        auto staging_pair =
            create_staging_buffer(context, span, Staging::Upload, static_cast<const std::byte*>(data) + offset);
        if (!staging_pair)
        {
            return std::unexpected(staging_pair.error());
        }
        auto [staging, staging_allocation] = *staging_pair;

        // Asynchronous since 0.18.0. The staging fill above already happened
        // on this thread (the bytes come from Python, so they cannot be
        // copied anywhere else), which leaves the queue drain as the only
        // cost of the old blocking path — and 30 meshes meant 30 full queue
        // drains at startup. The copy is submitted here, so every failure is
        // still raised at the create_buffer call; only the wait is gone.
        auto piece = deferred_submit(
            context,
            [&](VkCommandBuffer cmd)
            {
                VkBufferCopy region{.srcOffset = 0, .dstOffset = offset, .size = span};
                context.vk().vkCmdCopyBuffer(cmd, staging, buffer, 1, &region);
            });
        if (!piece)
        {
            vmaDestroyBuffer(context.allocator(), staging, staging_allocation);
            return std::unexpected(piece.error());
        }
        serial = *piece;

        if (last)
        {
            // The GPU reads this one after we return, so it retires on the
            // serial rather than here.
            context.defer_destroy([allocator = context.allocator(), staging, staging_allocation]
                                  { vmaDestroyBuffer(allocator, staging, staging_allocation); });
        }
        else if (auto r = context.wait_for_serial(serial); !r)
        {
            vmaDestroyBuffer(context.allocator(), staging, staging_allocation);
            return std::unexpected(r.error());
        }
        else
        {
            vmaDestroyBuffer(context.allocator(), staging, staging_allocation);
        }
    }
    return serial;
}

// ── DynamicBuffer ─────────────────────────────────────────────────────────────

DynamicBuffer::DynamicBuffer(
    std::shared_ptr<Context> context,
    std::vector<VkBuffer> buffers,
    std::vector<VmaAllocation> allocations,
    size_t size,
    BufferUsage type)
    : context_(std::move(context)),
      buffers_(std::move(buffers)),
      allocations_(std::move(allocations)),
      size_(size),
      type_(type)
{
}

DynamicBuffer::~DynamicBuffer()
{
    if (context_)
    {
        context_->defer_destroy(
            [allocator = context_->allocator(), buffers = std::move(buffers_), allocations = std::move(allocations_)]
            {
                for (size_t i = 0; i < buffers.size(); ++i)
                {
                    if (buffers[i] != VK_NULL_HANDLE)
                    {
                        vmaDestroyBuffer(allocator, buffers[i], allocations[i]);
                    }
                }
            });
    }
}

std::expected<std::vector<std::byte>, Error> DynamicBuffer::read_bytes()
{
    const uint32_t frame = context_->frame_index();
    void* mapped = nullptr;
    if (auto e = check(
            vmaMapMemory(context_->allocator(), allocations_[frame], &mapped),
            "map dynamic buffer for read",
            ErrorCode::Resource))
    {
        return std::unexpected(*e);
    }
    std::vector<std::byte> out(size_);
    std::memcpy(out.data(), mapped, size_);
    vmaUnmapMemory(context_->allocator(), allocations_[frame]);
    return out;
}

std::expected<void, Error> DynamicBuffer::update(std::span<const std::byte> data, size_t offset)
{
    if (!fits_within(offset, data.size(), size_))
    {
        return std::unexpected(err_resource(
            std::format(
                "Update of {} bytes at offset {} exceeds the buffer size of {} bytes", data.size(), offset, size_)));
    }
    uint32_t frame = context_->frame_index();
    void* mappedData = nullptr;
    if (auto e = check(
            vmaMapMemory(context_->allocator(), allocations_[frame], &mappedData),
            "map dynamic buffer memory for update",
            ErrorCode::Resource))
    {
        return std::unexpected(*e);
    }
    std::memcpy(static_cast<std::byte*>(mappedData) + offset, data.data(), data.size());
    vmaUnmapMemory(context_->allocator(), allocations_[frame]);
    return {};
}

std::expected<std::shared_ptr<DynamicBuffer>, Error> DynamicBuffer::create(
    Context& context,
    const void* data,
    size_t data_size,
    BufferUsage type)
{
    const VkBufferUsageFlags usage = buffer_usage_for(type, context.supports(Feature::BUFFER_ADDRESS));

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = data_size,
        .usage = usage,
        .sharingMode = context.sharing().mode,
        .queueFamilyIndexCount = context.sharing().family_count,
        .pQueueFamilyIndices = context.sharing().families};

    const std::uint32_t frames = context.frames_in_flight();
    std::vector<VkBuffer> buffers(frames, VK_NULL_HANDLE);
    std::vector<VmaAllocation> allocations(frames, VK_NULL_HANDLE);

    for (size_t i = 0; i < frames; ++i)
    {
        if (auto e = check(
                vmaCreateBuffer(context.allocator(), &bufferInfo, &allocInfo, &buffers[i], &allocations[i], nullptr),
                std::string("create dynamic ") + buffer_usage_name(type) + " buffer",
                ErrorCode::Resource))
        {
            for (size_t j = 0; j < i; ++j)
            {
                vmaDestroyBuffer(context.allocator(), buffers[j], allocations[j]);
            }
            return std::unexpected(*e);
        }

        if (data != nullptr && data_size > 0)
        {
            void* mappedData = nullptr;
            vmaMapMemory(context.allocator(), allocations[i], &mappedData);
            std::memcpy(mappedData, data, data_size);
            vmaUnmapMemory(context.allocator(), allocations[i]);
        }
    }
    return std::make_shared<DynamicBuffer>(
        context.shared_from_this(), std::move(buffers), std::move(allocations), data_size, type);
}

// ── The one funnel ────────────────────────────────────────────────────────────

std::expected<std::shared_ptr<Buffer>, Error> Buffer::create(
    Context& context,
    const void* data,
    size_t data_size,
    BufferUsage type,
    MemoryUsage usage)
{
    // The single funnel for both kinds, which is why the type is recorded here and
    // not in each derived create: one place, and it cannot drift between them.
    std::expected<std::shared_ptr<Buffer>, Error> buffer;
    if (usage == MemoryUsage::DYNAMIC)
    {
        auto made = DynamicBuffer::create(context, data, data_size, type);
        if (!made)
        {
            return std::unexpected(made.error());
        }
        buffer = *made;
    }
    else
    {
        auto made = StaticBuffer::create(context, data, data_size, type);
        if (!made)
        {
            return std::unexpected(made.error());
        }
        buffer = *made;
    }
    (*buffer)->set_buffer_usage(type);
    return buffer;
}
