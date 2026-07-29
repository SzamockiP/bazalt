#pragma once
#include <volk.h>
#include <vk_mem_alloc.h>
#include <cstddef>
#include <format>
#include <span>
#include <string>
#include <memory>
#include <expected>
#include <cstring>
#include <vector>
#include "Context.hpp"
#include "ImmediateSubmit.hpp"

enum class BufferType
{
    VERTEX,
    INDEX,
    UNIFORM,
    STORAGE
};

enum class DataType
{
    FLOAT,
    UINT32,
    UINT16,
    INT32
};

enum class MemoryUsage
{
    STATIC,
    DYNAMIC
};

class Buffer
{
public:
    virtual ~Buffer() = default;

    virtual VkBuffer get() const = 0;
    virtual size_t size() const = 0;
    virtual bool is_dynamic() const
    {
        return false;
    }
    virtual VkBuffer get_for_frame(uint32_t /*frame*/) const
    {
        return get();
    }

    // Fails through the unified Error channel, not a raw exception: at the
    // pybind boundary this surfaces as bz.ResourceError, so `except BazaltError`
    // actually catches it.
    virtual std::expected<void, Error> update(std::span<const std::byte> /*data*/, size_t /*offset*/ = 0)
    {
        return std::unexpected(err_resource(
            "update() is only supported on DYNAMIC buffers; "
            "create the buffer with MemoryUsage.DYNAMIC instead"));
    }

    // Copies the buffer's contents back to host memory. Buffers carry no
    // format (unlike Images), so the caller supplies the dtype at the binding
    // layer. STATIC buffers take a blocking GPU round trip; DYNAMIC ones map
    // the current frame's copy directly.
    virtual std::expected<std::vector<std::byte>, Error> read_bytes() = 0;

    // Which Context this object belongs to. Multi-context (0.15) made "a
    // resource from the other Context" a reachable mistake, and its symptom
    // without a check is a driver crash or a validation message from Vulkan
    // rather than from bazalt; the binding layer compares owners at record time.
    virtual const Context* owner() const = 0;

    // ── The buffer IS its own upload future (0.18.0) ──────────────────────────
    //
    // A STATIC buffer is filled by a staging copy that is submitted at create
    // time and NOT waited for. The serial that copy signals is the whole
    // mechanism: a submit that reads the buffer waits on it GPU-side (the
    // command buffer remembers which buffers a recording touches), read_bytes
    // waits on it CPU-side, and ready/wait are the explicit-control verbs.
    //
    // 0 means "nothing pending", which is the honest answer for a DYNAMIC
    // buffer (host-visible, written by mapping, never staged) and for a STATIC
    // one whose copy is already complete.
    virtual std::uint64_t upload_serial() const
    {
        return 0;
    }
    virtual bool ready() const
    {
        return true;
    }
    virtual void wait()
    {
    }

    // Remembered so bind_index_buffer doesn't have to assume. It used to hardcode
    // VK_INDEX_TYPE_UINT32 while create_buffer happily accepted UINT16 indices,
    // which were then read back at half the count with no error.
    DataType data_type() const
    {
        return data_type_;
    }
    void set_data_type(DataType type)
    {
        data_type_ = type;
    }

    VkIndexType index_type() const
    {
        return data_type_ == DataType::UINT16 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
    }

    // Remembered for the same reason data_type_ is, and it lives on the base rather
    // than on DynamicBuffer (where it used to) because the indirect draw verbs need
    // it from a plain Buffer&: only a STORAGE buffer carries
    // VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, so this turns a layers-only VUID into a
    // bazalt error that names the fix. Set once, in Buffer::create.
    BufferType buffer_type() const
    {
        return buffer_type_;
    }
    void set_buffer_type(BufferType type)
    {
        buffer_type_ = type;
    }

    static std::expected<std::shared_ptr<Buffer>, Error> create(
        Context& context,
        const void* data,
        size_t data_size,
        BufferType type,
        MemoryUsage usage);

protected:
    DataType data_type_ = DataType::FLOAT;
    BufferType buffer_type_ = BufferType::UNIFORM;
};

class StaticBuffer : public Buffer
{
public:
    const Context* owner() const override
    {
        return context_.get();
    }

    StaticBuffer(std::shared_ptr<Context> context, VkBuffer buffer, VmaAllocation allocation, size_t size)
        : context_(context),
          buffer_(buffer),
          allocation_(allocation),
          size_(size)
    {
    }

    // Deferred: the handle may still be referenced by an in-flight frame —
    // cmd.begin() drops the shared_ptrs that kept it alive while the previous
    // frame is still being consumed by the GPU.
    ~StaticBuffer() override
    {
        if (buffer_ != VK_NULL_HANDLE && context_)
        {
            context_->defer_destroy([allocator = context_->allocator(), buffer = buffer_, allocation = allocation_]
                                    { vmaDestroyBuffer(allocator, buffer, allocation); });
        }
    }

    StaticBuffer(const StaticBuffer&) = delete;
    StaticBuffer& operator=(const StaticBuffer&) = delete;

    VkBuffer get() const override
    {
        return buffer_;
    }
    size_t size() const override
    {
        return size_;
    }

    std::uint64_t upload_serial() const override
    {
        return upload_serial_;
    }
    bool ready() const override
    {
        return context_->completed_submit_serial() >= upload_serial_;
    }
    void wait() override
    {
        static_cast<void>(context_->wait_for_serial(upload_serial_));
    }
    void set_upload_serial(std::uint64_t serial)
    {
        upload_serial_ = serial;
    }

    // Blocking round trip through a readback staging buffer — device-local
    // memory is not mappable. A debugging/test path (SSBO results, mostly).
    std::expected<std::vector<std::byte>, Error> read_bytes() override
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

    static std::expected<std::shared_ptr<StaticBuffer>, Error> create(
        Context& context,
        const void* data,
        size_t data_size,
        BufferType type)
    {
        // TRANSFER_SRC so read() can copy the contents back out.
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        switch (type)
        {
            case BufferType::VERTEX:
                usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
                break;
            case BufferType::INDEX:
                usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
                break;
            // VERTEX also: a compute shader writing vertices into an SSBO that
            // the graphics pipeline then consumes via bind_vertex_buffer is the
            // canonical compute->graphics hand-off (examples/11_particles).
            //
            // INDIRECT too, and unconditionally (0.19). The whole point of an
            // indirect draw is a compute shader writing the draw arguments, so the
            // type that carries STORAGE_BUFFER is the type that carries this.
            // Gating it behind a fifth BufferType would make "which buffers can be
            // indirect" a second rule to remember for a usage bit that costs
            // nothing — the same reasoning that gave DYNAMIC buffers the transfer
            // bits in 0.18.
            case BufferType::STORAGE:
                usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                         VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
                break;
            // Constant data that never changes (e.g. baked matrices) is a legitimate
            // STATIC uniform buffer. Without this the buffer was created with only
            // TRANSFER usage and failed at bind time with a cryptic validation error.
            case BufferType::UNIFORM:
                usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
                break;
            default:
                break;
        }

        auto staging_pair = create_staging_buffer(context, data_size, Staging::Upload, data);
        if (!staging_pair)
        {
            return std::unexpected(staging_pair.error());
        }
        auto [stagingBuffer, stagingAllocation] = *staging_pair;

        VkBufferCreateInfo bufferInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = data_size,
            .usage = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr};

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        VkBuffer buffer;
        VmaAllocation allocation;
        if (auto e = check(
                vmaCreateBuffer(context.allocator(), &bufferInfo, &allocInfo, &buffer, &allocation, nullptr),
                "create device local buffer",
                ErrorCode::Resource))
        {
            vmaDestroyBuffer(context.allocator(), stagingBuffer, stagingAllocation);
            return std::unexpected(*e);
        }

        // Asynchronous since 0.18.0. The staging fill above already happened on
        // this thread (the bytes come from Python, so they cannot be copied
        // anywhere else), which leaves the queue drain as the only cost of
        // the old blocking path — and 30 meshes meant 30 full queue drains at
        // startup. The copy is submitted here, so every failure below is still
        // raised at the create_buffer call; only the wait is gone.
        auto serial = deferred_submit(
            context,
            [&](VkCommandBuffer cmd)
            {
                VkBufferCopy copyRegion{.srcOffset = 0, .dstOffset = 0, .size = data_size};
                context.vk().vkCmdCopyBuffer(cmd, stagingBuffer, buffer, 1, &copyRegion);
            });

        if (!serial)
        {
            vmaDestroyBuffer(context.allocator(), stagingBuffer, stagingAllocation);
            vmaDestroyBuffer(context.allocator(), buffer, allocation);
            return std::unexpected(serial.error());
        }
        // The GPU reads the staging buffer after this returns, so it retires on
        // the serial rather than here.
        context.defer_destroy([allocator = context.allocator(), stagingBuffer, stagingAllocation]
                              { vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation); });

        auto result = std::make_shared<StaticBuffer>(context.shared_from_this(), buffer, allocation, data_size);
        result->set_upload_serial(*serial);
        context.note_upload_serial(*serial);
        return result;
    }

private:
    std::shared_ptr<Context> context_;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    size_t size_ = 0;
    // Which submit fills this buffer. Plain, not atomic: it is written once by
    // create() before the shared_ptr escapes, and only read afterwards.
    std::uint64_t upload_serial_ = 0;
};

class DynamicBuffer : public Buffer
{
public:
    const Context* owner() const override
    {
        return context_.get();
    }

    // One buffer per frame in flight; the count is a runtime property of the
    // Context now, so these are vectors sized at creation.
    DynamicBuffer(
        std::shared_ptr<Context> context,
        std::vector<VkBuffer> buffers,
        std::vector<VmaAllocation> allocations,
        size_t size,
        BufferType type)
        : context_(context),
          buffers_(std::move(buffers)),
          allocations_(std::move(allocations)),
          size_(size),
          type_(type)
    {
    }

    ~DynamicBuffer() override
    {
        if (context_)
        {
            context_->defer_destroy(
                [allocator = context_->allocator(),
                 buffers = std::move(buffers_),
                 allocations = std::move(allocations_)]
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

    DynamicBuffer(const DynamicBuffer&) = delete;
    DynamicBuffer& operator=(const DynamicBuffer&) = delete;

    VkBuffer get() const override
    {
        return buffers_[context_->frame_index()];
    }

    size_t size() const override
    {
        return size_;
    }
    bool is_dynamic() const override
    {
        return true;
    }
    VkBuffer get_for_frame(uint32_t frame) const override
    {
        return buffers_[frame];
    }
    // Host-visible: map the current frame's copy, no GPU round trip. Note the
    // GPU may not have consumed it yet — this reads what update() wrote.
    std::expected<std::vector<std::byte>, Error> read_bytes() override
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

    std::expected<void, Error> update(std::span<const std::byte> data, size_t offset = 0) override
    {
        if (!fits_within(offset, data.size(), size_))
        {
            return std::unexpected(err_resource(
                std::format(
                    "Update of {} bytes at offset {} exceeds the buffer size of {} bytes",
                    data.size(),
                    offset,
                    size_)));
        }
        uint32_t frame = context_->frame_index();
        void* mappedData;
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

    static std::expected<std::shared_ptr<DynamicBuffer>, Error> create(
        Context& context,
        const void* data,
        size_t data_size,
        BufferType type)
    {
        // The transfer bits ride along so cmd.copy_buffer / cmd.fill_buffer work
        // on a DYNAMIC buffer too (0.18). A STATIC buffer has carried both since
        // it was written, and refusing them here would make "which buffers can
        // the GPU copy into" a second rule to remember for no gain: transfer
        // usage costs nothing on memory that is already host-visible.
        VkBufferUsageFlags usage = (type == BufferType::STORAGE) ? VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                                                 : VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        // Indirect for the same reason a STATIC storage buffer gets it (0.19), and
        // it matters here too: draw arguments the CPU rewrites every frame are a
        // DYNAMIC buffer, and that is the case where the count is not GPU-derived.
        if (type == BufferType::STORAGE)
        {
            usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        }

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
            .pQueueFamilyIndices = nullptr};

        const std::uint32_t frames = context.frames_in_flight();
        std::vector<VkBuffer> buffers(frames, VK_NULL_HANDLE);
        std::vector<VmaAllocation> allocations(frames, VK_NULL_HANDLE);

        for (size_t i = 0; i < frames; ++i)
        {
            if (auto e = check(
                    vmaCreateBuffer(
                        context.allocator(), &bufferInfo, &allocInfo, &buffers[i], &allocations[i], nullptr),
                    std::string("create ") + (type == BufferType::STORAGE ? "storage" : "uniform") + " buffer",
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
                void* mappedData;
                vmaMapMemory(context.allocator(), allocations[i], &mappedData);
                std::memcpy(mappedData, data, data_size);
                vmaUnmapMemory(context.allocator(), allocations[i]);
            }
        }
        return std::make_shared<DynamicBuffer>(
            context.shared_from_this(), std::move(buffers), std::move(allocations), data_size, type);
    }

private:
    std::shared_ptr<Context> context_;
    std::vector<VkBuffer> buffers_;
    std::vector<VmaAllocation> allocations_;
    size_t size_ = 0;
    BufferType type_ = BufferType::UNIFORM;
};

// Keep backward-compatible alias
using UniformBuffer = DynamicBuffer;

inline std::expected<std::shared_ptr<Buffer>, Error> Buffer::create(
    Context& context,
    const void* data,
    size_t data_size,
    BufferType type,
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
    (*buffer)->set_buffer_type(type);
    return buffer;
}
