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

// Bits since 0.30, so one buffer can be several things: a compute shader
// writes vertices into a STORAGE buffer the draw then binds as VERTEX, or a
// uniform block the GPU fills. Chosen now rather than later because
// test_stubs.py and 1.0 freeze the member ints, and VERTEX = 0 can never
// become a bit afterwards.
enum class BufferUsage : std::uint32_t
{
    VERTEX = 1u << 0,
    INDEX = 1u << 1,
    UNIFORM = 1u << 2,
    STORAGE = 1u << 3
};

inline constexpr BufferUsage operator|(BufferUsage a, BufferUsage b)
{
    return static_cast<BufferUsage>(static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}

inline constexpr bool has(BufferUsage usage, BufferUsage bit)
{
    return (static_cast<std::uint32_t>(usage) & static_cast<std::uint32_t>(bit)) != 0;
}

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

// "VERTEX|STORAGE": the spelling a user wrote, for the messages that name a
// buffer's usage. A pybind enum accepts any int, so unknown bits print as one.
inline std::string buffer_usage_name(BufferUsage usage)
{
    std::string out;
    const auto add = [&](BufferUsage bit, const char* name)
    {
        if (has(usage, bit))
        {
            out += out.empty() ? name : std::string("|") + name;
        }
    };
    add(BufferUsage::VERTEX, "VERTEX");
    add(BufferUsage::INDEX, "INDEX");
    add(BufferUsage::UNIFORM, "UNIFORM");
    add(BufferUsage::STORAGE, "STORAGE");
    return out.empty() ? "unknown" : out;
}

// What a buffer of this usage must be able to do. One function for both memory
// usages, because the answer is a property of the USAGE and nothing else — and
// because the two had drifted: StaticBuffer switched on the type while
// DynamicBuffer asked "is it STORAGE?" and called everything else a uniform
// buffer. A DYNAMIC vertex buffer — geometry rebuilt every frame, which is what
// DYNAMIC is for — therefore had no VERTEX_BUFFER bit and failed at bind time
// (VUID-vkCmdBindVertexBuffers-pBuffers-00627). Found by running
// examples/28_gpu_culling, not by a test: nothing in the suite made one.
//
// The transfer bits ride along on both, so cmd.copy_buffer / cmd.fill_buffer work
// on either (0.18). Refusing them on host-visible memory would make "which
// buffers can the GPU copy into" a second rule to remember, for nothing.
// `device_address` rides here rather than at the call sites (0.26) for the same
// reason the two usages were merged above: the bit has to be on EVERY buffer a
// Context with Feature::BUFFER_ADDRESS makes, and a call site is a place for
// the two kinds of buffer to disagree. They already had, once — the static path
// got the flag and the dynamic one did not, which a DYNAMIC buffer only reports
// at the vkGetBufferDeviceAddress that needs it.
inline constexpr VkBufferUsageFlags buffer_usage_for(BufferUsage usage_bits, bool device_address = false)
{
    // TRANSFER_SRC so read() can copy the contents back out.
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (has(usage_bits, BufferUsage::VERTEX))
    {
        usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    }
    if (has(usage_bits, BufferUsage::INDEX))
    {
        usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    }
    // VERTEX also: a compute shader writing vertices into an SSBO that
    // the graphics pipeline then consumes via bind_vertex_buffer is the
    // canonical compute->graphics hand-off (examples/11_particles).
    //
    // INDIRECT too, and unconditionally (0.19). The whole point of an
    // indirect draw is a compute shader writing the draw arguments, so the
    // usage that carries STORAGE_BUFFER is the usage that carries this.
    // Gating it behind a fifth member would make "which buffers can be
    // indirect" a second rule to remember for a usage bit that costs
    // nothing — the same reasoning that gave DYNAMIC buffers the transfer
    // bits in 0.18. It matters for a DYNAMIC storage buffer too: draw
    // arguments the CPU rewrites every frame are exactly that.
    // INDEX joined them in 0.29, and the argument is the VERTEX one seen
    // from the other end: a compute shader that compacts or rewrites an
    // index list is the same hand-off as one that writes vertices, and it
    // was the only third of GPU-driven work bazalt could not spell.
    if (has(usage_bits, BufferUsage::STORAGE))
    {
        usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                 VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    }
    // Constant data that never changes (e.g. baked matrices) is a legitimate
    // STATIC uniform buffer. Without this the buffer was created with only
    // TRANSFER usage and failed at bind time with a cryptic validation error.
    if (has(usage_bits, BufferUsage::UNIFORM))
    {
        usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    }
    // The allocator opted in for the whole Context (see
    // create_allocator_and_pool_), so this is the buffer's half of the same
    // decision: no per-buffer kwarg, because its wrong setting would only
    // surface much later, at the call that wants the address.
    if (device_address)
    {
        usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    return usage;
}

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
    virtual std::expected<void, Error> update(std::span<const std::byte> /*data*/, size_t /*offset*/ = 0);

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

    // Where this buffer lives on the device (0.26). Push it as a push constant
    // and a shader reaches the memory through a `buffer_reference` pointer
    // instead of a descriptor — which is the point, because a descriptor may
    // only see limits().max_storage_buffer of it and an address has no such
    // ceiling.
    //
    // Not virtual: get() already answers "which VkBuffer is current", so a
    // DYNAMIC buffer's address follows its per-frame copy for free. That also
    // means a DYNAMIC address is only good for the frame it was read in.
    //
    // BLOCKS on a pending upload, which is why it is not const. An address is
    // not a binding, so nothing downstream can see that a submit reads this
    // buffer — the record-time tracking that makes every other path wait for
    // the staging copy has nothing to key on. This is the one place the caller
    // must pass through to use the memory at all, so the wait goes here. It
    // costs a timeline query once and nothing on every read after that.
    std::expected<VkDeviceAddress, Error> address();

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
    BufferUsage buffer_usage() const
    {
        return buffer_usage_;
    }
    void set_buffer_usage(BufferUsage usage)
    {
        buffer_usage_ = usage;
    }

    static std::expected<std::shared_ptr<Buffer>, Error> create(
        Context& context,
        const void* data,
        size_t data_size,
        BufferUsage type,
        MemoryUsage usage);

protected:
    DataType data_type_ = DataType::FLOAT;
    BufferUsage buffer_usage_ = BufferUsage::UNIFORM;
};

class StaticBuffer : public Buffer
{
public:
    const Context* owner() const override
    {
        return context_.get();
    }

    StaticBuffer(std::shared_ptr<Context> context, VkBuffer buffer, VmaAllocation allocation, size_t size);

    // Deferred: the handle may still be referenced by an in-flight frame —
    // cmd.begin() drops the shared_ptrs that kept it alive while the previous
    // frame is still being consumed by the GPU.
    ~StaticBuffer() override;

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
    std::expected<std::vector<std::byte>, Error> read_bytes() override;

    static std::expected<std::shared_ptr<StaticBuffer>, Error> create(
        Context& context,
        const void* data,
        size_t data_size,
        BufferUsage type);

private:
    // How much host memory one upload may hold at a time. A staging buffer the
    // size of the upload is what capped a buffer at whatever the machine could
    // spare twice over, and it is the reason this is a loop rather than one
    // memcpy: filling VRAM should cost VRAM, not VRAM plus as much RAM.
    //
    // 64 MiB is large enough that the per-piece submit is noise next to the
    // copy, and small enough to be nothing on any machine that runs Vulkan.
    static constexpr VkDeviceSize kStagingChunk = 64ull << 20;

    // Copies `data` into `buffer` through bounded staging, and returns the
    // serial of the LAST piece — 0 when there was nothing to copy.
    //
    // Only the last piece stays in flight. Every earlier one is waited for and
    // destroyed on the spot, because handing them to defer_destroy would keep
    // them all alive until the Context next drains, which is the whole thing
    // this loop is avoiding. So a buffer under one chunk behaves exactly as it
    // did before 0.26 (one staging buffer, one submit, no wait) and a larger
    // one trades that asynchrony for a fixed memory ceiling.
    static std::expected<std::uint64_t, Error> fill_from_host(
        Context& context,
        VkBuffer buffer,
        const void* data,
        VkDeviceSize data_size);

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
        BufferUsage type);

    ~DynamicBuffer() override;

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
    std::expected<std::vector<std::byte>, Error> read_bytes() override;

    std::expected<void, Error> update(std::span<const std::byte> data, size_t offset = 0) override;

    static std::expected<std::shared_ptr<DynamicBuffer>, Error> create(
        Context& context,
        const void* data,
        size_t data_size,
        BufferUsage type);

private:
    std::shared_ptr<Context> context_;
    std::vector<VkBuffer> buffers_;
    std::vector<VmaAllocation> allocations_;
    size_t size_ = 0;
    BufferUsage type_ = BufferUsage::UNIFORM;
};

// Keep backward-compatible alias
using UniformBuffer = DynamicBuffer;
