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
        uint32_t index = 0);

    // Write a storage image to this descriptor set (all copies). No sampler:
    // a storage image is read/written by coordinate (imageLoad/imageStore), and
    // its descriptor layout is GENERAL — the only layout a storage image may be
    // accessed in. The auto-barrier tracker transitions the image to GENERAL
    // before the dispatch, so the recorded layout here is always what the GPU
    // finds at execute time.
    std::expected<void, Error> set_storage_image(uint32_t binding, std::shared_ptr<Image> image, uint32_t index = 0);

    // Write a buffer to this descriptor set
    // For frame descriptor sets + DynamicBuffer: writes per-frame buffer to each copy
    // For static descriptor sets + DynamicBuffer: bz.ResourceError
    std::expected<void, Error> set_buffer(uint32_t binding, std::shared_ptr<Buffer> buffer, uint32_t index = 0);

    // Get the VkDescriptorSet for the given frame
    VkDescriptorSet get(uint32_t currentFrame) const;

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
    std::expected<Pipeline::BindingInfo, Error> check_binding(uint32_t binding, uint32_t index, const char* what) const;

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
        std::shared_ptr<Sampler> sampler);

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
        uint32_t storageImageCount);

    // Auto mode (0.23): no sizes at all. Blocks are allocated as sets are, each
    // sized from the layout being served, so the caller stops doing Vulkan's
    // arithmetic — which depended on frames_in_flight, a number that never
    // appeared in the old call. A whole count=N array must fit its block
    // (MoltenVK refuses a partial fit, 0.22), and sizing from the layout is
    // what guarantees it.
    static std::expected<std::shared_ptr<DescriptorPool>, Error> create_auto(Context& context);

    // Deferred, so it lands in the queue after every set's free (sets hold the
    // pool, so their destructors necessarily run first).
    ~DescriptorPool();

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
        const std::shared_ptr<Pipeline>& pipeline,
        uint32_t setIndex,
        bool frame_set);

    VkResult try_allocate_(
        VkDescriptorPool from,
        const std::vector<VkDescriptorSetLayout>& layouts,
        std::vector<VkDescriptorSet>& sets);

    // A new auto-mode block. Sized max(default, this request) per descriptor
    // type: the default keeps small sets from each costing a VkDescriptorPool,
    // and the request half is what lets one bindless array larger than any
    // default land in a block of its own.
    std::expected<VkDescriptorPool, Error> grow_for_(const Pipeline& pipeline, uint32_t setIndex, uint32_t set_count);

    static std::expected<VkDescriptorPool, Error> create_block_(
        Context& context,
        uint32_t maxSets,
        const std::vector<VkDescriptorPoolSize>& poolSizes);

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
    // What each block was DECLARED to hold, beside the handle. Declared rather
    // than remaining, on purpose: it needs no per-allocation bookkeeping, and
    // being wrong low simply falls through to the retry that already exists
    // (0.26).
    struct Block
    {
        VkDescriptorPool pool = VK_NULL_HANDLE;
        std::unordered_map<VkDescriptorType, uint32_t> capacity;
        uint32_t max_sets = 0;
    };
    std::vector<Block> blocks_;
    bool grows_ = false;

    // What one request needs, per descriptor type. Factored out of grow_for_ so
    // the question "does this fit" and the answer "then size a block like this"
    // cannot drift apart.
    static std::unordered_map<VkDescriptorType, uint32_t> needed_(
        const Pipeline& pipeline,
        uint32_t setIndex,
        uint32_t set_count);

    // Whether the newest block could not possibly satisfy this request, so the
    // allocation attempt would be a guaranteed VK_ERROR_OUT_OF_POOL_MEMORY —
    // which the validation layers report before the retry quietly fixes it.
    static bool exceeds_(
        const Block& block,
        const std::unordered_map<VkDescriptorType, uint32_t>& needed,
        uint32_t set_count);
};
