#pragma once
#include <volk.h>
#include <vk_mem_alloc.h>

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <expected>
#include <format>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "stb_image.h"
#include "Context.hpp"
#include "Error.hpp"
#include "Format.hpp"
#include "Image.hpp"
#include "ImmediateSubmit.hpp"

// The async transport behind ctx.load_image().
//
// One worker thread decodes files, fills staging buffers and submits copy +
// mipgen work to the *graphics* queue (variant A: no dedicated transfer queue,
// no ownership-transfer barriers — the Python API would be identical either
// way, and this removes 100% of the actual pain, the vkQueueWaitIdle per
// upload). Every submit signals the Context's submission timeline, which is
// how frames wait for their textures GPU-side with zero CPU stalls.
//
// The worker NEVER touches the GIL — the deadlock class this rules out is why
// the invariant is stated here. One thread on purpose: stbi_failure_reason()
// is a global buffer; a pool would need that revisited.
class UploadManager final : public UploadManagerBase
{
public:
    explicit UploadManager(Context& context)
        : context_(context)
    {
        VkCommandPoolCreateInfo poolInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
            .queueFamilyIndex = context.graphics_queue_family()};
        // Command pools are externally synchronized; the worker gets its own.
        context.vk().vkCreateCommandPool(context.device(), &poolInfo, nullptr, &pool_);

        worker_ = std::jthread([this](std::stop_token stop) { run_(stop); });
    }

    // Abandons undecoded jobs (their images end Failed so any waiter wakes),
    // finishes at most the job in flight, joins, and tears down the pool.
    // ~Context runs this before vkDeviceWaitIdle, while the device is alive.
    ~UploadManager() override
    {
        worker_.request_stop();
        {
            std::lock_guard lock(mutex_);
            for (auto& job : jobs_)
            {
                job.image->set_upload_failed("Context destroyed before this upload was decoded");
            }
            jobs_.clear();
        }
        cv_.notify_all();
        worker_.join();

        // The pool must not disappear under pending GPU work, and the staging
        // buffers parked in the deletion queue may as well go now. Destroying
        // the pool frees its remaining command buffers implicitly (the worker
        // has joined, so this thread is the pool's sole owner).
        {
            std::lock_guard lock(context_.queue_mutex());
            context_.vk().vkDeviceWaitIdle(context_.device());
        }
        context_.flush_deletion_queue();

        if (pool_ != VK_NULL_HANDLE)
        {
            context_.vk().vkDestroyCommandPool(context_.device(), pool_, nullptr);
        }
    }

    // Main thread: validate the file header synchronously (a missing or
    // mangled file fails HERE, at the call site, and width/height are correct
    // immediately), create the empty image, and hand the decode to the worker.
    std::expected<std::shared_ptr<Image>, Error> load(const std::string& path, bool mipmaps = true)
    {
        int width = 0, height = 0, comp = 0;
        if (!stbi_info(path.c_str(), &width, &height, &comp))
        {
            const char* reason = stbi_failure_reason();
            return std::unexpected(err_resource(
                reason ? std::format("Failed to load image: {} ({})", path, reason)
                       : std::format("Failed to load image: {}", path)));
        }

        const Format format = Format::RGBA8_SRGB;
        const std::uint32_t mips =
            mipmaps && Image::can_generate_mips(context_, format)
                ? Image::full_mip_count(static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height))
                : 1;

        auto image = Image::create_empty(
            context_, static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), format, mips);
        if (!image)
        {
            return image;
        }
        // The layout the upload will leave it in, recorded on THIS thread:
        // Image's layout state belongs to the main thread (see
        // set_upload_submitted).
        (*image)->mark_has_contents(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        (*image)->set_upload_pending();

        {
            std::lock_guard lock(mutex_);
            ++batch_started_;
            jobs_.push_back({*image, path, mips});
        }
        cv_.notify_all();
        return image;
    }

    // Main thread: the same load, from encoded bytes instead of a path.
    //
    // A PNG downloaded over the network, unpacked from a zip or produced by PIL
    // has no path on disk, so the only way to hand it to bazalt was to write a
    // temporary file. The header is validated here for the same reason the file
    // version validates it here: a mangled blob fails at the call site, with
    // width and height correct immediately.
    //
    // Never hot-reloaded, by construction: bazalt has no path to watch.
    std::expected<std::shared_ptr<Image>, Error> load_memory(std::vector<std::byte> blob, bool mipmaps = true)
    {
        int width = 0, height = 0, comp = 0;
        if (!stbi_info_from_memory(
                reinterpret_cast<const stbi_uc*>(blob.data()), static_cast<int>(blob.size()), &width, &height, &comp))
        {
            const char* reason = stbi_failure_reason();
            return std::unexpected(err_resource(
                reason ? std::format("load_image(bytes): not a decodable image ({})", reason)
                       : "load_image(bytes): not a decodable image"));
        }

        const Format format = Format::RGBA8_SRGB;
        const std::uint32_t mips =
            mipmaps && Image::can_generate_mips(context_, format)
                ? Image::full_mip_count(static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height))
                : 1;

        auto image = Image::create_empty(
            context_, static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), format, mips);
        if (!image)
        {
            return image;
        }
        (*image)->mark_has_contents(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        (*image)->set_upload_pending();

        {
            std::lock_guard lock(mutex_);
            ++batch_started_;
            Job job;
            job.image = *image;
            job.mips = mips;
            job.encoded = std::move(blob);
            jobs_.push_back(std::move(job));
        }
        cv_.notify_all();
        return image;
    }

    // Main thread: a layered async load (texture array or cubemap) from N files.
    // Validates every header synchronously (all faces must share a size), builds
    // the empty layered image, and hands the decode + concatenate + single upload
    // to the worker. One batch unit, exactly like a single load.
    std::expected<std::shared_ptr<Image>, Error> load_layered(
        const std::vector<std::string>& paths,
        bool cube,
        bool mipmaps = true)
    {
        if (paths.empty())
        {
            return std::unexpected(err_resource("load_image: the path list is empty"));
        }
        if (cube && paths.size() != 6)
        {
            return std::unexpected(err_resource(
                std::format("load_image(cube=True): a cubemap needs exactly 6 faces, got {}", paths.size())));
        }

        int width = 0, height = 0, comp = 0;
        for (std::size_t i = 0; i < paths.size(); ++i)
        {
            int w = 0, h = 0, c = 0;
            if (!stbi_info(paths[i].c_str(), &w, &h, &c))
            {
                const char* reason = stbi_failure_reason();
                return std::unexpected(err_resource(
                    reason ? std::format("Failed to load image: {} ({})", paths[i], reason)
                           : std::format("Failed to load image: {}", paths[i])));
            }
            if (i == 0)
            {
                width = w;
                height = h;
            }
            else if (w != width || h != height)
            {
                return std::unexpected(err_resource(
                    std::format(
                        "load_image: every layer must be the same size. {} is {}x{}, expected {}x{}",
                        paths[i],
                        w,
                        h,
                        width,
                        height)));
            }
        }
        if (cube && width != height)
        {
            return std::unexpected(
                err_resource(std::format("load_image(cube=True): faces must be square, got {}x{}", width, height)));
        }

        const Format format = Format::RGBA8_SRGB;
        const std::uint32_t mips =
            mipmaps && Image::can_generate_mips(context_, format)
                ? Image::full_mip_count(static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height))
                : 1;

        auto image = Image::create_empty(
            context_,
            static_cast<std::uint32_t>(width),
            static_cast<std::uint32_t>(height),
            format,
            mips,
            static_cast<std::uint32_t>(paths.size()),
            cube);
        if (!image)
        {
            return image;
        }
        // The layout the upload will leave it in, recorded on THIS thread:
        // Image's layout state belongs to the main thread (see
        // set_upload_submitted).
        (*image)->mark_has_contents(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        (*image)->set_upload_pending();

        {
            std::lock_guard lock(mutex_);
            ++batch_started_;
            Job job;
            job.image = *image;
            job.mips = mips;
            job.layers = paths;
            jobs_.push_back(std::move(job));
        }
        cv_.notify_all();
        return image;
    }

    // Hot reload: re-decode `path` into the EXISTING image. The header is NOT
    // re-validated here — this is called from the watcher drain, and a bad file
    // must not throw; the worker decodes, checks the size, and on any problem
    // logs a WARNING and keeps the old contents. Same size and format only in
    // v1 (a resize would need a new VkImage and every descriptor set rewritten).
    // Reuses the image's existing mip count.
    void reload(std::shared_ptr<Image> image, std::string path)
    {
        const std::uint32_t mips = image->mip_levels();
        {
            std::lock_guard lock(mutex_);
            jobs_.push_back({std::move(image), std::move(path), mips, /*reload=*/true});
        }
        cv_.notify_all();
    }

    // Main thread: change the pixels of an existing image. `pixels` is a copy of
    // the caller's rectangle, tightly packed, already validated against the
    // format by the binding layer.
    //
    // Asynchronous, on the same worker as load_image, because the case this
    // exists for is a video frame at 60 fps and a blocking update would spend
    // the frame budget on a memcpy. The queue is FIFO on ONE worker, so two
    // updates of the same image in one frame reach the GPU in call order — that
    // is a guarantee, not an accident of the implementation.
    //
    // `from` is read here, on the main thread, because Image's layout state
    // belongs to this thread and the worker must not touch it.
    void update(
        std::shared_ptr<Image> image,
        std::vector<std::byte> pixels,
        std::uint32_t layer,
        std::uint32_t mip,
        VkOffset2D offset,
        VkExtent2D extent)
    {
        Job job;
        job.from_layout = image->layout_of(layer, mip);
        // The subresource ends sampleable, and saying so here (rather than from
        // the worker) keeps every write to the layout state on the main thread.
        image->mark_subresource_contents(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, layer, 1, mip, 1);
        // Pending BEFORE the job is queued, exactly as load() does. Without it
        // an image created synchronously has upload state None, so img.wait()
        // returns at once and a read() right after an update races the worker —
        // it sees the previous contents, one update behind, every time.
        image->set_upload_pending();
        job.image = std::move(image);
        job.pixels = std::move(pixels);
        job.layer = layer;
        job.mip = mip;
        job.offset = offset;
        job.extent = extent;
        job.update = true;
        {
            std::lock_guard lock(mutex_);
            ++batch_started_;
            jobs_.push_back(std::move(job));
        }
        cv_.notify_all();
    }

    // ── UploadManagerBase (the Python-visible aggregate state) ────────────────

    // An upload that never entered the decode queue: already-decoded bytes that
    // create_buffer / create_image(array) submitted on the calling thread. It is
    // started and submitted in the same breath, so both counters move together
    // and the CPU-side predicate in wait_all() stays balanced.
    void note_direct_upload(std::uint64_t serial) override
    {
        {
            std::lock_guard lock(mutex_);
            ++batch_started_;
            submitted_serials_.push_back(serial);
        }
        cv_.notify_all();
    }

    // Progress of the current batch, 0.0 .. 1.0 (1.0 when idle). The batch
    // resets once fully done, so a second loading screen starts from 0 again.
    double upload_progress() override
    {
        std::lock_guard lock(mutex_);
        if (batch_started_ == 0)
        {
            return 1.0;
        }
        const std::uint64_t done = done_count_();
        if (done == batch_started_)
        {
            reset_batch_();
            return 1.0;
        }
        return static_cast<double>(done) / static_cast<double>(batch_started_);
    }

    void wait_all() override
    {
        std::uint64_t wait_serial = 0;
        {
            std::unique_lock lock(mutex_);
            // First the CPU side: every enqueued job decoded and submitted (or
            // failed) …
            cv_.wait(lock, [&] { return failed_count_ + submitted_serials_.size() == batch_started_; });
            if (!submitted_serials_.empty())
            {
                wait_serial = *std::ranges::max_element(submitted_serials_);
            }
        }
        // … then the GPU side: the timeline reaching the last upload.
        // A failed wait surfaces on the image itself (img.wait() reports it);
        // the aggregate verb has no single resource to blame.
        static_cast<void>(context_.wait_for_serial(wait_serial));
    }

private:
    struct Job
    {
        std::shared_ptr<Image> image;
        std::string path;
        std::uint32_t mips = 1;
        // A hot-reload re-upload into an existing image, not a fresh load. It
        // discards nothing on failure (the old contents stay), never marks the
        // image Failed, and stays out of the batch counters — a re-saved texture
        // must not make a loading bar jump.
        bool reload = false;
        // Non-empty → a layered load (texture array / cubemap): these paths, one
        // per layer, decode into one N-layer staging buffer and one submit.
        std::vector<std::string> layers;

        // Non-empty → the encoded file contents, decoded from memory instead of
        // read from `path`. Everything after the decode is identical.
        std::vector<std::byte> encoded;

        // A pixel update into an existing image (image.update). No decode: the
        // bytes are already the caller's rectangle in the image's own format,
        // so this path skips stbi entirely and copies one subresource.
        bool update = false;
        std::vector<std::byte> pixels;
        std::uint32_t layer = 0;
        // Which level, as opposed to `mips` above, which is how many a load
        // generates. An update writes exactly one.
        std::uint32_t mip = 0;
        VkOffset2D offset{0, 0};
        VkExtent2D extent{0, 0};
        VkImageLayout from_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    };

    // Both counters below describe the current batch. A batch is every upload
    // requested since the last time the queue fully drained.
    std::uint64_t done_count_() // call with mutex_ held
    {
        const std::uint64_t completed = context_.completed_submit_serial();
        const auto gpu_done = static_cast<std::uint64_t>(
            std::ranges::count_if(submitted_serials_, [&](std::uint64_t s) { return s <= completed; }));
        return failed_count_ + gpu_done;
    }

    void reset_batch_() // call with mutex_ held
    {
        batch_started_ = 0;
        failed_count_ = 0;
        submitted_serials_.clear();
    }

    void run_(std::stop_token stop)
    {
        while (true)
        {
            Job job;
            {
                std::unique_lock lock(mutex_);
                cv_.wait(lock, [&] { return stop.stop_requested() || !jobs_.empty(); });
                if (jobs_.empty())
                {
                    return; // stop requested and nothing left
                }
                job = std::move(jobs_.front());
                jobs_.pop_front();
            }

            reclaim_retired_();
            process_(job);
            cv_.notify_all();

            if (stop.stop_requested())
            {
                // Finish the popped job (done above), leave the rest to ~UploadManager.
                continue;
            }
        }
    }

    void process_(Job& job)
    {
        if (job.update)
        {
            process_update_(job);
            return;
        }
        if (!job.layers.empty())
        {
            process_layered_(job);
            return;
        }
        // Decode. Forcing RGBA to match the RGBA8_SRGB image. From memory when
        // the job carries the bytes, from the path otherwise — the two differ by
        // this call and nothing else.
        int width = 0, height = 0, channels = 0;
        stbi_uc* pixels = job.encoded.empty() ? stbi_load(job.path.c_str(), &width, &height, &channels, STBI_rgb_alpha)
                                              : stbi_load_from_memory(
                                                    reinterpret_cast<const stbi_uc*>(job.encoded.data()),
                                                    static_cast<int>(job.encoded.size()),
                                                    &width,
                                                    &height,
                                                    &channels,
                                                    STBI_rgb_alpha);
        if (!pixels)
        {
            // A load failure poisons the image (waiters get the error); a reload
            // failure just warns and keeps the good contents already on the GPU.
            if (job.reload)
                warn_reload_(job, stbi_failure_reason());
            else
                fail_(job, stbi_failure_reason());
            return;
        }
        if (static_cast<std::uint32_t>(width) != job.image->width() ||
            static_cast<std::uint32_t>(height) != job.image->height())
        {
            stbi_image_free(pixels);
            if (job.reload)
            {
                // v1 reloads are same-size only: a new size needs a new VkImage and
                // every descriptor set holding it rewritten. Keep the old image.
                if (auto logger = context_.logger())
                    logger->log(
                        Severity::Warning,
                        Source::Upload,
                        std::format(
                            "Hot reload: {} changed size ({}x{} -> {}x{}). Bazalt keeps the existing "
                            "image (a resize needs a restart)",
                            job.path,
                            job.image->width(),
                            job.image->height(),
                            static_cast<std::uint32_t>(width),
                            static_cast<std::uint32_t>(height)));
            }
            else
            {
                // The file changed between stbi_info and the decode. Exotic, but
                // uploading mismatched bytes would be worse than failing.
                fail_(job, "file changed on disk while it was being loaded");
            }
            return;
        }

        auto staging = job.image->create_filled_staging(context_, pixels);
        stbi_image_free(pixels);
        if (!staging)
        {
            if (job.reload)
                warn_reload_(job, staging.error().message.c_str());
            else
                fail_(job, staging.error().message.c_str());
            return;
        }
        auto [stagingBuffer, stagingAllocation] = *staging;

        // One-shot command buffer from the worker's own pool.
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        if (const VkResult r = allocate_cmd_(cmd); r != VK_SUCCESS)
        {
            vmaDestroyBuffer(context_.allocator(), stagingBuffer, stagingAllocation);
            fail_(job, std::format("the GPU upload could not start ({})", vk_result_name(r)));
            return;
        }

        VkCommandBufferBeginInfo beginInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr};
        context_.vk().vkBeginCommandBuffer(cmd, &beginInfo);
        // Only the initial layout transition differs: a reload preserves the live
        // contents against in-flight reads, a first upload discards UNDEFINED.
        if (job.reload)
            job.image->record_reload_commands(cmd, stagingBuffer, job.mips);
        else
            job.image->record_upload_commands(cmd, stagingBuffer, job.mips);
        context_.vk().vkEndCommandBuffer(cmd);

        std::uint64_t serial = 0;
        if (!submit_(cmd, serial, job.image->upload_serial()))
        {
            vmaDestroyBuffer(context_.allocator(), stagingBuffer, stagingAllocation);
            context_.vk().vkFreeCommandBuffers(context_.device(), pool_, 1, &cmd);
            if (job.reload)
                warn_reload_(job, "the GPU upload was refused");
            else
                fail_(job, "the GPU upload was refused");
            return;
        }

        // The staging buffer retires through the shared deletion queue (VMA is
        // internally synchronized, so the main thread may free it). The command
        // buffer does NOT: freeing it back into pool_ must happen on THIS
        // thread — command pools are externally synchronized, and the main
        // thread draining the deletion queue while the worker allocates from
        // the same pool is a race the validation layers rightly flag.
        context_.defer_destroy([allocator = context_.allocator(), stagingBuffer, stagingAllocation]
                               { vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation); });
        retired_.emplace_back(serial, cmd);

        // Both paths point the image at the new serial, so frames wait for the
        // re-upload and img.ready/.wait() track it. Only a load feeds the batch
        // accounting; a reload deliberately does not (see Job::reload).
        job.image->set_upload_submitted(serial);
        if (!job.reload)
        {
            std::lock_guard lock(mutex_);
            submitted_serials_.push_back(serial);
        }
    }

    // A pixel update: no decode at all, because the caller handed over bytes in
    // the image's own format. Otherwise the same staging -> copy -> submit tail
    // as every other job, so the frame that samples the image waits on the same
    // Context timeline.
    void process_update_(Job& job)
    {
        auto staging = create_staging_buffer(context_, job.pixels.size(), Staging::Upload, job.pixels.data());
        if (!staging)
        {
            fail_update_(job, staging.error().message.c_str());
            return;
        }
        auto [stagingBuffer, stagingAllocation] = *staging;

        VkCommandBuffer cmd = VK_NULL_HANDLE;
        if (const VkResult r = allocate_cmd_(cmd); r != VK_SUCCESS)
        {
            vmaDestroyBuffer(context_.allocator(), stagingBuffer, stagingAllocation);
            fail_update_(job, std::format("the GPU upload could not start ({})", vk_result_name(r)));
            return;
        }

        VkCommandBufferBeginInfo beginInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr};
        context_.vk().vkBeginCommandBuffer(cmd, &beginInfo);
        job.image->record_update_commands(
            cmd, stagingBuffer, job.layer, job.mip, job.offset, job.extent, job.from_layout);
        context_.vk().vkEndCommandBuffer(cmd);

        std::uint64_t serial = 0;
        if (!submit_(cmd, serial, job.image->upload_serial()))
        {
            vmaDestroyBuffer(context_.allocator(), stagingBuffer, stagingAllocation);
            context_.vk().vkFreeCommandBuffers(context_.device(), pool_, 1, &cmd);
            fail_update_(job, "failed to submit the update command buffer");
            return;
        }

        context_.defer_destroy([allocator = context_.allocator(), stagingBuffer, stagingAllocation]
                               { vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation); });
        retired_.emplace_back(serial, cmd);

        job.image->set_upload_submitted(serial);
        {
            std::lock_guard lock(mutex_);
            submitted_serials_.push_back(serial);
        }
    }

    // An update cannot poison the image the way a failed load does: the pixels
    // that are already there stay valid and keep rendering, exactly like a hot
    // reload that could not complete. It still has to leave the batch counters
    // balanced, or upload_progress would never reach 1.0 again.
    void fail_update_(Job& job, std::string_view reason)
    {
        if (auto logger = context_.logger())
        {
            logger->log(
                Severity::Error,
                Source::Upload,
                std::format(
                    "image.update failed ({}). The previous contents are unchanged",
                    reason.empty() ? std::string_view("?") : reason));
        }
        std::lock_guard lock(mutex_);
        ++failed_count_;
    }

    // One-shot command buffer from the worker's own pool.
    // Returns the VkResult rather than a bool, because the caller puts it in the
    // message. "failed to allocate an upload command buffer" told a Python user
    // nothing they could act on and threw away the one fact worth reporting.
    VkResult allocate_cmd_(VkCommandBuffer& cmd)
    {
        VkCommandBufferAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr,
            .commandPool = pool_,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1};
        return context_.vk().vkAllocateCommandBuffers(context_.device(), &allocInfo, &cmd);
    }

    // The worker's half of a one-shot submit, and the only place this thread
    // submits from. Context::submit_one_shot does the Vulkan part; the worker
    // keeps what is its own — freeing the command buffer back into pool_ from this
    // thread (see retired_), and the ORDER.
    //
    // The order is the point. "One worker, one queue, so the call order IS the GPU
    // order" is a promise this file makes (see the class comment and DESIGN.md),
    // and submitting in sequence does not keep it: two submits on one queue may
    // overlap or reorder unless one waits for the other. Two image.update() calls
    // therefore raced, and the second could land first — a video decoder showing
    // frames backwards, and a wrong answer rather than a slow one.
    //
    // Chaining every upload behind the previous one costs nothing real: they are
    // already issued by a single thread, and the work they order is a memcpy-shaped
    // copy. It was three copies of the submit block before this, which is how the
    // two inline ones came to have waitSemaphoreCount = 0 and stay that way.
    //
    // `after` is the image's own last upload, which is NOT always in this
    // worker's chain: create_image(array) has nothing to decode, so it submits
    // inline on the calling thread. Without it, the very first update of such an
    // image races the copy that created it, and losing that race leaves the
    // image holding what it was created with — the first test in
    // test_streaming.py, passing on every driver that happens to serialize.
    bool submit_(VkCommandBuffer cmd, std::uint64_t& serial, std::uint64_t after = 0)
    {
        auto submitted = context_.submit_one_shot(cmd, (std::ranges::max)(last_upload_serial_, after));
        if (!submitted)
        {
            return false;
        }
        serial = *submitted;
        last_upload_serial_ = serial;
        return true;
    }

    // A layered load: decode every face into one contiguous N-layer block, then
    // the exact same staging → copy → mipgen → submit path as a single upload
    // (Image handles the per-layer copy). Layered loads are never hot reloads,
    // so this mirrors process_'s non-reload tail without the reload branches.
    void process_layered_(Job& job)
    {
        const std::uint32_t w = job.image->width();
        const std::uint32_t h = job.image->height();
        const std::size_t layer_bytes = static_cast<std::size_t>(w) * h * 4; // RGBA8

        std::vector<stbi_uc> pixels(layer_bytes * job.layers.size());
        for (std::size_t i = 0; i < job.layers.size(); ++i)
        {
            int lw = 0, lh = 0, lc = 0;
            stbi_uc* p = stbi_load(job.layers[i].c_str(), &lw, &lh, &lc, STBI_rgb_alpha);
            if (!p)
            {
                fail_(job, stbi_failure_reason());
                return;
            }
            if (static_cast<std::uint32_t>(lw) != w || static_cast<std::uint32_t>(lh) != h)
            {
                // A face changed between the load_layered header check and now.
                stbi_image_free(p);
                fail_(job, "a layer changed size on disk while it was being loaded");
                return;
            }
            std::memcpy(pixels.data() + i * layer_bytes, p, layer_bytes);
            stbi_image_free(p);
        }

        auto staging = job.image->create_filled_staging(context_, pixels.data());
        if (!staging)
        {
            fail_(job, staging.error().message.c_str());
            return;
        }
        auto [stagingBuffer, stagingAllocation] = *staging;

        VkCommandBuffer cmd = VK_NULL_HANDLE;
        if (const VkResult r = allocate_cmd_(cmd); r != VK_SUCCESS)
        {
            vmaDestroyBuffer(context_.allocator(), stagingBuffer, stagingAllocation);
            fail_(job, std::format("the GPU upload could not start ({})", vk_result_name(r)));
            return;
        }

        VkCommandBufferBeginInfo beginInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr};
        context_.vk().vkBeginCommandBuffer(cmd, &beginInfo);
        job.image->record_upload_commands(cmd, stagingBuffer, job.mips);
        context_.vk().vkEndCommandBuffer(cmd);

        std::uint64_t serial = 0;
        if (!submit_(cmd, serial, job.image->upload_serial()))
        {
            vmaDestroyBuffer(context_.allocator(), stagingBuffer, stagingAllocation);
            context_.vk().vkFreeCommandBuffers(context_.device(), pool_, 1, &cmd);
            fail_(job, "the GPU upload was refused");
            return;
        }

        context_.defer_destroy([allocator = context_.allocator(), stagingBuffer, stagingAllocation]
                               { vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation); });
        retired_.emplace_back(serial, cmd);

        job.image->set_upload_submitted(serial);
        {
            std::lock_guard lock(mutex_);
            submitted_serials_.push_back(serial);
        }
    }

    // Worker thread only: free one-shot command buffers whose upload the GPU
    // has provably finished.
    void reclaim_retired_()
    {
        if (retired_.empty())
        {
            return;
        }
        const std::uint64_t completed = context_.completed_submit_serial();
        std::erase_if(
            retired_,
            [&](const auto& entry)
            {
                if (entry.first <= completed)
                {
                    context_.vk().vkFreeCommandBuffers(context_.device(), pool_, 1, &entry.second);
                    return true;
                }
                return false;
            });
    }

    void fail_(Job& job, std::string_view reason)
    {
        const std::string message = !reason.empty() ? std::format("Failed to load image: {} ({})", job.path, reason)
                                                    : std::format("Failed to load image: {}", job.path);
        if (auto logger = context_.logger())
        {
            logger->log(Severity::Error, Source::Upload, message);
        }
        job.image->set_upload_failed(message);
        {
            std::lock_guard lock(mutex_);
            ++failed_count_;
        }
    }

    // A reload that couldn't complete: WARNING, not the Error fail_ raises, and
    // the image is left exactly as it was — its previous contents keep
    // rendering. Never touches upload state or the batch counters. Symmetry with
    // a shader hot reload: a bad edit can't take the application down.
    void warn_reload_(Job& job, std::string_view reason)
    {
        if (auto logger = context_.logger())
        {
            logger->log(
                Severity::Warning,
                Source::Upload,
                !reason.empty()
                    ? std::format("Hot reload: {} ({}). Bazalt keeps the previous contents", job.path, reason)
                    : std::format("Hot reload: {}. Bazalt keeps the previous contents", job.path));
        }
    }

    Context& context_;
    VkCommandPool pool_ = VK_NULL_HANDLE;

    // Worker-thread-only: submitted one-shot cmds awaiting GPU completion.
    std::vector<std::pair<std::uint64_t, VkCommandBuffer>> retired_;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<Job> jobs_;
    std::uint64_t batch_started_ = 0;
    std::uint64_t failed_count_ = 0;
    std::vector<std::uint64_t> submitted_serials_;

    // The serial of the last upload this worker submitted, so the next one can be
    // ordered behind it. Touched only by the worker thread, which is the same
    // thread that owns retired_ and pool_.
    std::uint64_t last_upload_serial_ = 0;

    std::jthread worker_; // last member: joins before the rest tears down
};