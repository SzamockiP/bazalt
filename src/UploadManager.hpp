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
#include <thread>
#include <vector>

#include "stb_image.h"
#include "Context.hpp"
#include "Error.hpp"
#include "Format.hpp"
#include "Image.hpp"

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
        vkCreateCommandPool(context.device(), &poolInfo, nullptr, &pool_);

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
            vkDeviceWaitIdle(context_.device());
        }
        context_.flush_deletion_queue();

        if (pool_ != VK_NULL_HANDLE)
        {
            vkDestroyCommandPool(context_.device(), pool_, nullptr);
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
        (*image)->set_upload_pending();

        {
            std::lock_guard lock(mutex_);
            ++batch_started_;
            jobs_.push_back({*image, path, mips});
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
                        "load_image: every layer must be the same size; {} is {}x{}, expected {}x{}",
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

    // ── UploadManagerBase (the Python-visible aggregate state) ────────────────

    bool uploads_done() override
    {
        std::lock_guard lock(mutex_);
        return done_count_() == batch_started_;
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
        if (wait_serial > 0)
        {
            VkSemaphore timeline = context_.submit_timeline();
            VkSemaphoreWaitInfo waitInfo{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                .pNext = nullptr,
                .flags = 0,
                .semaphoreCount = 1,
                .pSemaphores = &timeline,
                .pValues = &wait_serial};
            vkWaitSemaphores(context_.device(), &waitInfo, UINT64_MAX);
        }
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
        if (!job.layers.empty())
        {
            process_layered_(job);
            return;
        }
        // Decode. Forcing RGBA to match the RGBA8_SRGB image.
        int width = 0, height = 0, channels = 0;
        stbi_uc* pixels = stbi_load(job.path.c_str(), &width, &height, &channels, STBI_rgb_alpha);
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
                            "Hot reload: {} changed size ({}x{} -> {}x{}); keeping the existing "
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
        VkCommandBufferAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr,
            .commandPool = pool_,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1};
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(context_.device(), &allocInfo, &cmd) != VK_SUCCESS)
        {
            vmaDestroyBuffer(context_.allocator(), stagingBuffer, stagingAllocation);
            fail_(job, "failed to allocate an upload command buffer");
            return;
        }

        VkCommandBufferBeginInfo beginInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr};
        vkBeginCommandBuffer(cmd, &beginInfo);
        // Only the initial layout transition differs: a reload preserves the live
        // contents against in-flight reads, a first upload discards UNDEFINED.
        if (job.reload)
            job.image->record_reload_commands(cmd, stagingBuffer, job.mips);
        else
            job.image->record_upload_commands(cmd, stagingBuffer, job.mips);
        vkEndCommandBuffer(cmd);

        std::uint64_t serial = 0;
        {
            std::lock_guard lock(context_.queue_mutex());
            serial = context_.advance_submit_serial();

            VkSemaphore timeline = context_.submit_timeline();
            VkTimelineSemaphoreSubmitInfo timelineInfo{
                .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                .pNext = nullptr,
                .waitSemaphoreValueCount = 0,
                .pWaitSemaphoreValues = nullptr,
                .signalSemaphoreValueCount = 1,
                .pSignalSemaphoreValues = &serial};
            VkSubmitInfo submitInfo{
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .pNext = &timelineInfo,
                .waitSemaphoreCount = 0,
                .pWaitSemaphores = nullptr,
                .pWaitDstStageMask = nullptr,
                .commandBufferCount = 1,
                .pCommandBuffers = &cmd,
                .signalSemaphoreCount = 1,
                .pSignalSemaphores = &timeline};
            if (vkQueueSubmit(context_.graphics_queue(), 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS)
            {
                vmaDestroyBuffer(context_.allocator(), stagingBuffer, stagingAllocation);
                vkFreeCommandBuffers(context_.device(), pool_, 1, &cmd);
                if (job.reload)
                    warn_reload_(job, "failed to submit the upload command buffer");
                else
                    fail_(job, "failed to submit the upload command buffer");
                return;
            }
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

        VkCommandBufferAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr,
            .commandPool = pool_,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1};
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(context_.device(), &allocInfo, &cmd) != VK_SUCCESS)
        {
            vmaDestroyBuffer(context_.allocator(), stagingBuffer, stagingAllocation);
            fail_(job, "failed to allocate an upload command buffer");
            return;
        }

        VkCommandBufferBeginInfo beginInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr};
        vkBeginCommandBuffer(cmd, &beginInfo);
        job.image->record_upload_commands(cmd, stagingBuffer, job.mips);
        vkEndCommandBuffer(cmd);

        std::uint64_t serial = 0;
        {
            std::lock_guard lock(context_.queue_mutex());
            serial = context_.advance_submit_serial();

            VkSemaphore timeline = context_.submit_timeline();
            VkTimelineSemaphoreSubmitInfo timelineInfo{
                .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                .pNext = nullptr,
                .waitSemaphoreValueCount = 0,
                .pWaitSemaphoreValues = nullptr,
                .signalSemaphoreValueCount = 1,
                .pSignalSemaphoreValues = &serial};
            VkSubmitInfo submitInfo{
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .pNext = &timelineInfo,
                .waitSemaphoreCount = 0,
                .pWaitSemaphores = nullptr,
                .pWaitDstStageMask = nullptr,
                .commandBufferCount = 1,
                .pCommandBuffers = &cmd,
                .signalSemaphoreCount = 1,
                .pSignalSemaphores = &timeline};
            if (vkQueueSubmit(context_.graphics_queue(), 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS)
            {
                vmaDestroyBuffer(context_.allocator(), stagingBuffer, stagingAllocation);
                vkFreeCommandBuffers(context_.device(), pool_, 1, &cmd);
                fail_(job, "failed to submit the upload command buffer");
                return;
            }
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
                    vkFreeCommandBuffers(context_.device(), pool_, 1, &entry.second);
                    return true;
                }
                return false;
            });
    }

    void fail_(Job& job, const char* reason)
    {
        const std::string message = reason ? std::format("Failed to load image: {} ({})", job.path, reason)
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
    void warn_reload_(Job& job, const char* reason)
    {
        if (auto logger = context_.logger())
        {
            logger->log(
                Severity::Warning,
                Source::Upload,
                reason ? std::format("Hot reload: {} ({}); keeping the previous contents", job.path, reason)
                       : std::format("Hot reload: {}; keeping the previous contents", job.path));
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

    std::jthread worker_; // last member: joins before the rest tears down
};