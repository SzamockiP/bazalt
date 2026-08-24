#include "UploadManager.hpp"

#include <algorithm>
#include <cstring>
#include <format>
#include <utility>

UploadManager::UploadManager(Context& context)
    : context_(context)
{
    // Command pools are externally synchronized; the worker gets its own, one
    // per queue it submits to (the compute slot stays empty).
    for (QueueKind kind : {QueueKind::Transfer, QueueKind::Graphics})
    {
        VkCommandPoolCreateInfo poolInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
            .queueFamilyIndex = context.queue_family(kind)};
        context.vk().vkCreateCommandPool(context.device(), &poolInfo, nullptr, &pools_[queue_index(kind)]);
    }

    worker_ = std::jthread([this](std::stop_token stop) { run_(std::move(stop)); });
}

UploadManager::~UploadManager()
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
        // Every queue mutex: an idle drains every queue, so every queue's
        // submitter must be held off. lock_queues() skips a lock that aliases
        // the graphics one (locking one mutex twice is undefined).
        auto locks = context_.lock_queues();
        context_.vk().vkDeviceWaitIdle(context_.device());
    }
    context_.flush_deletion_queue();

    for (VkCommandPool pool : pools_)
    {
        if (pool != VK_NULL_HANDLE)
        {
            context_.vk().vkDestroyCommandPool(context_.device(), pool, nullptr);
        }
    }
}

std::expected<std::shared_ptr<Image>, Error> UploadManager::load(const std::string& path, bool mipmaps)
{
    int width = 0;
    int height = 0;
    int comp = 0;
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
        jobs_.push_back({.image = *image, .path = path, .mips = mips});
    }
    cv_.notify_all();
    return image;
}

std::expected<std::shared_ptr<Image>, Error> UploadManager::load_memory(std::vector<std::byte> blob, bool mipmaps)
{
    int width = 0;
    int height = 0;
    int comp = 0;
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

std::expected<std::shared_ptr<Image>, Error> UploadManager::load_layered(
    const std::vector<std::string>& paths,
    bool cube,
    bool mipmaps)
{
    if (paths.empty())
    {
        return std::unexpected(err_resource("load_image: the path list is empty"));
    }
    if (cube && paths.size() != 6)
    {
        return std::unexpected(
            err_resource(std::format("load_image(cube=True): a cubemap needs exactly 6 faces, got {}", paths.size())));
    }

    int width = 0;
    int height = 0;
    int comp = 0;
    for (std::size_t i = 0; i < paths.size(); ++i)
    {
        int w = 0;
        int h = 0;
        int c = 0;
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

void UploadManager::reload(std::shared_ptr<Image> image, std::string path)
{
    const std::uint32_t mips = image->mip_levels();
    // Pending, exactly as load() and update() mark it (0.30). A mipped upload
    // is two submits now, and between them the image sits in TRANSFER_DST —
    // a reader that only waited the OLD serials would land its own submit in
    // that window. Pending makes every reader block CPU-side until both
    // halves are submitted and the new serials are published.
    image->set_upload_pending();
    {
        std::lock_guard lock(mutex_);
        jobs_.push_back({.image = std::move(image), .path = std::move(path), .mips = mips, .reload = true});
    }
    cv_.notify_all();
}

void UploadManager::update(
    std::shared_ptr<Image> image,
    std::vector<std::byte> pixels,
    std::uint32_t layer,
    std::uint32_t mip,
    VkOffset3D offset,
    VkExtent3D extent)
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

void UploadManager::note_direct_upload(const QueueSerials& serials)
{
    {
        std::lock_guard lock(mutex_);
        ++batch_started_;
        submitted_serials_.push_back(serials);
    }
    cv_.notify_all();
}

double UploadManager::upload_progress()
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

void UploadManager::wait_all()
{
    QueueSerials wait_serial{};
    {
        std::unique_lock lock(mutex_);
        // First the CPU side: every enqueued job decoded and submitted (or
        // failed) …
        cv_.wait(lock, [&] { return failed_count_ + submitted_serials_.size() == batch_started_; });
        for (const QueueSerials& s : submitted_serials_)
        {
            max_merge(wait_serial, s);
        }
    }
    // … then the GPU side: every timeline reaching the last upload it carries.
    // A failed wait surfaces on the image itself (img.wait() reports it);
    // the aggregate verb has no single resource to blame.
    static_cast<void>(context_.wait_for_serials(wait_serial));
}

std::uint64_t UploadManager::done_count_()
{
    QueueSerials completed{};
    for (std::size_t i = 0; i < kQueueCount; ++i)
    {
        completed[i] = context_.completed_submit_serial(static_cast<QueueKind>(i));
    }
    const auto gpu_done = static_cast<std::uint64_t>(std::ranges::count_if(
        submitted_serials_,
        [&](const QueueSerials& s)
        {
            for (std::size_t i = 0; i < kQueueCount; ++i)
            {
                if (s[i] > completed[i])
                {
                    return false;
                }
            }
            return true;
        }));
    return failed_count_ + gpu_done;
}

void UploadManager::reset_batch_()
{
    batch_started_ = 0;
    failed_count_ = 0;
    submitted_serials_.clear();
}

VkResult UploadManager::allocate_cmd_(QueueKind kind, VkCommandBuffer& cmd)
{
    VkCommandBufferAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = pools_[queue_index(kind)],
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1};
    return context_.vk().vkAllocateCommandBuffers(context_.device(), &allocInfo, &cmd);
}

bool UploadManager::submit_(QueueKind kind, VkCommandBuffer cmd, std::uint64_t& serial, QueueSerials after)
{
    const std::size_t own = queue_index(kind);
    after[own] = (std::ranges::max)(after[own], last_upload_serial_[own]);
    auto submitted = context_.submit_one_shot(cmd, kind, after);
    if (!submitted)
    {
        return false;
    }
    serial = *submitted;
    last_upload_serial_[own] = serial;
    return true;
}

// Defined ahead of its three callers below: it is a template (the three
// callbacks are auto&&), so the definition has to be in this TU.
void UploadManager::submit_recorded_(
    Job& job,
    VkBuffer stagingBuffer,
    VmaAllocation stagingAllocation,
    auto&& record,
    auto&& on_alloc_fail,
    auto&& on_submit_fail)
{
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (const VkResult r = allocate_cmd_(QueueKind::Transfer, cmd); r != VK_SUCCESS)
    {
        vmaDestroyBuffer(context_.allocator(), stagingBuffer, stagingAllocation);
        on_alloc_fail(std::format("the GPU upload could not start ({})", vk_result_name(r)));
        return;
    }

    VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr};
    context_.vk().vkBeginCommandBuffer(cmd, &beginInfo);
    record(cmd);
    context_.vk().vkEndCommandBuffer(cmd);

    // The image's own chain, and — for a job that overwrites live contents —
    // the frames that may still be sampling them (see the declaration).
    QueueSerials after = job.image->upload_serial();
    if (job.reload || job.update)
    {
        const std::size_t g = queue_index(QueueKind::Graphics);
        after[g] = (std::ranges::max)(after[g], context_.submitted_serials()[g]);
    }
    std::uint64_t copied = 0;
    if (!submit_(QueueKind::Transfer, cmd, copied, after))
    {
        vmaDestroyBuffer(context_.allocator(), stagingBuffer, stagingAllocation);
        context_.vk().vkFreeCommandBuffers(context_.device(), pools_[queue_index(QueueKind::Transfer)], 1, &cmd);
        on_submit_fail();
        return;
    }

    context_.defer_destroy([allocator = context_.allocator(), stagingBuffer, stagingAllocation]
                           { vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation); });
    retired_.push_back({.queue = QueueKind::Transfer, .serial = copied, .cmd = cmd});

    QueueSerials serials{};
    serials[queue_index(QueueKind::Transfer)] = copied;

    // The blit half. An update job has no chain to generate (it writes one
    // subresource); a load or a reload with mips > 1 left every level in
    // TRANSFER_DST, and this cascade is what finishes it. A failure here
    // leaves the image there, so it is reported through the same path as a
    // refused copy: Failed for a load, a warning for a reload.
    if (!job.update && job.mips > 1)
    {
        VkCommandBuffer blit = VK_NULL_HANDLE;
        if (const VkResult r = allocate_cmd_(QueueKind::Graphics, blit); r != VK_SUCCESS)
        {
            on_alloc_fail(std::format("the mip chain could not be generated ({})", vk_result_name(r)));
            return;
        }
        context_.vk().vkBeginCommandBuffer(blit, &beginInfo);
        job.image->record_upload_mips(blit, job.mips);
        context_.vk().vkEndCommandBuffer(blit);

        std::uint64_t blitted = 0;
        if (!submit_(QueueKind::Graphics, blit, blitted, serials))
        {
            context_.vk().vkFreeCommandBuffers(context_.device(), pools_[queue_index(QueueKind::Graphics)], 1, &blit);
            on_submit_fail();
            return;
        }
        retired_.push_back({.queue = QueueKind::Graphics, .serial = blitted, .cmd = blit});
        serials[queue_index(QueueKind::Graphics)] = blitted;
    }

    // Both paths point the image at the new serials, so frames wait for the
    // re-upload and img.ready/.wait() track it. Only a load feeds the batch
    // accounting; a reload deliberately does not (see Job::reload).
    job.image->set_upload_submitted(serials);
    if (!job.reload)
    {
        std::lock_guard lock(mutex_);
        submitted_serials_.push_back(serials);
    }
}

void UploadManager::run_(std::stop_token stop)
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

void UploadManager::process_(Job& job)
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
    int width = 0;
    int height = 0;
    int channels = 0;
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
        {
            warn_reload_(job, stbi_failure_reason());
        }
        else
        {
            fail_(job, stbi_failure_reason());
        }
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
            // Balances the Pending mark exactly as warn_reload_ does — this
            // branch logs its own message, so it cannot go through it.
            job.image->abandon_upload();
            if (auto logger = context_.logger())
            {
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
        {
            warn_reload_(job, staging.error().message);
        }
        else
        {
            fail_(job, staging.error().message);
        }
        return;
    }
    auto [stagingBuffer, stagingAllocation] = *staging;

    submit_recorded_(
        job,
        stagingBuffer,
        stagingAllocation,
        // Only the initial layout transition differs: a reload preserves the
        // live contents against in-flight reads, a first upload discards
        // UNDEFINED.
        [&](VkCommandBuffer cmd)
        {
            const VkPipelineStageFlags legal = context_.queue_stages(QueueKind::Transfer);
            if (job.reload)
            {
                job.image->record_reload_commands(cmd, stagingBuffer, job.mips, legal);
            }
            else
            {
                job.image->record_upload_commands(cmd, stagingBuffer, job.mips, legal);
            }
        },
        [&](std::string_view reason) { fail_(job, reason); },
        [&]
        {
            if (job.reload)
            {
                warn_reload_(job, "the GPU upload was refused");
            }
            else
            {
                fail_(job, "the GPU upload was refused");
            }
        });
}

void UploadManager::process_update_(Job& job)
{
    auto staging = create_staging_buffer(context_, job.pixels.size(), Staging::Upload, job.pixels.data());
    if (!staging)
    {
        fail_update_(job, staging.error().message);
        return;
    }
    auto [stagingBuffer, stagingAllocation] = *staging;

    submit_recorded_(
        job,
        stagingBuffer,
        stagingAllocation,
        [&](VkCommandBuffer cmd)
        {
            job.image->record_update_commands(
                cmd,
                stagingBuffer,
                job.layer,
                job.mip,
                job.offset,
                job.extent,
                job.from_layout,
                context_.queue_stages(QueueKind::Transfer));
        },
        [&](std::string_view reason) { fail_update_(job, reason); },
        [&] { fail_update_(job, "failed to submit the update command buffer"); });
}

void UploadManager::fail_update_(Job& job, std::string_view reason)
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

void UploadManager::process_layered_(Job& job)
{
    const std::uint32_t w = job.image->width();
    const std::uint32_t h = job.image->height();
    const std::size_t layer_bytes = static_cast<std::size_t>(w) * h * 4; // RGBA8

    std::vector<stbi_uc> pixels(layer_bytes * job.layers.size());
    for (std::size_t i = 0; i < job.layers.size(); ++i)
    {
        int lw = 0;
        int lh = 0;
        int lc = 0;
        stbi_uc* p = stbi_load(job.layers[i].c_str(), &lw, &lh, &lc, STBI_rgb_alpha);
        if (!p)
        {
            fail_(job, stbi_failure_reason());
            return;
        }
        if (std::cmp_not_equal(lw, w) || std::cmp_not_equal(lh, h))
        {
            // A face changed between the load_layered header check and now.
            stbi_image_free(p);
            fail_(job, "a layer changed size on disk while it was being loaded");
            return;
        }
        std::memcpy(pixels.data() + (i * layer_bytes), p, layer_bytes);
        stbi_image_free(p);
    }

    auto staging = job.image->create_filled_staging(context_, pixels.data());
    if (!staging)
    {
        fail_(job, staging.error().message);
        return;
    }
    auto [stagingBuffer, stagingAllocation] = *staging;

    submit_recorded_(
        job,
        stagingBuffer,
        stagingAllocation,
        [&](VkCommandBuffer cmd)
        {
            job.image->record_upload_commands(cmd, stagingBuffer, job.mips, context_.queue_stages(QueueKind::Transfer));
        },
        [&](std::string_view reason) { fail_(job, reason); },
        [&] { fail_(job, "the GPU upload was refused"); });
}

void UploadManager::reclaim_retired_()
{
    if (retired_.empty())
    {
        return;
    }
    std::erase_if(
        retired_,
        [&](const Retired& entry)
        {
            if (entry.serial <= context_.completed_submit_serial(entry.queue))
            {
                context_.vk().vkFreeCommandBuffers(context_.device(), pools_[queue_index(entry.queue)], 1, &entry.cmd);
                return true;
            }
            return false;
        });
}

void UploadManager::fail_(Job& job, std::string_view reason)
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

void UploadManager::warn_reload_(Job& job, std::string_view reason)
{
    // The job was marked Pending when it was queued; nothing was submitted, so
    // the previous contents stand and the waiters wake to them. (A blit half
    // refused after its copy succeeded also lands here and is not undone —
    // that is a lost device, and the next reload restarts from the file.)
    job.image->abandon_upload();
    if (auto logger = context_.logger())
    {
        logger->log(
            Severity::Warning,
            Source::Upload,
            !reason.empty() ? std::format("Hot reload: {} ({}). Bazalt keeps the previous contents", job.path, reason)
                            : std::format("Hot reload: {}. Bazalt keeps the previous contents", job.path));
    }
}
