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
class UploadManager final
{
public:
    explicit UploadManager(Context& context);

    // Abandons undecoded jobs (their images end Failed so any waiter wakes),
    // finishes at most the job in flight, joins, and tears down the pool.
    // ~Context runs this before vkDeviceWaitIdle, while the device is alive.
    ~UploadManager();

    // Main thread: validate the file header synchronously (a missing or
    // mangled file fails HERE, at the call site, and width/height are correct
    // immediately), create the empty image, and hand the decode to the worker.
    std::expected<std::shared_ptr<Image>, Error> load(const std::string& path, bool mipmaps = true);

    // Main thread: the same load, from encoded bytes instead of a path.
    //
    // A PNG downloaded over the network, unpacked from a zip or produced by PIL
    // has no path on disk, so the only way to hand it to bazalt was to write a
    // temporary file. The header is validated here for the same reason the file
    // version validates it here: a mangled blob fails at the call site, with
    // width and height correct immediately.
    //
    // Never hot-reloaded, by construction: bazalt has no path to watch.
    std::expected<std::shared_ptr<Image>, Error> load_memory(std::vector<std::byte> blob, bool mipmaps = true);

    // Main thread: a layered async load (texture array or cubemap) from N files.
    // Validates every header synchronously (all faces must share a size), builds
    // the empty layered image, and hands the decode + concatenate + single upload
    // to the worker. One batch unit, exactly like a single load.
    std::expected<std::shared_ptr<Image>, Error> load_layered(
        const std::vector<std::string>& paths,
        bool cube,
        bool mipmaps = true);

    // Hot reload: re-decode `path` into the EXISTING image. The header is NOT
    // re-validated here — this is called from the watcher drain, and a bad file
    // must not throw; the worker decodes, checks the size, and on any problem
    // logs a WARNING and keeps the old contents. Same size and format only in
    // v1 (a resize would need a new VkImage and every descriptor set rewritten).
    // Reuses the image's existing mip count.
    void reload(std::shared_ptr<Image> image, std::string path);

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
        VkOffset3D offset,
        VkExtent3D extent);

    // ── UploadManagerBase (the Python-visible aggregate state) ────────────────

    // An upload that never entered the decode queue: already-decoded bytes that
    // create_buffer / create_image(array) submitted on the calling thread. It is
    // started and submitted in the same breath, so both counters move together
    // and the CPU-side predicate in wait_all() stays balanced.
    void note_direct_upload(std::uint64_t serial);

    // Progress of the current batch, 0.0 .. 1.0 (1.0 when idle). The batch
    // resets once fully done, so a second loading screen starts from 0 again.
    double upload_progress();

    void wait_all();

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
        VkOffset3D offset{0, 0, 0};
        VkExtent3D extent{0, 0, 0};
        VkImageLayout from_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    };

    // Both counters below describe the current batch. A batch is every upload
    // requested since the last time the queue fully drained.
    std::uint64_t done_count_(); // call with mutex_ held

    void reset_batch_(); // call with mutex_ held

    void run_(std::stop_token stop);

    void process_(Job& job);

    // A pixel update: no decode at all, because the caller handed over bytes in
    // the image's own format. Otherwise the same staging -> copy -> submit tail
    // as every other job, so the frame that samples the image waits on the same
    // Context timeline.
    void process_update_(Job& job);

    // An update cannot poison the image the way a failed load does: the pixels
    // that are already there stay valid and keep rendering, exactly like a hot
    // reload that could not complete. It still has to leave the batch counters
    // balanced, or upload_progress would never reach 1.0 again.
    void fail_update_(Job& job, std::string_view reason);

    // One-shot command buffer from the worker's own pool.
    // Returns the VkResult rather than a bool, because the caller puts it in the
    // message. "failed to allocate an upload command buffer" told a Python user
    // nothing they could act on and threw away the one fact worth reporting.
    VkResult allocate_cmd_(VkCommandBuffer& cmd);

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
    bool submit_(VkCommandBuffer cmd, std::uint64_t& serial, std::uint64_t after = 0);

    // The tail every job shares once its staging buffer is filled: one-shot
    // command buffer, record, submit ordered behind the image's own chain,
    // retire the staging buffer, point the image at the new serial. The callers
    // differ only in what they record and how they report a failure — `record`
    // fills the command buffer, `on_alloc_fail(reason)` reports an allocation
    // failure, `on_submit_fail()` reports a refused submit (its message and its
    // reload handling are the caller's).
    //
    // The staging buffer retires through the shared deletion queue (VMA is
    // internally synchronized, so the main thread may free it). The command
    // buffer does NOT: freeing it back into pool_ must happen on THIS
    // thread — command pools are externally synchronized, and the main
    // thread draining the deletion queue while the worker allocates from
    // the same pool is a race the validation layers rightly flag.
    void submit_recorded_(
        Job& job,
        VkBuffer stagingBuffer,
        VmaAllocation stagingAllocation,
        auto&& record,
        auto&& on_alloc_fail,
        auto&& on_submit_fail);

    // A layered load: decode every face into one contiguous N-layer block, then
    // the exact same staging → copy → mipgen → submit path as a single upload
    // (Image handles the per-layer copy). Layered loads are never hot reloads,
    // so this mirrors process_'s non-reload tail without the reload branches.
    void process_layered_(Job& job);

    // Worker thread only: free one-shot command buffers whose upload the GPU
    // has provably finished.
    void reclaim_retired_();

    void fail_(Job& job, std::string_view reason);

    // A reload that couldn't complete: WARNING, not the Error fail_ raises, and
    // the image is left exactly as it was — its previous contents keep
    // rendering. Never touches upload state or the batch counters. Symmetry with
    // a shader hot reload: a bad edit can't take the application down.
    void warn_reload_(Job& job, std::string_view reason);

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
