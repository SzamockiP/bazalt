#pragma once
#include <volk.h>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>
#include <array>
#include <expected>
#include <functional>
#include <string>
#include "Context.hpp"
#include "RenderTarget.hpp"
#include "Pipeline.hpp"
#include "Buffer.hpp"
#include "DescriptorSet.hpp"
#include "ResourceTracker.hpp"

// One resource touch, as the graph compiler consumes it (0.28). In pass mode
// the recorder appends these instead of computing barriers inline: the graph
// folds every pass's events through one ResourceTracker at compile time, which
// is what turns the per-recording floors into precise cross-pass edges. A Note
// is a manual barrier (or a transfer verb's aftermath) seeding the fold, the
// same contract note_buffer_access/note_image_layout carry inline.
struct UseEvent
{
    enum class Kind
    {
        BufferUse,
        ImageUse,
        BufferNote,
        ImageNote
    };
    Kind kind;
    std::shared_ptr<Buffer> buffer;
    std::shared_ptr<Image> image;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkPipelineStageFlags stages = 0;
    VkAccessFlags access = 0;
    bool writes = false;
    bool shader_writable = false;
    // "Only if something already wrote this image" — the sampled-image rule.
    // An uploaded texture the tracker never saw rests in SHADER_READ_ONLY
    // already, and transitioning it from a tracker's UNDEFINED would DISCARD
    // it. Inline mode answers this at record time against its own tracker; a
    // pass cannot, because the writer is another pass, so the flag travels and
    // the graph's fold decides.
    bool only_if_tracked = false;
    // Index into commands_ this use precedes — where a computed barrier must
    // land in a pass without a target. A render pass puts every barrier in its
    // entry batch instead (vkCmdPipelineBarrier is illegal inside dynamic
    // rendering), so there the position only orders the fold.
    std::size_t position = 0;
};

// The four parts of a rendering scope, called by the graph executor. They are
// separate because the compile decides the transitions with the whole frame in
// view: two passes that render into one target and preserve it keep the
// attachment in its layout, so the executor skips the transition halves at
// that seam and emits one in-place barrier instead.
void record_render_pass_transitions_in(const VolkDeviceTable& vk, VkCommandBuffer cmd, RenderTarget& rt, bool preserve);
void record_render_pass_begin(
    const VolkDeviceTable& vk,
    VkCommandBuffer cmd,
    RenderTarget& rt,
    const std::optional<std::vector<std::array<float, 4>>>& clear_colors,
    float clear_depth,
    std::uint32_t clear_stencil);
void record_render_pass_end(const VolkDeviceTable& vk, VkCommandBuffer cmd);
void record_render_pass_transitions_out(const VolkDeviceTable& vk, VkCommandBuffer cmd, RenderTarget& rt);

// Records the commands of ONE pass as closures, and reports what they touch.
//
// This is the engine behind Pass, and since 0.28 it is nothing else. The Graph
// owns the VkCommandBuffers, decides every barrier by folding the passes'
// events, and replays these closures itself — so a recorder holds no Vulkan
// object of its own except its query pools, and it computes no barrier. It
// could not: whatever wrote what this pass reads was recorded by a DIFFERENT
// recorder, and only the graph sees both.
//
// The recorded lambdas take a FrameContext rather than a SwapchainRenderer&.
// That is what lets one recording be replayed against a window, an offscreen
// image or a compute-only submit: this file does not know swapchains exist.
// The debug-utils label pair, shared by cmd.begin_label/end_label and the
// per-pass label Graph::execute wraps every named pass in (0.30). Loaded by
// volkLoadInstanceOnly, null without VK_EXT_debug_utils — then both are no-ops.
inline void begin_debug_label(VkCommandBuffer cmd, const std::string& name)
{
    if (vkCmdBeginDebugUtilsLabelEXT == nullptr)
    {
        return;
    }
    VkDebugUtilsLabelEXT label{
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
        .pNext = nullptr,
        .pLabelName = name.c_str(),
        .color = {0.0f, 0.0f, 0.0f, 0.0f}};
    vkCmdBeginDebugUtilsLabelEXT(cmd, &label);
}

inline void end_debug_label(VkCommandBuffer cmd)
{
    if (vkCmdEndDebugUtilsLabelEXT != nullptr)
    {
        vkCmdEndDebugUtilsLabelEXT(cmd);
    }
}

class CommandBuffer
{
public:
    // Takes a Context, not a renderer: this is a device resource and has
    // nothing to do with presentation. It is what lets a headless Context with
    // no renderer at all record commands.
    static std::expected<std::shared_ptr<CommandBuffer>, Error> create(
        Context& context,
        std::optional<bool> auto_barriers = std::nullopt);

    // Where resource uses are reported. The Pass owns both the recorder and
    // the sink, and sets this immediately after create().
    // Which queue will replay this recording. The timer pool asks it, because
    // timestampValidBits is per queue family and a pass on the compute queue
    // must be measured against the family that runs it (0.29).
    void set_queue(QueueKind queue)
    {
        queue_ = queue;
    }

    void set_event_sink(std::vector<UseEvent>* sink)
    {
        event_sink_ = sink;
    }

    // Deferred: a query pool may still be in use by a submit in flight when
    // the Python object is dropped.
    ~CommandBuffer();

    CommandBuffer(const CommandBuffer&) = delete;
    CommandBuffer& operator=(const CommandBuffer&) = delete;

    // Which Context this command buffer records for; see Buffer::owner().
    const Context* owner() const
    {
        return context_.get();
    }

    // Clears the recording. The Graph calls it when a pass retires, which is
    // what makes a Timer handle from before a reset report Superseded.
    CommandBuffer& begin();

    // Explicit override for split-screen and similar. The no-argument version is
    // gone: a render pass covers the whole-target case itself.
    CommandBuffer& set_viewport(float x, float y, float width, float height);

    CommandBuffer& set_scissor(std::int32_t x, std::int32_t y, std::uint32_t width, std::uint32_t height);

    CommandBuffer& bind_pipeline(const std::shared_ptr<Pipeline>& pipeline);

    // binding= selects which of the pipeline's vertex bindings this buffer
    // feeds: 0 is vertex_format (per vertex), 1 is instance_format (per
    // instance). A kwarg on the existing verb rather than a second method —
    // binding one buffer and binding the other are the same operation.
    // Returns expected since 0.30, for the reason bind_index_buffer gives: a
    // usage is a set of bits now, and a buffer without VERTEX or STORAGE used
    // to reach the layers as VUID-vkCmdBindVertexBuffers-pBuffers-00627.
    std::expected<void, Error> bind_vertex_buffer(const std::shared_ptr<Buffer>& buffer, std::uint32_t binding = 0);

    // Returns expected since 0.29: a STORAGE buffer is a legitimate index
    // buffer now (a compute shader that rewrites an index list), so the verb
    // has a type to check, and a VERTEX or UNIFORM buffer here used to reach
    // the layers as a VUID naming neither the call nor the fix. Same argument
    // and same shape as the indirect verbs.
    std::expected<void, Error> bind_index_buffer(const std::shared_ptr<Buffer>& buffer);

    // instances= is a kwarg on both draw verbs rather than a third verb: the
    // instance count is one argument of a draw, and draw_indexed_instanced was a
    // second name for a call that already existed.
    CommandBuffer& draw(uint32_t vertexCount, uint32_t instances = 1);

    CommandBuffer& draw_indexed(
        uint32_t indexCount,
        uint32_t firstIndex = 0,
        int32_t vertexOffset = 0,
        uint32_t instances = 1);

    CommandBuffer& dispatch(uint32_t groupCountX, uint32_t groupCountY = 1, uint32_t groupCountZ = 1);

    // ── Indirect draw and dispatch (0.19) ─────────────────────────────────────
    //
    // The arguments come out of a buffer the GPU can write, so a compute pass
    // decides what gets drawn and the CPU never learns the answer. That is the
    // whole feature: culling, LOD selection and particle compaction stop needing a
    // readback between the pass that decides and the draw that obeys.
    //
    // Three verbs rather than a buffer= kwarg on draw/draw_indexed/dispatch,
    // because such a kwarg would invalidate vertex_count and instances in the same
    // signature — the shape 0.15 rejected for the cross-Context transfer. They
    // return expected because they have real preconditions, unlike draw().
    //
    // bazalt declares no struct type for the arguments. The layout is
    // VkDrawIndirectCommand and numpy already writes it:
    //
    //     np.array([[vertex_count, instance_count, first_vertex, first_instance]],
    //              dtype=np.uint32)
    //
    // A dtype that exists only to be converted fails the scope test's second
    // question, and a std430 struct in GLSL is byte-identical to the above.
    // count_buffer moves the last CPU-side number onto the GPU: with one, `count`
    // becomes the MAXIMUM and the 4 bytes at count_offset say how many of those
    // commands to issue. A compute pass then decides the number of draws, not only
    // their contents.
    //
    // A kwarg on the verb rather than a fourth verb: where the count comes from is
    // one argument of a draw that already exists, and a draw_indirect_count name
    // would need a copy of every future argument to draw_indirect — the reasoning
    // that made draw_indexed_instanced disappear in 0.17.
    std::expected<void, Error> draw_indirect(
        std::shared_ptr<Buffer> buffer,
        VkDeviceSize offset = 0,
        std::uint32_t count = 1,
        std::shared_ptr<Buffer> count_buffer = nullptr,
        VkDeviceSize count_offset = 0,
        std::uint32_t stride = 0);

    std::expected<void, Error> draw_indexed_indirect(
        std::shared_ptr<Buffer> buffer,
        VkDeviceSize offset = 0,
        std::uint32_t count = 1,
        std::shared_ptr<Buffer> count_buffer = nullptr,
        VkDeviceSize count_offset = 0,
        std::uint32_t stride = 0);

    // No count: vkCmdDispatchIndirect takes exactly one VkDispatchIndirectCommand
    // (12 bytes, x/y/z group counts), so there is no multi-dispatch to gate.
    // Deliberately NOT refused inside a rendering scope, because plain dispatch()
    // is not either — one rule for both.
    std::expected<void, Error> dispatch_indirect(std::shared_ptr<Buffer> buffer, VkDeviceSize offset = 0);

    // Manual-mode barrier (also legal, if redundant, in auto mode). Only in a
    // pass without a target, which the binding checks: vkCmdPipelineBarrier is
    // invalid inside a rendering scope, and a render pass IS one. Nothing is
    // moved out of the way by magic — that would be a second, implicit way of
    // doing the explicit thing.
    std::expected<void, Error> barrier(std::shared_ptr<Buffer> buffer, Access src, Access dst);

    // The image counterpart: transition an image between shader accesses by
    // hand, across every mip and layer. The one cross-submit case the automatic
    // tracker can't reach — a compute shader bakes a storage image (GENERAL) in
    // one submit and later frames sample it (SHADER_READ_ONLY) — becomes
    // `cmd.barrier(image, Access.SHADER_WRITE, Access.SHADER_READ)` once, after
    // the dispatch, so the asset is generated once instead of every frame. The
    // layout is inferred from the access (WRITE->GENERAL, READ->SHADER_READ_ONLY);
    // only those two shader accesses name an image layout. In auto mode this also
    // updates the tracker, so mixing it with automatic uses of the same image in
    // one recording is safe — no stale-oldLayout double transition.
    std::expected<void, Error> barrier(std::shared_ptr<Image> image, Access src, Access dst);

    // Fills mip levels 1..N-1 of a mipped image by blitting mip 0 down the chain
    // (every array layer / cube face at once), leaving every level sampleable in
    // SHADER_READ_ONLY. The pair to create_image(..., mip_levels=N): write mip 0
    // (upload, compute, or a render pass), then generate the rest here.
    //
    // `src` names mip 0's CURRENT layout via the same access vocabulary as
    // cmd.barrier: SHADER_READ (SHADER_READ_ONLY — an uploaded or already-baked
    // image, the default) or SHADER_WRITE (GENERAL — mip 0 fresh from a compute
    // imageStore). Its scope doubles as the barrier waiting on that producer.
    // Refused inside a rendering scope (blits and barriers are illegal there).
    std::expected<void, Error> generate_mipmaps(const std::shared_ptr<Image>& image, Access src = Access::SHADER_READ);

    // Copy one image into another of the same size and format. The history
    // buffer every temporal effect needs: keep last frame's result to blend
    // against this one (motion blur, TAA, a feedback trail), or ping-pong two
    // storage images across dispatches.
    //
    // Both images are left in SHADER_READ_ONLY, because reading the copy is the
    // only reason to make one. `src` names the source's CURRENT layout in
    // cmd.barrier's vocabulary, exactly as generate_mipmaps does: SHADER_READ for
    // an image that is sampled (the default) or SHADER_WRITE for one a compute
    // dispatch just wrote. The destination is discarded, since the copy
    // overwrites all of it.
    //
    // Refused inside a rendering scope, like every other transfer verb.
    std::expected<void, Error> copy_image(
        const std::shared_ptr<Image>& src,
        const std::shared_ptr<Image>& dst,
        Access src_access = Access::SHADER_READ);

    // A copy that RESIZES: the two images need not share an extent, and `filter`
    // says how the pixels are sampled on the way.
    //
    // copy_image demands identical size and format; generate_mipmaps scales but
    // only inside one image. Downsampling for bloom, upscaling a compute result
    // and making a thumbnail all sat in that gap, and each one was a full
    // graphics pass with a fullscreen shader to do what the transfer queue does
    // in one command.
    //
    // Same `src_access` vocabulary as copy_image and generate_mipmaps: the
    // tracker treats an image's layout at the start of a replay as UNDEFINED, so
    // the caller names where the source actually is.
    std::expected<void, Error> blit_image(
        const std::shared_ptr<Image>& src,
        const std::shared_ptr<Image>& dst,
        Access src_access = Access::SHADER_READ,
        VkFilter filter = VK_FILTER_LINEAR);

    // Copy bytes from one buffer into another, GPU-side.
    //
    // There was no way to move buffer contents without a round trip through the
    // host or a compute shader written to do nothing but assign. A compute
    // ping-pong and "keep last frame's values" are both this one command.
    std::expected<void, Error> copy_buffer(
        std::shared_ptr<Buffer> src,
        std::shared_ptr<Buffer> dst,
        VkDeviceSize src_offset = 0,
        VkDeviceSize dst_offset = 0,
        // 0 means "the rest of the source", which is the whole buffer by
        // default. VK_WHOLE_SIZE is not legal in a copy region, so the real
        // length is computed in the definition rather than passed through.
        VkDeviceSize size = 0);

    // Fill a buffer with a repeated 32-bit value, GPU-side. Zeroing is the
    // reason it exists: a counter an atomic increments, or an accumulation
    // buffer, has to start each frame at a known value, and the only way to say
    // that was a dispatch whose whole body was an assignment.
    //
    // 32-bit because vkCmdFillBuffer is: the offset and the size must both be
    // multiples of 4, and the value is one dword repeated.
    std::expected<void, Error> fill_buffer(
        std::shared_ptr<Buffer> buffer,
        std::uint32_t value = 0,
        VkDeviceSize offset = 0,
        VkDeviceSize size = 0);

    // Writes up to 65536 bytes into a buffer from the command stream itself
    // (vkCmdUpdateBuffer) — no staging buffer, no second submit, so a small
    // patch (a counter, a few uniforms) lands inside the frame that needs it.
    // Legal on every queue, including Queue.TRANSFER. Size and offset must be
    // multiples of 4; anything larger is copy_buffer's job.
    std::expected<void, Error> update_buffer(
        std::shared_ptr<Buffer> buffer,
        std::vector<std::byte> data,
        VkDeviceSize offset = 0);

    // Copies one (layer, mip) of an image out of a buffer, tightly packed —
    // the whole level, so the old contents are discarded rather than waited
    // for. The buffer bytes start at buffer_offset. Legal on every queue,
    // including Queue.TRANSFER; the subresource ends in SHADER_READ_ONLY.
    std::expected<void, Error> copy_buffer_to_image(
        std::shared_ptr<Buffer> buffer,
        const std::shared_ptr<Image>& image,
        std::uint32_t layer = 0,
        std::uint32_t mip = 0,
        VkDeviceSize buffer_offset = 0);

    // The mirror: one (layer, mip) into a buffer at buffer_offset, tightly
    // packed. src_access names the image's resting state exactly as
    // copy_image's does; the subresource ends in SHADER_READ_ONLY.
    std::expected<void, Error> copy_image_to_buffer(
        const std::shared_ptr<Image>& image,
        std::shared_ptr<Buffer> buffer,
        std::uint32_t layer = 0,
        std::uint32_t mip = 0,
        VkDeviceSize buffer_offset = 0,
        Access src_access = Access::SHADER_READ);

    // Fill an image with one colour, with no pipeline and no pass. Resetting an
    // accumulation or history buffer, or clearing a storage image a compute
    // shader only writes part of. A depth image is refused: clearing depth is
    // what a rendering pass does, and it needs the depth clear value.
    std::expected<void, Error> clear_image(const std::shared_ptr<Image>& image, std::array<float, 4> color);

    // ── GPU timers ──────────────────────────────────────────────────────────
    //
    // A GPU timer is a pair of query slots — exactly a Vulkan timestamp query.
    // The Python-facing handle (struct Timer, in bindings/Common.hpp) owns one pair:
    // cmd.timer() records the opening timestamp and hands back the handle, which
    // is stopped explicitly (t.stop()) or by a `with`, and read back off itself
    // (t.ms). The handle IS the identity — no name, no key — so multiple, nested
    // and overlapping timers all just work.
    //
    // Unlike renderer.gpu_time_ms this needs no window and no frame loop:
    // the headless submit blocks, so the readback is ready as soon as
    // ctx.submit() returns (profiling a dispatch is the use case).
    //
    // Self-gating: the query pool is created only when a timer is actually used,
    // so an app that never calls timer() pays nothing, no Context flag required.
    // Best-effort: a device without timestamp support reports None, never errors.

    // Records the opening timestamp and returns the timer's index (its two query
    // slots are 2*index / 2*index+1). Paired with stop_timer.
    std::size_t start_timer();

    void stop_timer(std::size_t index);

    // Which recording a timer belongs to. begin() bumps this, so a handle read
    // after the command buffer was re-recorded reports None (its slots now hold
    // a different timer's data) instead of a misleading number.
    std::uint64_t recording_generation() const
    {
        return recording_generation_;
    }

    struct TimerReading
    {
        QueryStatus status = QueryStatus::NotReady;
        double ms = 0.0;
    };

    struct OcclusionReading
    {
        QueryStatus status = QueryStatus::NotReady;
        std::uint64_t samples = 0;
    };

    // The measured time of one timer in milliseconds, with the reason attached
    // when there is none.
    TimerReading read_timer(std::size_t index, std::uint64_t generation) const;

    // ── Debug labels ────────────────────────────────────────────────────────
    //
    // A named scope in a capture. bazalt has named its OBJECTS since 0.8, which
    // answers "which image is that", but a RenderDoc capture was still a flat
    // list of draws with nothing saying where the shadow pass ended and the
    // composite began.
    //
    // Silent no-op without VK_EXT_debug_utils, exactly like set_debug_name, and
    // for the same reason: vk-bootstrap only requests the extension when a debug
    // callback is set. So a release run pays nothing and simply shows no labels.
    //
    // The entry points stay on volk's globals rather than moving to ctx.vk(),
    // which is the documented rule for debug utils: it is an INSTANCE extension,
    // so vkGetInstanceProcAddr is the sanctioned route and vkGetDeviceProcAddr
    // may legally return null. They are loader trampolines dispatching on the
    // VkCommandBuffer, so one pointer is right for every Context.
    CommandBuffer& begin_label(const std::string& name);

    CommandBuffer& end_label();

    // ── Occlusion queries ───────────────────────────────────────────────────
    //
    // How many fragments of the draws inside the scope passed the depth and
    // stencil tests. The handle IS the identity, exactly as for timers (0.9), so
    // several queries in one recording need no names and no keys.
    //
    // Vulkan requires an occlusion query to begin and end inside the SAME render
    // pass, which is why this refuses outside a rendering scope: the alternative
    // is a validation error at submit naming neither the call nor the reason.
    std::expected<std::size_t, Error> start_occlusion_query();

    void stop_occlusion_query(std::size_t index);

    // The sample count of one query, with the same reasons attached. There is no
    // Unsupported here: an imprecise occlusion query is core Vulkan behind no
    // feature, so the only answers are the number, a stale handle, and "not
    // yet".
    OcclusionReading read_occlusion_query(std::size_t index, std::uint64_t generation) const;

    // The short form, for the pipeline that is already bound (0.25). Uses the
    // LAST pipeline bound whatever its bind point, because push constants belong
    // to a pipeline layout rather than to a bind point, and a recording that
    // pushes for a pipeline it has not bound is already confused.
    std::expected<void, Error> push_constants(uint32_t offset, uint32_t size, const void* data);

    // No stage argument: the Pipeline already knows which stages its push constant
    // range covers, so passing a mismatched one was a validation error for no gain.
    CommandBuffer& push_constants(
        const std::shared_ptr<Pipeline>& pipeline,
        uint32_t offset,
        uint32_t size,
        const void* data);

    // The short form: bind the set where it was allocated to go, on the pipeline
    // that is already bound (0.25, ergonomics #3). Both arguments the long form
    // takes are known — the set records its index and bind point, and
    // bind_pipeline records the pipeline — so repeating them is a chance to
    // disagree with the truth, not information.
    //
    // The long form stays for a recording split across functions, where the
    // pipeline was bound somewhere this code cannot see, and for binding a set
    // against a DIFFERENT pipeline with a compatible layout.
    std::expected<void, Error> bind_descriptor_set(const std::shared_ptr<DescriptorSet>& descSet);

    CommandBuffer& bind_descriptor_set(
        const std::shared_ptr<DescriptorSet>& descSet,
        const std::shared_ptr<Pipeline>& pipeline,
        uint32_t setIndex);

    const std::vector<std::shared_ptr<DescriptorSet>>& used_sets() const
    {
        return used_sets_;
    }

    // The buffers this recording binds or copies, for the same reason
    // used_sets exists: a STATIC buffer's fill is a submit of its own since
    // 0.18.0, and the submit path waits on it. Recorded by record_buffer_use_,
    // which is deliberately NOT part of track_use_ — that one returns early
    // with auto_barriers=False, and residency is not a barrier question.
    const std::vector<std::shared_ptr<Buffer>>& used_buffers() const
    {
        return used_buffers_;
    }

    // ── The replay surface the graph executor drives ────────────────────────
    //
    // The executor replays a pass's commands itself, interleaving the barriers
    // the compile scheduled — so it takes the pieces rather than a whole
    // execute(): the query-pool resets (illegal inside a render pass, so they
    // run before any pass opens) and a range replay.

    void reset_query_pools(VkCommandBuffer vkCmd, const FrameContext& frame);

    void replay_range(VkCommandBuffer vkCmd, const FrameContext& frame, std::size_t from, std::size_t to) const
    {
        for (std::size_t i = from; i < to; ++i)
        {
            commands_[i](vkCmd, frame);
        }
    }

    std::size_t command_count() const
    {
        return commands_.size();
    }

private:
    CommandBuffer(std::shared_ptr<Context> context)
        : context_(context)
    {
    }

    // The tail copy_image and blit_image share. BOTH ends, not just the
    // destination: the transfer leaves the source sampleable too, and an Image
    // (or a tracker) that still believed it was in GENERAL would hand a stale
    // oldLayout to the next use — a validation error with no obvious author.
    void finish_image_transfer_(const std::shared_ptr<Image>& src, const std::shared_ptr<Image>& dst);

    // The one spelling of "this recording just put the resource into this
    // state, visible to (stages, access)". A manual barrier, generate_mipmaps
    // and the image transfers all report through it, and it reaches the sink
    // whatever auto_barriers_ says — a manual pass's barriers are exactly what
    // the fold needs in order to place its neighbours' barriers correctly.
    void note_buffer_state_(
        const std::shared_ptr<Buffer>& buffer,
        VkPipelineStageFlags dst_stages,
        VkAccessFlags dst_access);

    void note_image_state_(
        const std::shared_ptr<Image>& image,
        VkImageLayout layout,
        VkPipelineStageFlags dst_stages,
        VkAccessFlags dst_access);

    // A recorded timestamp write. Captures `this` (safe: the lambda only runs
    // while the graph replays this recorder) and reads timer_pool_ then, so it
    // no-ops when timestamps are unsupported and follows the pool across a grow.
    void record_timer_write_(std::uint32_t slot, VkPipelineStageFlagBits stage);

    // Best-effort query pool sized for `needed` slots. Queries timestamp
    // support once; on an unsupported device timer_pool_ stays null and every
    // timer becomes a silent no-op (timer_ms returns None). Grows by recreating
    // (deferred destroy of the old pool) — rare, only when a later recording
    // declares more scopes than any before it.
    void ensure_timer_pool_(std::size_t needed);

    // The occlusion counterpart. Simpler than the timer pool: occlusion queries
    // are core Vulkan with no feature bit and no per-queue-family validity to
    // check, so there is nothing to probe — only the allocation can fail, and a
    // failure leaves the pool null and every query reporting None.
    void ensure_occlusion_pool_(std::size_t needed);

    // Residency bookkeeping, kept apart from track_use_ on purpose: that one
    // returns early with auto_barriers=False, and waiting for a buffer's fill
    // to land is not a barrier the caller can take over. Buffers with no
    // pending upload are skipped, so a recording of DYNAMIC buffers stores
    // nothing.
    void record_buffer_use_(const std::shared_ptr<Buffer>& buffer);

    // Everything the three indirect verbs check, in one place so they cannot
    // disagree about which buffer is legal or what the message says.
    std::expected<void, Error> check_indirect_(
        const std::shared_ptr<Buffer>& buffer,
        VkDeviceSize offset,
        std::uint32_t count,
        VkDeviceSize stride,
        VkDeviceSize argument_size,
        const char* what);

    // The count buffer gets the same three checks the argument buffer gets — it is
    // read by the same command processor at the same stage, and 4 bytes is a
    // VkDeviceSize past the end just as readily as 16 are. The feature is the one
    // thing that differs: drawIndirectCount is optional, and without it the entry
    // point is a null pointer in the dispatch table rather than a diagnostic.
    std::expected<void, Error> check_count_buffer_(
        const std::shared_ptr<Buffer>& count_buffer,
        VkDeviceSize count_offset,
        const char* what);

    // The command processor reads the arguments at DRAW_INDIRECT, which is earlier
    // than any shader stage — so a pass that wrote them needs the barrier this
    // reports, and the graph puts it in this pass's entry batch, before the
    // rendering scope opens.
    void track_indirect_(const std::shared_ptr<Buffer>& buffer);

    void track_use_(
        const std::shared_ptr<Buffer>& buffer,
        VkPipelineStageFlags stages,
        VkAccessFlags access,
        bool writes);

    void track_image_use_(
        const std::shared_ptr<Image>& image,
        VkImageLayout layout,
        VkPipelineStageFlags stages,
        VkAccessFlags access,
        bool writes,
        bool only_if_tracked = false);

    // Does the pipeline bound at this bind point write (set, binding)?
    //
    // Asks each of its shaders' reflection at record time, so a hot-reload
    // replace() is picked up without invalidating anything. Fail-open on purpose:
    // with no pipeline bound there is nothing to ask, and the answer has to be the
    // conservative one — a draw with no pipeline is a bug the layers name precisely,
    // and guessing "not written" there would silently drop a real barrier.
    static bool pipeline_writes_(const std::shared_ptr<Pipeline>& pipeline, std::uint32_t set, std::uint32_t binding);

    // The shared body of track_draw_ and track_dispatch_. Before 0.19 these were
    // two functions that disagreed about the same question: the graphics one called
    // every storage buffer a READ (so a fragment SSBO write was invisible) and
    // handled storage images not at all, while the compute one called both
    // READ+WRITE unconditionally. Now they differ only in their stage mask, which is
    // the only thing that was ever really different about them.
    //
    // What reflection changes in each direction:
    //
    //  * Graphics gains the writes it never saw. A fragment imageStore was worse
    //    than untracked — DescriptorSet::set_storage_image already recorded the
    //    image as resting in GENERAL, so the layout the descriptor promised and the
    //    layout the image was in disagreed unless the user wrote a barrier by hand.
    //  * Compute LOSES barriers it did not need. `use(..., writes=true)` wipes read
    //    state, so two dispatches that only read the same input SSBO used to get a
    //    WAW barrier between them, and a reported write also switches on the graph's
    //    per-replay memory barrier. A `readonly buffer` shared down a chain of
    //    passes is the common case, not an exotic one.
    //
    // Safe in both directions because of the fail-open invariant in
    // SpirvReflect.hpp: a barrier only ever disappears on a positive proof that no
    // store, atomic or imageWrite touches that binding.
    void track_descriptor_uses_(
        const std::unordered_map<uint32_t, std::shared_ptr<DescriptorSet>>& sets,
        const std::shared_ptr<Pipeline>& pipeline,
        VkPipelineStageFlags stages);

    // The stage mask is the Context's, not a VERTEX|FRAGMENT literal: with
    // tessellation or geometry enabled a graphics pipeline has stages those two bits
    // do not name, and a barrier that omits a stage does not cover the read it was
    // recorded for. Legal by construction — the mask only ever holds bits whose
    // feature the device enabled.
    void track_draw_();

    void track_dispatch_();

    std::shared_ptr<Context> context_;
    std::vector<std::function<void(VkCommandBuffer, const FrameContext&)>> commands_;
    std::vector<std::shared_ptr<DescriptorSet>> used_sets_;
    std::vector<std::shared_ptr<Buffer>> used_buffers_;

    // Where resource uses are reported, for the graph to fold. Owned by the
    // Pass that owns this recorder, and set before anything is recorded.
    std::vector<UseEvent>* event_sink_ = nullptr;
    // The queue this recording's pass runs on. One value for its whole life —
    // a Pass names its queue at add_pass and never moves — which is why the
    // timer's memoized answer below stays valid.
    QueueKind queue_ = QueueKind::Graphics;

    // ── record-time state (reset by begin(), never touched at replay) ──
    bool auto_barriers_ = true;
    std::unordered_map<uint32_t, std::shared_ptr<DescriptorSet>> bound_graphics_sets_;
    std::unordered_map<uint32_t, std::shared_ptr<DescriptorSet>> bound_compute_sets_;
    // Record-time state, so the tracker can read the bound shaders' reflection.
    // Cleared in begin() with the sets, because both describe one recording.
    std::shared_ptr<Pipeline> bound_graphics_pipeline_;
    std::shared_ptr<Pipeline> bound_compute_pipeline_;
    // Whichever of the two was bound last. Push constants belong to a layout, not
    // to a bind point, so "the pipeline that is already known" is this one.
    std::shared_ptr<Pipeline> bound_last_pipeline_;

    // ── GPU timers (query pool survives begin(); results read after submit) ──
    VkQueryPool timer_pool_ = VK_NULL_HANDLE;
    std::uint32_t timer_capacity_ = 0;
    float timer_period_ = 0.0f;
    std::uint32_t timer_valid_bits_ = 0;
    std::optional<bool> timer_supported_;    // queried once, lazily
    std::size_t timer_count_ = 0;            // timers declared this recording
    std::uint64_t recording_generation_ = 0; // bumped by begin(); stale-handle guard

    // ── Occlusion queries (same lifetime rules as the timer pool above) ──
    VkQueryPool occlusion_pool_ = VK_NULL_HANDLE;
    std::uint32_t occlusion_capacity_ = 0;
    // Read once per recording rather than per query: the feature set is fixed for
    // the Context's whole life, so asking again per draw would be the same answer
    // through a hash lookup.
    bool precise_occlusion_ = context_->supports(Feature::PRECISE_OCCLUSION);
    std::size_t occlusion_count_ = 0;

    // Depth of the label nesting declared this recording, so end_label() can
    // refuse to close one that was never opened.
    std::size_t open_labels_ = 0;
};
