#pragma once
#include <volk.h>
#include <array>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <span>
#include <utility>
#include <vector>
#include "CommandBuffer.hpp"

// QueueKind lives in Queue.hpp since 0.29: Context owns the runtimes and the
// tracker has to know which queue folded a use, and neither header may include
// the other. Reached from here through CommandBuffer.hpp.

class Graph;

// One node of a Graph: a recording with a boundary the compiler can trust.
// A pass with a target is a render pass (the whole pass is one dynamic
// rendering scope); a pass without one holds compute and transfer work. The
// pass is a HANDLE — `with` is optional sugar that seals it early, and an
// unsealed pass seals itself at the graph's first compile.
class Pass
{
public:
    // Which verbs a pass of this kind accepts. Render-only: the draw family,
    // viewport/scissor, occlusion queries, vertex/index binds. General-only:
    // dispatch and every transfer/barrier verb — all illegal inside a dynamic
    // rendering scope, and a render pass IS one. The refusal happens at record
    // time with a message naming the fix, where today the same mistake is a
    // replay-time validation error naming nothing.
    enum class VerbScope
    {
        Render,
        General,
        Any
    };

    // What a verb needs from the queue its pass runs on. Refused by QueueKind
    // rather than by the family the queue happens to sit on, so the contract
    // reads the same on a device whose compute or transfer runtime aliases
    // the graphics queue. One rule, not one per driver.
    //
    // Graphics: a blit (and generate_mipmaps, a chain of blits) needs a
    // graphics family and always will. Shader: a dispatch, a bound pipeline,
    // a descriptor set or a clear (vkCmdClearColorImage) need GRAPHICS or
    // COMPUTE — a transfer family runs copies only. Any: copies, fills,
    // barriers, labels, timers.
    enum class QueueNeeds
    {
        Any,
        Shader,
        Graphics
    };

    bool is_render() const
    {
        return target_ != nullptr;
    }
    const std::shared_ptr<RenderTarget>& target() const
    {
        return target_;
    }
    const std::string& name() const
    {
        return name_;
    }
    QueueKind queue() const
    {
        return queue_;
    }

    // A disabled pass is absent from the compile: barriers are computed as if
    // it were not there, and whatever it wrote is stale for the passes after
    // it — the caller's responsibility, exactly as a submit it chose to skip.
    bool enabled() const
    {
        return enabled_;
    }
    void set_enabled(bool on);

    // Sealing is the point after which the compile may trust the pass's use
    // list. `with` seals on exit; a naked pass is sealed by the first compile.
    bool sealed() const
    {
        return sealed_;
    }
    void seal()
    {
        sealed_ = true;
    }

    bool removed() const
    {
        return removed_;
    }

    bool preserve() const
    {
        return !clear_colors_.has_value();
    }

    CommandBuffer& recorder()
    {
        return *recorder_;
    }
    const std::vector<UseEvent>& events() const
    {
        return events_;
    }

    // The gate every recording verb passes first: not removed, not sealed,
    // the right kind of pass for the verb, and a queue that can run it.
    std::expected<void, Error> guard(VerbScope scope, const char* verb, QueueNeeds needs = QueueNeeds::Any) const;

    // Marks the owning graph dirty, surviving the graph's death (a Python
    // handle may outlive it).
    void mark_graph_dirty();

private:
    friend class Graph;

    std::weak_ptr<Graph> graph_;
    std::shared_ptr<CommandBuffer> recorder_;
    std::vector<UseEvent> events_;
    std::shared_ptr<RenderTarget> target_;
    std::optional<std::vector<std::array<float, 4>>> clear_colors_;
    float clear_depth_ = 1.0f;
    std::uint32_t clear_stencil_ = 0;
    std::string name_;
    QueueKind queue_ = QueueKind::Graphics;
    bool enabled_ = true;
    bool sealed_ = false;
    bool removed_ = false;
};

// THE way to describe GPU work since 0.28: passes on a graph, in add order.
// The graph compiles barriers and attachment transitions for the whole frame
// at once — every use has a visible predecessor, so a first touch gets a
// precise edge instead of a per-recording floor, and two passes that preserve
// one target stop round-tripping the attachment layout.
//
// Record once and submit every frame, or reset() and rebuild each frame —
// both replay closures, so neither allocates on the device. The compile runs
// on the first submit after anything changed (add_pass, remove, reset,
// enabled) and is cached until then.
//
// The graph never reorders passes. On one queue a topological sort could only
// produce the order the caller wrote or a surprise, and determinism is a
// prototyping feature. Two queues change nothing about that: the fold groups
// maximal runs of same-queue passes into BATCHES, in add order, and a use whose
// producer sits in another batch becomes a semaphore wait rather than a moved
// pass. (0.28's comment here claimed the batches already existed. They did not
// — Pass::queue_ was written and never read. 0.29 built them.)
class Graph : public std::enable_shared_from_this<Graph>
{
public:
    static std::expected<std::shared_ptr<Graph>, Error> create(Context& context);

    // Deferred: a per-frame VkCommandBuffer may still be executing when the
    // Python object is dropped.
    ~Graph();

    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    const Context* owner() const
    {
        return context_.get();
    }

    // target=null adds a compute/transfer pass; a target makes it a render
    // pass (rule 1: the variant differs by one parameter). clear_colors
    // follows begin_rendering's contract: nullopt preserves, empty clears to
    // black. The binding layer refuses clear arguments without a target.
    std::shared_ptr<Pass> add_pass(
        std::shared_ptr<RenderTarget> target,
        std::optional<std::vector<std::array<float, 4>>> clear_colors,
        float clear_depth,
        std::uint32_t clear_stencil,
        std::string name,
        QueueKind queue,
        std::optional<bool> auto_barriers);

    std::expected<void, Error> remove(const std::shared_ptr<Pass>& pass);

    // Drop every pass and keep the GPU objects (per-slot command buffers,
    // and each pass's query pools die with the pass). The rebuild-per-frame
    // idiom: reset(), re-add passes, submit.
    void reset();

    void mark_dirty()
    {
        dirty_ = true;
    }

    const std::vector<std::shared_ptr<Pass>>& passes() const
    {
        return passes_;
    }

    // A maximal run of consecutive enabled passes on ONE queue, which is what
    // a submit is made of. Add order decides them, and nothing reorders: a
    // pass whose producer sits in an earlier batch on the other queue gets a
    // semaphore wait, never a move.
    struct Batch
    {
        QueueKind queue = QueueKind::Graphics;
        // The half-open range of compiled_ this batch replays.
        std::size_t first = 0;
        std::size_t last = 0;
        // The n-th batch on this queue, which picks its command-buffer row.
        std::size_t ordinal = 0;
        // Earlier batches on the OTHER queue whose work this one must wait for
        // (the cross-queue half of the fold). Sorted, unique, and every entry
        // is below this batch's own index — a batch never waits for one that
        // has not been submitted yet.
        std::vector<std::size_t> waits;
    };

    // Compile when dirty, sealing every pass. The batches and their command
    // buffers are both decided here, so a caller that wants either asks for
    // this first.
    std::expected<void, Error> compile();

    // A human-readable report of what the compile decided: each enabled pass,
    // its queue and batch, and every barrier, timeline wait and attachment
    // transition the fold emitted, with the pass that produced each
    // dependency. Compiles first when the graph changed. A debugging aid —
    // the text is not API.
    std::expected<std::string, Error> explain();

    std::span<const Batch> batches() const
    {
        return batches_;
    }

    VkCommandBuffer command_buffer(const Batch& batch, std::uint32_t frame_index) const
    {
        const std::vector<VkCommandBuffer>& row = command_buffers_[queue_index(batch.queue)];
        return row[batch.ordinal * context_->frames_in_flight() + frame_index];
    }

    // What THIS graph's previous replay left running on each queue. A batch
    // waits the other queues' values before it starts, which is the
    // wrap-around barrier's argument one level up: frame N+1 of a graph races
    // its own frame N, and a pipeline barrier cannot reach across a queue.
    //
    // Ungated, unlike the wrap-around barrier. That flag counts descriptor
    // writes only, and the writes that matter most here are the ones it cannot
    // see: an attachment a render pass draws into, and a copy_image or a
    // clear_image, which reach the fold as notes. A wait on a value the other
    // queue has already passed costs one semaphore entry and nothing else, so
    // the cheap answer is also the correct one.
    QueueSerials replay_wait() const
    {
        return last_replay_;
    }

    // Merged rather than assigned: a submit that failed halfway still put work
    // on one queue, and forgetting it would leave the next replay unordered
    // against work that is running.
    void note_replay_serials(const QueueSerials& serials)
    {
        max_merge(last_replay_, serials);
    }

    // One Graph owns one VkCommandBuffer per batch per ring slot — the same
    // limit a CommandBuffer carried, moved here with the message rewritten.
    std::expected<void, Error> claim_for_frame(std::uint64_t serial);

    // Replay one batch's passes into vkCmd: entry barriers, the rendering
    // scope for render passes, the pass's commands with scheduled barriers
    // interleaved, exit transitions.
    void execute_batch(const Batch& batch, VkCommandBuffer vkCmd, const FrameContext& frame);

private:
    explicit Graph(std::shared_ptr<Context> context)
        : context_(std::move(context))
    {
    }

    // The barriers between two points of the replay, emitted as ONE
    // vkCmdPipelineBarrier with merged stage masks — strictly narrower than
    // per-resource calls in count and no wider in meaning (a union of stage
    // masks is sound; the access masks stay per resource). Handles resolve at
    // replay, never at compile: a DynamicBuffer has one VkBuffer per ring
    // slot and a swapchain hands out a different image each frame.
    struct BarrierBatch
    {
        std::vector<std::pair<std::shared_ptr<Buffer>, ResourceTracker::Barrier>> buffers;
        std::vector<std::pair<std::shared_ptr<Image>, ResourceTracker::ImageBarrier>> images;

        bool empty() const
        {
            return buffers.empty() && images.empty();
        }
        void record(VkCommandBuffer cmd, const FrameContext& frame) const;
    };

    struct CompiledPass
    {
        Pass* pass = nullptr;
        // Which batch replays it — the index into batches_.
        std::size_t batch = 0;
        // A render pass hoists every barrier to its entry (vkCmdPipelineBarrier
        // is illegal inside dynamic rendering); a general pass schedules them
        // at the command index the use recorded.
        BarrierBatch entry;
        std::vector<std::pair<std::size_t, BarrierBatch>> mid;
        // The look-ahead: this pass and the next render into one target and
        // the next preserves, so the attachment layout survives the seam.
        // elide_exit skips the retire transitions; elide_entry replaces the
        // entry transitions with an in-place execution barrier
        // (attachment-write before attachment-read-and-write, no layout
        // change). MSAA never preserves (refused at the binding), so the
        // elision never meets a resolve image.
        bool elide_exit = false;
        bool elide_entry = false;
    };

    // One line of graph.explain(): something the compile decided, attributed
    // to the pass it runs for and the pass that produced the dependency.
    // Filled beside the real emission in compile_, so the report cannot drift
    // from what the executor replays.
    struct ExplainEntry
    {
        enum class Kind
        {
            Barrier,    // a vkCmdPipelineBarrier the fold computed
            Floor,      // the same, from a first-use floor (no producer here)
            Wait,       // a cross-queue edge that became a timeline wait
            Attachment, // a render pass's entry transition (from the target)
            Retire,     // its exit transition
            Elided,     // an entry/exit the look-ahead removed
            Unordered   // a manual pass's use no barrier covers (the lint)
        };
        Kind kind;
        std::size_t pass = 0;
        std::shared_ptr<Buffer> buffer; // one of the two, or neither for an
        std::shared_ptr<Image> image;   // attachment row
        std::size_t producer = ResourceTracker::kNoPass;
        bool producer_wrote = true;
        ResourceTracker::ImageBarrier b{}; // layouts UNDEFINED for buffers
        std::string note;                  // extra text: "color[0]", "(clear)", ...
    };
    std::vector<ExplainEntry> explain_;

    // Classify what one tracker call did and append the entries. `prev` is
    // the state snapshot from before the call (null on a true first use).
    // The manual-pass lint (0.30): one WARNING per (pass, resource) per
    // compile when a manual use needs a barrier or a wait no p.barrier() in
    // the pass established. Never an exception — a manual pass exists because
    // it may know better, and an exception would close the escape hatch rule
    // 2 requires.
    void warn_manual_hazard_(
        std::size_t pass_index,
        const UseEvent& e,
        const ResourceTracker::BufferState* buffer_state,
        const ResourceTracker::ImageState* image_state,
        ResourceTracker::Peek peek);

    template <typename State>
    void record_explain_(
        std::size_t pass,
        const std::shared_ptr<Buffer>& buffer,
        const std::shared_ptr<Image>& image,
        const State* prev,
        bool had_prev,
        QueueKind queue,
        std::size_t waits_before,
        const std::vector<std::size_t>& waits,
        const std::optional<ResourceTracker::ImageBarrier>& barrier);

    std::expected<void, Error> compile_();

    // Allocate whatever command-buffer rows the batches now need, from each
    // batch's own queue pool. Grow-only: a row entry is re-recorded only by
    // the same (queue, ordinal, slot), and per-queue slot pacing proves that
    // slot's previous replay finished on every queue first — the same argument
    // one buffer per slot always rested on.
    std::expected<void, Error> ensure_command_buffers_();
    // Route one computed barrier to where the executor must emit it: a render
    // pass's entry batch, or a general pass's schedule at `position`.
    static BarrierBatch& batch_at_(CompiledPass& cp, std::size_t position);

    // Bring a preserving pass's attachments to the layout its own entry
    // transition assumes, when something in this graph moved them since.
    // A member since 0.30: it appends explain entries.
    void correct_preserve_entry_(CompiledPass& cp, ResourceTracker& tracker, std::vector<std::size_t>& waits);

    // Report a render pass's attachment writes to the fold, so a later pass
    // that samples one is ordered against the drawing.
    static void note_attachment_writes_(const Pass& pass, ResourceTracker& tracker);

    std::shared_ptr<Context> context_;
    std::vector<std::shared_ptr<Pass>> passes_;
    // Per queue, indexed [ordinal * frames_in_flight + slot].
    std::array<std::vector<VkCommandBuffer>, kQueueCount> command_buffers_;
    std::vector<CompiledPass> compiled_;
    std::vector<Batch> batches_;
    // What the previous replay signalled on each queue — see replay_wait().
    QueueSerials last_replay_{};
    bool dirty_ = true;
    // Any enabled pass writes a tracked resource → the replay wrap-around
    // barrier at the top, exactly as a writing CommandBuffer recording emits
    // it: frame N+1 of this graph races its own frame N.
    bool tracked_writes_ = false;
    // Frame serial of the last replay; UINT64_MAX = never replayed (headless
    // serial 0 is legitimate).
    std::uint64_t recorded_serial_ = UINT64_MAX;
};
