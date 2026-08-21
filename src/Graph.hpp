#pragma once
#include <volk.h>
#include <array>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "CommandBuffer.hpp"

// Which queue a pass runs on. One member today, and Queue.COMPUTE deliberately
// does NOT exist yet: 0.29 adds the member (the Feature-row pattern — a new
// capability is a new enum value, never a new parameter), together with the
// second QueueRuntime it needs. Accepting COMPUTE now and running it
// sequentially was rejected: 0.29 would then silently change the scheduling of
// unedited 0.28 programs, a behaviour break dressed as a no-op.
enum class QueueKind
{
    Graphics
};

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
    // and the right kind of pass for the verb.
    std::expected<void, Error> guard(VerbScope scope, const char* verb) const;

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
// prototyping feature. The 0.29 cross-queue edges come from the same fold
// without reordering either: the fold groups maximal runs of same-queue
// passes into batches, and today that is always exactly one batch.
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

    VkCommandBuffer get(std::uint32_t frame_index) const
    {
        return command_buffers_[frame_index];
    }

    // One Graph owns one VkCommandBuffer per ring slot — the same limit a
    // CommandBuffer carried, moved here with the message rewritten.
    std::expected<void, Error> claim_for_frame(std::uint64_t serial);

    // Compile if dirty (sealing every pass), then replay every enabled pass
    // into vkCmd: entry barriers, the rendering scope for render passes, the
    // pass's commands with scheduled barriers interleaved, exit transitions.
    void execute(VkCommandBuffer vkCmd, const FrameContext& frame);

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

    void compile_();
    // Route one computed barrier to where the executor must emit it: a render
    // pass's entry batch, or a general pass's schedule at `position`.
    BarrierBatch& batch_at_(CompiledPass& cp, std::size_t position);

    // Bring a preserving pass's attachments to the layout its own entry
    // transition assumes, when something in this graph moved them since.
    void correct_preserve_entry_(CompiledPass& cp, ResourceTracker& tracker);

    // Report a render pass's attachment writes to the fold, so a later pass
    // that samples one is ordered against the drawing.
    void note_attachment_writes_(const Pass& pass, ResourceTracker& tracker);

    std::shared_ptr<Context> context_;
    std::vector<std::shared_ptr<Pass>> passes_;
    std::vector<VkCommandBuffer> command_buffers_;
    std::vector<CompiledPass> compiled_;
    bool dirty_ = true;
    // Any enabled pass writes a tracked resource → the replay wrap-around
    // barrier at the top, exactly as a writing CommandBuffer recording emits
    // it: frame N+1 of this graph races its own frame N.
    bool tracked_writes_ = false;
    // Frame serial of the last replay; UINT64_MAX = never replayed (headless
    // serial 0 is legitimate).
    std::uint64_t recorded_serial_ = UINT64_MAX;
};
