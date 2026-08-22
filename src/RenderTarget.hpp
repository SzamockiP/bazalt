#pragma once
#include <volk.h>
#include <vk_mem_alloc.h>

#include <cstdint>
#include <expected>
#include <map>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#include "Context.hpp"
#include "Error.hpp"
#include "Format.hpp"
#include "Image.hpp"
#include "ImmediateSubmit.hpp"

// Anything that can be drawn into.
//
// This interface exists to answer the only four questions a recorded command
// actually asks at replay time: which colour attachments, which depth
// attachment, how big, and what layout the result must end in. CommandBuffer
// used to take a `SwapchainRenderer&` for exactly that, which is why headless
// rendering, render-to-texture and MRT were all impossible at once, and why
// end_rendering could hardcode VK_IMAGE_LAYOUT_PRESENT_SRC_KHR.
//
// A swapchain is now one implementation of this, not the whole world.

// Turn a user-facing sample count (1/2/4/8/…) into the Vulkan flag bit, rejecting
// anything this GPU can't back with both a colour and a depth attachment. One
// count serves every attachment in a pass, so validating against
// Context::max_samples() (the colour∩depth intersection) is the whole check.
// Same constructor-contract shape as the format guards below: a bad value fails
// loudly with a fix now, not a validation-layer crash at draw time.
std::expected<VkSampleCountFlagBits, Error> validate_sample_count(std::uint32_t samples, const Context& context);

// Transition every subresource of `image` that is not already in `layout` up to
// it, then collapse the layout state. A no-op — no barrier recorded at all —
// when the image is already uniform, which is every image that was written
// whole. See RenderTarget::record_even_out for why this exists.
void even_out_image(
    const VolkDeviceTable& vk,
    VkCommandBuffer cmd,
    Image& image,
    VkImageLayout layout,
    VkImageAspectFlags aspect);

class RenderTarget
{
public:
    virtual ~RenderTarget() = default;

    virtual std::uint32_t color_count() const = 0;

    // Queried at replay time, not record time: a swapchain hands out a different
    // image every frame.
    virtual VkImage color_image(std::uint32_t index) const = 0;
    virtual VkImageView color_view(std::uint32_t index) const = 0;
    virtual VkFormat color_format(std::uint32_t index) const = 0;

    // VK_NULL_HANDLE when the target has no depth attachment.
    virtual VkImage depth_image() const = 0;
    virtual VkImageView depth_view() const = 0;
    virtual VkFormat depth_format() const = 0;

    virtual VkExtent2D extent() const = 0;

    // Which Context this object belongs to. Multi-context (0.15) made "a
    // resource from the other Context" a reachable mistake, and its symptom
    // without a check is a driver crash or a validation message from Vulkan
    // rather than from bazalt; the binding layer compares owners at record time.
    virtual const Context* owner() const = 0;

    // ── MSAA ──────────────────────────────────────────────────────────────────
    // A non-multisampled target answers 1-sample / VK_NULL_HANDLE to all of these,
    // so CommandBuffer's resolve wiring vanishes for it (resolveMode stays NONE).
    //
    // When samples() > 1, color_image/color_view return the *multisampled* images
    // that are rendered into, and color_resolve_* return the single-sample images
    // the pass resolves into — the ones that become sampleable/presentable and
    // that final_layout() applies to. Depth resolves the same way (SAMPLE_ZERO)
    // only when the target keeps its depth (offscreen); a swapchain's scratch
    // depth is multisampled but never resolved.
    virtual VkSampleCountFlagBits samples() const
    {
        return VK_SAMPLE_COUNT_1_BIT;
    }
    virtual VkImage color_resolve_image(std::uint32_t) const
    {
        return VK_NULL_HANDLE;
    }
    virtual VkImageView color_resolve_view(std::uint32_t) const
    {
        return VK_NULL_HANDLE;
    }
    virtual VkImage depth_resolve_image() const
    {
        return VK_NULL_HANDLE;
    }
    virtual VkImageView depth_resolve_view() const
    {
        return VK_NULL_HANDLE;
    }

    // Whether the multisampled attachment survives the pass that renders it.
    //
    // False is the default and the cheap answer: the samples are resolved and
    // then discarded (STORE_OP_DONT_CARE), so on a tiled GPU they never leave
    // tile memory. True stores them, which costs the bandwidth of a full
    // multisample buffer and is the price of reading them back through a
    // sampler2DMS — a custom resolve.
    //
    // It is a property of the target rather than of the pass because the target
    // owns the image, and because the recording that renders it cannot see the
    // one that reads it: those are different command buffers, often different
    // frames.
    virtual bool keep_samples() const
    {
        return false;
    }

    // ── Subresource selection (render-to-layer / render-to-mip) ────────────────
    // Which array layer(s) and mip of each attachment a pass actually writes.
    // The default is the whole-image, base-subresource case every target used
    // before 0.13: layer 0, mip 0, one of each. A SubresourceTarget overrides
    // these so CommandBuffer's attachment barriers hit exactly the layer/mip the
    // view renders into — the view and the barrier both read from here, so they
    // cannot drift. extent() (mip-scaled) covers renderArea/viewport/scissor.
    struct Subresource
    {
        std::uint32_t base_layer = 0;
        std::uint32_t layer_count = 1;
        std::uint32_t base_mip = 0;
        std::uint32_t mip_count = 1;
    };
    virtual Subresource color_subresource() const
    {
        return {};
    }
    virtual Subresource depth_subresource() const
    {
        return {};
    }

    // Multiview: a non-zero mask renders every set bit's layer in ONE pass, the
    // shader keying per-view work off gl_ViewIndex. 0 (the default) is the ordinary
    // single-layer path. A MultiviewTarget (RenderTarget.all_layers()) returns
    // (1<<N)-1; its color/depth views span all N layers and its subresource covers
    // them, so the attachment barriers transition the whole array.
    virtual std::uint32_t view_mask() const
    {
        return 0;
    }

    // The layout the colour attachments must be left in when rendering ends.
    // A swapchain needs PRESENT_SRC_KHR; an offscreen target that will be sampled
    // needs SHADER_READ_ONLY_OPTIMAL. This being a virtual is what removes the
    // hardcoded present transition from CommandBuffer.
    virtual VkImageLayout final_layout() const = 0;

    // Same question for the depth attachment. The swapchain's depth buffer is
    // scratch (stays DEPTH_ATTACHMENT_OPTIMAL, store DONT_CARE); an offscreen
    // depth ends sampleable, which is the whole of what makes `shadow.depth` a
    // texture with zero extra API. end_rendering also derives its store-op from
    // this: a depth that will be consumed must be stored.
    //
    // DEPTH_ATTACHMENT_OPTIMAL covers the depth aspect alone, and Vulkan
    // forbids it outright for an image that also carries stencil — so the
    // default reads the format rather than naming one layout. A window with
    // stencil=True lands here, and it is the reason this is not a constant.
    virtual VkImageLayout depth_final_layout() const
    {
        return has_stencil(depth_format()) ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                                           : VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    }

    // The Image objects a pass rendering into this target WRITES, for the graph
    // to fold (0.28). Without them the compile cannot know that a later pass
    // sampling `target.color[0]` reads what an earlier pass drew — an
    // attachment write is not a descriptor use, so nothing else reports it, and
    // "render into a texture, then sample it" is the most ordinary thing a
    // frame does.
    //
    // Empty by default and empty for a swapchain: its images belong to the
    // driver, nothing can sample them, and the only ordering they need is the
    // present semaphore. What comes back is the SAMPLEABLE image, which under
    // MSAA is the resolve target rather than the multisampled one.
    virtual std::vector<std::shared_ptr<Image>> written_color_images() const
    {
        return {};
    }

    virtual std::shared_ptr<Image> written_depth_image() const
    {
        return nullptr;
    }

    // Called by CommandBuffer when the end-of-rendering barrier is recorded into
    // a real submit. An OffscreenTarget uses this to learn that its image has
    // left UNDEFINED — the submit paths never see the target (it lives inside the
    // recorded lambdas), so the notification has to come from the recording
    // itself. No-op for targets that don't care.
    virtual void on_rendering_recorded()
    {
    }

    // Bring the subresources this pass did NOT write up to the final layout too,
    // so the whole image ends where the RenderTarget contract promises.
    //
    // "The layout the result must end in" has always been a promise about the
    // IMAGE, and until 0.18 a subresource pass honoured it for the part it drew
    // and left the rest wherever it was. `target.mip(1)` therefore produced an
    // image that was SHADER_READ_ONLY at level 1 and UNDEFINED at level 0, and
    // the sampler saw one view over both — a validation error at the sample,
    // a long way from the pass that caused it. The old note called this "render
    // every layer and every mip before you sample", which reads as advice and
    // was really a missing barrier.
    //
    // Costs nothing on a whole-image pass: the layout state is uniform there, so
    // there is nothing to even out and this records no barrier at all. Runs
    // AFTER on_rendering_recorded, which is what makes the state true.
    //
    // Transitioning an untouched subresource out of UNDEFINED discards contents
    // that were undefined anyway, so it loses nothing that existed.
    virtual void record_even_out(const VolkDeviceTable& /*vk*/, VkCommandBuffer /*cmd*/)
    {
    }
};

// Everything a recorded command needs that isn't known until replay.
//
// Deliberately does NOT carry the target: begin_rendering captures its own, so a
// single command buffer can render into a shadow map and then a window. A target
// here would be both dead weight and a limit.
struct FrameContext
{
    std::uint32_t frame_index = 0;

    // The recording Context's device dispatch table (Context::vk()). Every
    // vkCmd* inside a recorded lambda goes through this — one field instead of
    // capturing a table pointer in each of the ~35 lambdas, and it keeps the
    // rule that a deferred lambda holds nothing that (indirectly) holds the
    // Context. A raw pointer into a Context that outlives its own recordings.
    const VolkDeviceTable* vk = nullptr;

    // Which pipeline stage bits a barrier recorded into THIS batch's command
    // buffer may carry — the queue family's legal set (0.29). A compute-only
    // family supports far fewer than the graphics one, and every stage bit in
    // a vkCmdPipelineBarrier must be one the pool's family supports. Defaults
    // to everything, so a recorder nobody told is unchanged.
    VkPipelineStageFlags legal_stages = ~VkPipelineStageFlags{0};
};

// A render target backed by Images this object owns, with no swapchain and no
// window involved. This is what makes headless rendering — and therefore the
// test suite — possible. The attachments are ordinary bz.Image objects, which
// is the whole render-to-texture story: `target.color[0]` and `target.depth`
// go straight into set_image() with no extra API.
class OffscreenTarget : public RenderTarget, public std::enable_shared_from_this<OffscreenTarget>
{
public:
    static std::expected<std::shared_ptr<OffscreenTarget>, Error> create(
        Context& context,
        std::uint32_t width,
        std::uint32_t height,
        std::vector<Format> colors,
        std::optional<Format> depth,
        std::uint32_t samples = 1,
        std::uint32_t layers = 1,
        bool cube = false,
        std::uint32_t mip_levels = 1,
        const std::string& name = "",
        bool keep_samples = false);

    // A target on images the caller already owns, instead of attachments this
    // class allocates. Everything create() takes as a knob — extent, layers, cube,
    // mip_levels — is read off the images instead, because the images already
    // answer those questions and a second answer could disagree with them.
    //
    // It earns being a separate entry point rather than optional width/height on
    // create() for the same reason blit_image is not copy_image(scale=True): the
    // two do different jobs. "Allocate attachments for me" and "render into these"
    // share only what happens afterwards.
    //
    // What this makes reachable: a graphics ping-pong between two textures, drawing
    // over a texture a compute pass baked, and drawing into an image brought from
    // another Context. All three were impossible while a target insisted on owning
    // its attachments.
    //
    // Ownership needs no work: the images are held by shared_ptr exactly as the
    // allocated ones are, and the destructor already destroys only its own views.
    // Note that the target does WRITE to a borrowed image's layout tracking
    // (mark_rendered / record_even_out) — that is the point, since final_layout()
    // leaves the result sampleable.
    //
    // samples>1 works here exactly as it does on create(): bazalt allocates the
    // multisampled attachments and the images handed in become their resolve
    // targets, which is what they already are on the allocating path. The
    // alternative — the caller creating a multisampled image and handing THAT in —
    // would need matching resolve images passed alongside it and would put a second
    // MSAA idiom in the API, one where bazalt owns the multisampled image and one
    // where the caller does.
    static std::expected<std::shared_ptr<OffscreenTarget>, Error> create_from_images(
        Context& context,
        std::vector<std::shared_ptr<Image>> colors,
        std::shared_ptr<Image> depth,
        std::uint32_t samples = 1,
        const std::string& name = "",
        bool keep_samples = false);

    OffscreenTarget(const OffscreenTarget&) = delete;
    OffscreenTarget& operator=(const OffscreenTarget&) = delete;

    std::uint32_t color_count() const override
    {
        return static_cast<std::uint32_t>(colors_.size());
    }
    // With MSAA the multisampled image is the one rendered into; colors_ is its
    // resolve target (returned by color_resolve_* below).
    VkImage color_image(std::uint32_t i) const override
    {
        return msaa_colors_.empty() ? colors_[i]->vk_image() : msaa_colors_[i]->vk_image();
    }
    VkImageView color_view(std::uint32_t i) const override
    {
        return msaa_colors_.empty() ? colors_[i]->view() : msaa_colors_[i]->view();
    }
    VkFormat color_format(std::uint32_t i) const override
    {
        return colors_[i]->vk_format();
    }
    VkImage depth_image() const override;
    VkImageView depth_view() const override;
    VkFormat depth_format() const override
    {
        return depth_ ? depth_->vk_format() : VK_FORMAT_UNDEFINED;
    }
    VkExtent2D extent() const override
    {
        return extent_;
    }

    VkSampleCountFlagBits samples() const override
    {
        return samples_;
    }
    VkImage color_resolve_image(std::uint32_t i) const override
    {
        return msaa_colors_.empty() ? VK_NULL_HANDLE : colors_[i]->vk_image();
    }
    VkImageView color_resolve_view(std::uint32_t i) const override
    {
        return msaa_colors_.empty() ? VK_NULL_HANDLE : colors_[i]->view();
    }
    VkImage depth_resolve_image() const override
    {
        return msaa_depth_ ? depth_->vk_image() : VK_NULL_HANDLE;
    }
    VkImageView depth_resolve_view() const override
    {
        return msaa_depth_ ? depth_->view() : VK_NULL_HANDLE;
    }

    // Left ready to be sampled, so using the result as a texture needs no extra
    // step — colour and depth both.
    VkImageLayout final_layout() const override
    {
        return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    // A depth-only attachment is left sampleable, which is the whole of what
    // makes `shadow.depth` a texture. A combined depth/stencil one is not: its
    // view carries two aspects and no sampler can read it, so it stays in the
    // attachment layout and a second pass loads it with no round trip.
    VkImageLayout depth_final_layout() const override;

    // The attachments as Images, for Python and for readback. With MSAA these are
    // the single-sample resolve images — the ones a shader normally reads.
    const std::vector<std::shared_ptr<Image>>& colors() const
    {
        return colors_;
    }
    const std::shared_ptr<Image>& depth() const
    {
        return depth_;
    }

    // What a pass rendering here writes, for the graph's fold. The sampleable
    // images, which is exactly what colors()/depth() already are — under MSAA
    // the resolve targets rather than the multisampled attachments.
    std::vector<std::shared_ptr<Image>> written_color_images() const override
    {
        return colors_;
    }

    std::shared_ptr<Image> written_depth_image() const override
    {
        return depth_;
    }

    // The multisampled attachments themselves, for a custom resolve: bind one to a
    // sampler2DMS and read the samples with texelFetch, which is what per-sample
    // edge detection and a TAA resolve need — the driver's own resolve has already
    // averaged them away.
    //
    // Empty unless the target was created with keep_samples=True, even though the
    // images exist either way. Without it the pass discards them (DONT_CARE), so
    // handing them out would hand out undefined contents: an empty list fails at
    // the line that reads it, and a subtly wrong picture fails nowhere at all.
    const std::vector<std::shared_ptr<Image>>& multisampled_colors() const;
    const std::shared_ptr<Image>& multisampled_depth() const;
    bool keep_samples() const override
    {
        return keep_samples_;
    }

    // Copies colour attachment 0 back to host memory; kept as the ergonomic
    // spelling for tests (target.color[0].read() is the general form).
    std::expected<std::vector<std::byte>, Error> read_pixels();

    // Runs at execute() time, inside a real submit — the attachments learn they
    // have contents exactly when that becomes true (the 0.4.1 read_pixels fix,
    // now spelled per-Image). Depth included: that is what makes shadow maps
    // readable and sampleable.
    void on_rendering_recorded() override
    {
        mark_rendered(color_subresource(), depth_subresource());
    }

    void record_even_out(const VolkDeviceTable& vk, VkCommandBuffer cmd) override;

    // What a pass actually wrote, named by the target that ran it.
    //
    // Until 0.18 this marked the WHOLE image whatever the pass covered, so
    // `target.mip(1)` claimed mip 0 was in the final layout too — and the next
    // barrier over the whole image handed the driver an oldLayout that was true
    // of one level and a lie about the rest. That is why the old note said to
    // render every layer and every mip before sampling. A SubresourceTarget and
    // a MultiviewTarget call this with THEIR subresource, so the statement is
    // always made by whoever knows what was drawn.
    void mark_rendered(const Subresource& color_sr, const Subresource& depth_sr);

    // The layout a depth attachment is in *during* a pass. end_rendering compares
    // against exactly this to decide whether the depth retires at all, so the two
    // read the same source instead of spelling the rule twice.
    VkImageLayout attachment_depth_layout_() const
    {
        return depth_ && has_stencil(depth_->vk_format()) ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                                                          : VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    }

    ~OffscreenTarget();

    // ── render-to-layer / render-to-mip ────────────────────────────────────────
    // A single (layer, mip) subresource of an attachment, as a plain 2D view —
    // what a SubresourceTarget renders into. Cached and owned here; slicing the
    // same subresource twice returns the same view. The *_view accessors return
    // the image actually rendered into (the multisampled one under MSAA); the
    // *_resolve_view accessors return the single-sample resolve subresource (null
    // without MSAA), so a layered MSAA target resolves each layer on its own.
    VkImageView color_subresource_view(std::uint32_t attachment, std::uint32_t layer, std::uint32_t mip);
    VkImageView color_resolve_subresource_view(std::uint32_t attachment, std::uint32_t layer, std::uint32_t mip);
    VkImageView depth_subresource_view(std::uint32_t layer, std::uint32_t mip);
    VkImageView depth_resolve_subresource_view(std::uint32_t layer, std::uint32_t mip);

    // Multiview attachment views: a 2D_ARRAY view over ALL layers at mip 0, what a
    // MultiviewTarget renders every layer through in one pass. The *_resolve_ ones
    // are the single-sample resolve targets (null without MSAA) — a multiview MSAA
    // pass resolves every view into the matching resolve layer.
    VkImageView color_array_view(std::uint32_t attachment);
    VkImageView color_resolve_array_view(std::uint32_t attachment);
    VkImageView depth_array_view();
    VkImageView depth_resolve_array_view();
    std::uint32_t array_layers() const
    {
        return layers_;
    }

    // Bounds-checked slices. Returned as a RenderTargetBase (the view is a
    // RenderTarget), so it passes straight into graph.add_pass(target, ...). `layer(i, mip)`
    // is the general form (both axes); `mip(m)` is sugar for layer 0. A layered
    // AND mipped target (e.g. a mipped cube for prefiltered reflections) needs the
    // combined form.
    std::expected<std::shared_ptr<RenderTarget>, Error> layer(std::uint32_t i, std::uint32_t mip = 0);

    // Multiview: render into EVERY layer in one pass (the shader keys per-view work
    // off gl_ViewIndex) instead of a pass per layer. Needs a layered target and the
    // multiview GPU feature; composes with MSAA (resolves each view per layer).
    std::expected<std::shared_ptr<RenderTarget>, Error> all_layers();

    // True when the color attachment is a 3D image. Such a target is rendered
    // one Z slice at a time through layer(z): the slice view is legal, the 3D
    // main view as an attachment is not — which is why begin_rendering on the
    // whole target is refused at the binding layer.
    bool is_3d() const
    {
        return slices_ > 1;
    }
    std::uint32_t slices() const
    {
        return slices_;
    }

    const Context* owner() const override
    {
        return context_.get();
    }

private:
    explicit OffscreenTarget(std::shared_ptr<Context> context)
        : context_(std::move(context))
    {
    }

    // Every attachment of one target shares its extent, layer count and mip count.
    // Checked against the first attachment rather than pairwise, which is the same
    // result in fewer comparisons and gives a message naming a concrete reference.
    static std::expected<void, Error> require_matching_attachment(
        const Image& reference,
        const Image& image,
        const char* role);

    // Shared body of the view accessors: cache lookup keyed by (VkImage, baseLayer,
    // layerCount, mip) — the image handle disambiguates render vs resolve vs depth —
    // on miss create a single-mip view (2D for one layer, 2D_ARRAY for a multiview
    // span) and store it.
    VkImageView view_(
        const std::shared_ptr<Image>& image,
        VkImageAspectFlags aspect,
        std::uint32_t base_layer,
        std::uint32_t layer_count,
        std::uint32_t mip);

    std::shared_ptr<Context> context_;
    VkExtent2D extent_{};

    // Sampleable single-sample attachments (Python's target.color/target.depth).
    // With MSAA these are the resolve targets; without it they're rendered into
    // directly.
    std::vector<std::shared_ptr<Image>> colors_;
    std::shared_ptr<Image> depth_;

    // The multisampled images actually rendered into. Empty / null unless
    // samples_ > 1; colors_/depth_ then serve as their resolve targets.
    std::vector<std::shared_ptr<Image>> msaa_colors_;
    std::shared_ptr<Image> msaa_depth_;
    bool keep_samples_ = false;
    VkSampleCountFlagBits samples_ = VK_SAMPLE_COUNT_1_BIT;

    // Layer / mip counts every attachment shares. layers_ == 6 for a cube.
    std::uint32_t layers_ = 1;
    std::uint32_t mip_levels_ = 1;
    // Z slices of a 3D attachment (1 for every 2D target). A volume has one
    // array layer, so layers_ and slices_ are never both > 1; layer(i) means
    // "slice i" exactly when slices_ > 1.
    std::uint32_t slices_ = 1;

    // The destructor's whole body, out of line so the destructor itself can wrap
    // it: this allocates, and a throwing destructor terminates.
    void retire_subresource_views_();

    // Lazily created views, keyed (VkImage handle, base layer, layer count, mip).
    // Owned here, destroyed (deferred) in the destructor.
    std::map<std::tuple<std::uint64_t, std::uint32_t, std::uint32_t, std::uint32_t>, VkImageView> subresource_views_;
};

// A render target that is one (layer, mip) subresource of an OffscreenTarget —
// what target.layer(i) / target.mip(m) hand back. It owns no Vulkan objects: the
// per-subresource views live in the parent's cache, the attachment Images are the
// parent's. begin_rendering needs no change to render into it — the attachment
// views come back single-subresource (color_view/depth_view), the barriers read
// the {layer,1,mip,1} range (color_subresource/depth_subresource), and extent()
// is mip-scaled so renderArea/viewport/scissor shrink to the mip automatically.
class SubresourceTarget : public RenderTarget
{
public:
    const Context* owner() const override
    {
        return parent_->owner();
    }

    SubresourceTarget(std::shared_ptr<OffscreenTarget> parent, std::uint32_t layer, std::uint32_t mip)
        : parent_(std::move(parent)),
          layer_(layer),
          mip_(mip)
    {
    }

    std::uint32_t color_count() const override
    {
        return parent_->color_count();
    }
    VkImage color_image(std::uint32_t i) const override
    {
        return parent_->color_image(i);
    }
    VkImageView color_view(std::uint32_t i) const override
    {
        return parent_->color_subresource_view(i, layer_, mip_);
    }
    VkFormat color_format(std::uint32_t i) const override
    {
        return parent_->color_format(i);
    }
    VkImage depth_image() const override
    {
        return parent_->depth_image();
    }
    VkImageView depth_view() const override
    {
        return parent_->depth_subresource_view(layer_, mip_);
    }
    VkFormat depth_format() const override
    {
        return parent_->depth_format();
    }

    // The whole point of "no CommandBuffer edits for renderArea": a .mip(m) target
    // reports the mip's dimensions, so the pass covers exactly that mip.
    VkExtent2D extent() const override;

    // MSAA composes with layers: the multisampled attachment and its single-sample
    // resolve target are both sliced to this subresource, so CommandBuffer resolves
    // exactly this layer. Without MSAA these return 1 / VK_NULL_HANDLE (from the
    // parent) and the resolve wiring stays off.
    VkSampleCountFlagBits samples() const override
    {
        return parent_->samples();
    }
    VkImage color_resolve_image(std::uint32_t i) const override
    {
        return parent_->color_resolve_image(i);
    }
    VkImageView color_resolve_view(std::uint32_t i) const override
    {
        return parent_->color_resolve_subresource_view(i, layer_, mip_);
    }
    VkImage depth_resolve_image() const override
    {
        return parent_->depth_resolve_image();
    }
    VkImageView depth_resolve_view() const override
    {
        return parent_->depth_resolve_subresource_view(layer_, mip_);
    }

    VkImageLayout final_layout() const override
    {
        return parent_->final_layout();
    }
    VkImageLayout depth_final_layout() const override
    {
        return parent_->depth_final_layout();
    }

    // The parent's images: rendering into one slice still writes that image, and
    // a later pass sampling the whole thing has to be ordered against it.
    std::vector<std::shared_ptr<Image>> written_color_images() const override
    {
        return parent_->written_color_images();
    }
    std::shared_ptr<Image> written_depth_image() const override
    {
        return parent_->written_depth_image();
    }

    // For a 3D parent the view axis and the barrier axis diverge, on purpose —
    // the one exception to the 0.13 "view and barrier come from one (layer,
    // mip)" rule. The slice index feeds only the VIEW (baseArrayLayer selects
    // the Z slice of a 2D_ARRAY_COMPATIBLE volume); Vulkan tracks the layout of
    // a 3D image per mip with exactly one array layer, so the barrier and the
    // marking must name layer 0 or they would index past the layout state. The
    // count is spelled VK_REMAINING_ARRAY_LAYERS for a volume — the layers warn
    // about the narrower form — and mark_subresource_contents clamps it back.
    // Consequence: rendering one slice marks the whole mip — correct, because
    // that IS the granularity a volume's layout has.
    Subresource color_subresource() const override
    {
        return parent_->is_3d() ? Subresource{0, VK_REMAINING_ARRAY_LAYERS, mip_, 1} : Subresource{layer_, 1, mip_, 1};
    }
    Subresource depth_subresource() const override
    {
        return parent_->is_3d() ? Subresource{0, VK_REMAINING_ARRAY_LAYERS, mip_, 1} : Subresource{layer_, 1, mip_, 1};
    }

    // Tells the parent that exactly THIS layer and mip are now in the final
    // layout — not the whole image, which is what it used to say and what made
    // "render every layer before you sample" a rule rather than an optimization.
    // The Image collapses back to one layout as soon as the last subresource
    // catches up, so a fully rendered target still costs one barrier.
    void on_rendering_recorded() override
    {
        parent_->mark_rendered(color_subresource(), depth_subresource());
    }

    void record_even_out(const VolkDeviceTable& vk, VkCommandBuffer cmd) override
    {
        parent_->record_even_out(vk, cmd);
    }

private:
    std::shared_ptr<OffscreenTarget> parent_;
    std::uint32_t layer_;
    std::uint32_t mip_;
};

// Renders into EVERY layer of an OffscreenTarget in one pass via multiview — what
// target.all_layers() hands back. The attachment views span all layers (2D_ARRAY),
// view_mask() lights one bit per layer, and the barriers cover the whole array; the
// shader selects per-layer work with gl_ViewIndex. Composes with MSAA (each view
// resolves into its own layer), and because it renders every layer, the whole-image
// sampleable mark is exactly correct — no partial-render caveat.
class MultiviewTarget : public RenderTarget
{
public:
    const Context* owner() const override
    {
        return parent_->owner();
    }

    explicit MultiviewTarget(std::shared_ptr<OffscreenTarget> parent)
        : parent_(std::move(parent))
    {
    }

    std::uint32_t color_count() const override
    {
        return parent_->color_count();
    }
    VkImage color_image(std::uint32_t i) const override
    {
        return parent_->color_image(i);
    }
    VkImageView color_view(std::uint32_t i) const override
    {
        return parent_->color_array_view(i);
    }
    VkFormat color_format(std::uint32_t i) const override
    {
        return parent_->color_format(i);
    }
    VkImage depth_image() const override
    {
        return parent_->depth_image();
    }
    VkImageView depth_view() const override
    {
        return parent_->depth_array_view();
    }
    VkFormat depth_format() const override
    {
        return parent_->depth_format();
    }
    VkExtent2D extent() const override
    {
        return parent_->extent();
    }

    std::uint32_t view_mask() const override
    {
        const std::uint32_t n = parent_->array_layers();
        return n >= 32 ? 0xFFFFFFFFu : ((1u << n) - 1u);
    }

    // MSAA composes with multiview: one pass renders every layer of the
    // multisampled attachment and resolves each view into the matching resolve
    // layer. The array views span all layers, so the resolve is per-view. Without
    // MSAA these come back 1 / VK_NULL_HANDLE from the parent (resolve wiring off).
    VkSampleCountFlagBits samples() const override
    {
        return parent_->samples();
    }
    VkImage color_resolve_image(std::uint32_t i) const override
    {
        return parent_->color_resolve_image(i);
    }
    VkImageView color_resolve_view(std::uint32_t i) const override
    {
        return parent_->color_resolve_array_view(i);
    }
    VkImage depth_resolve_image() const override
    {
        return parent_->depth_resolve_image();
    }
    VkImageView depth_resolve_view() const override
    {
        return parent_->depth_resolve_array_view();
    }

    Subresource color_subresource() const override
    {
        return {0, parent_->array_layers(), 0, 1};
    }
    Subresource depth_subresource() const override
    {
        return {0, parent_->array_layers(), 0, 1};
    }

    VkImageLayout final_layout() const override
    {
        return parent_->final_layout();
    }
    VkImageLayout depth_final_layout() const override
    {
        return parent_->depth_final_layout();
    }

    // The parent's images: rendering into one slice still writes that image, and
    // a later pass sampling the whole thing has to be ordered against it.
    std::vector<std::shared_ptr<Image>> written_color_images() const override
    {
        return parent_->written_color_images();
    }
    std::shared_ptr<Image> written_depth_image() const override
    {
        return parent_->written_depth_image();
    }

    // Multiview writes every layer in one pass, so its subresource already spans
    // the whole array and the parent marks all of it.
    void on_rendering_recorded() override
    {
        parent_->mark_rendered(color_subresource(), depth_subresource());
    }

    void record_even_out(const VolkDeviceTable& vk, VkCommandBuffer cmd) override
    {
        parent_->record_even_out(vk, cmd);
    }

private:
    std::shared_ptr<OffscreenTarget> parent_;
};
