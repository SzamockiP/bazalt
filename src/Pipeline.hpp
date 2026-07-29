#pragma once
#include <volk.h>
#include <algorithm>
#include <functional>
#include <vector>
#include <memory>
#include <expected>
#include <array>
#include <map>
#include <optional>
#include <ranges>
#include <unordered_map>
#include <utility>
#include "ShaderCompiler.hpp"
#include "Context.hpp"
#include "RenderTarget.hpp"
#include "ScopeGuard.hpp"
// For CompareOp, which the depth test shares with compare samplers rather than
// declaring a second eight-value enum of its own.
#include "Sampler.hpp"

// Renamed from Format: this describes a vertex attribute, and `Format` is needed
// for pixel formats in 0.5.
// New entries are APPENDED: a pybind enum's underlying values are part of the
// API the moment somebody pickles or stores one.
enum class VertexFormat
{
    FLOAT2,
    FLOAT3,
    FLOAT4,
    FLOAT,
    // Four bytes read as 0..1 floats — vertex colours and skin weights, which
    // are a quarter of the size of the FLOAT4 they used to need.
    UBYTE4_NORM,
    // An unsigned integer attribute (`in uint` in GLSL, no conversion). A
    // material index or an object id carried per instance.
    UINT,
};

// The Vulkan format and the byte size of one attribute. One table instead of a
// switch per consumer: the offsets, the stride and the attribute description all
// have to agree, and they used to be derived in one place by accident rather
// than by construction.
struct VertexFormatInfo
{
    VkFormat vk;
    std::uint32_t size;
};

inline constexpr VertexFormatInfo vertex_format_info(VertexFormat format)
{
    switch (format)
    {
        case VertexFormat::FLOAT2:
            return {VK_FORMAT_R32G32_SFLOAT, 8};
        case VertexFormat::FLOAT3:
            return {VK_FORMAT_R32G32B32_SFLOAT, 12};
        case VertexFormat::FLOAT4:
            return {VK_FORMAT_R32G32B32A32_SFLOAT, 16};
        case VertexFormat::FLOAT:
            return {VK_FORMAT_R32_SFLOAT, 4};
        case VertexFormat::UBYTE4_NORM:
            return {VK_FORMAT_R8G8B8A8_UNORM, 4};
        case VertexFormat::UINT:
            return {VK_FORMAT_R32_UINT, 4};
    }
    // Not std::unreachable(): pybind enums accept arbitrary ints.
    return {VK_FORMAT_R32G32B32_SFLOAT, 12};
}

enum class CullMode
{
    NONE,
    BACK,
    FRONT,
    FRONT_AND_BACK
};

enum class FrontFace
{
    CLOCKWISE,
    COUNTER_CLOCKWISE
};

enum class Topology
{
    TRIANGLE_LIST,
    POINT_LIST,
    LINE_LIST,
    // Strips: each new vertex extends the primitive instead of starting one, so
    // a quad is 4 vertices instead of 6. primitiveRestartEnable stays FALSE —
    // one strip per draw. A restart index is a separate decision, and it would
    // change what an index buffer's largest value means.
    TRIANGLE_STRIP,
    LINE_STRIP,
    // The input to a tessellation control shader: a run of patch_control_points
    // vertices with no implied topology at all. What the patch becomes is decided
    // by the tessellation evaluation shader's own `layout(triangles)` and the
    // tessellation levels the control shader writes, so there is no PATCH_STRIP
    // and no separate triangle/quad/isoline spelling here.
    PATCH_LIST,
};

// How a fragment's colour combines with what the attachment already holds.
// blend(True) used to mean ALPHA and nothing else, which left additive glow and
// premultiplied compositing unreachable.
enum class BlendMode
{
    // src.a * src + (1 - src.a) * dst — ordinary transparency.
    ALPHA,
    // src + dst. Particles, glow, light accumulation: order does not matter and
    // nothing is ever darkened.
    ADDITIVE,
    // src + (1 - src.a) * dst, for colours that already carry their alpha —
    // what a composited texture or a text atlas wants.
    PREMULTIPLIED
};

// What happens to a stencil value when a fragment arrives. 1:1 with VkStencilOp.
// The compare op is CompareOp, shared with the depth test and the compare
// samplers rather than declared a third time.
enum class StencilOp
{
    KEEP,
    ZERO,
    REPLACE, // write the reference value — how a mask is painted
    INCREMENT_CLAMP,
    DECREMENT_CLAMP,
    INVERT,
    INCREMENT_WRAP,
    DECREMENT_WRAP,
};

inline constexpr VkStencilOp to_vk(StencilOp op)
{
    switch (op)
    {
        case StencilOp::KEEP:
            return VK_STENCIL_OP_KEEP;
        case StencilOp::ZERO:
            return VK_STENCIL_OP_ZERO;
        case StencilOp::REPLACE:
            return VK_STENCIL_OP_REPLACE;
        case StencilOp::INCREMENT_CLAMP:
            return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
        case StencilOp::DECREMENT_CLAMP:
            return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
        case StencilOp::INVERT:
            return VK_STENCIL_OP_INVERT;
        case StencilOp::INCREMENT_WRAP:
            return VK_STENCIL_OP_INCREMENT_AND_WRAP;
        case StencilOp::DECREMENT_WRAP:
            return VK_STENCIL_OP_DECREMENT_AND_WRAP;
    }
    // Not std::unreachable(): pybind enums accept arbitrary ints.
    return VK_STENCIL_OP_KEEP;
}

// Fill triangles, or draw only their edges/vertices. LINE is the wireframe debug
// view. All three are core Vulkan with no feature bit; only a lineWidth other
// than 1.0 would need the optional wideLines, and nothing here sets one.
enum class PolygonMode
{
    FILL,
    LINE,
    POINT
};

inline constexpr VkPolygonMode to_vk(PolygonMode mode)
{
    switch (mode)
    {
        case PolygonMode::LINE:
            return VK_POLYGON_MODE_LINE;
        case PolygonMode::POINT:
            return VK_POLYGON_MODE_POINT;
        case PolygonMode::FILL:
            return VK_POLYGON_MODE_FILL;
    }
    return VK_POLYGON_MODE_FILL;
}

inline constexpr VkPrimitiveTopology to_vk(Topology topology)
{
    switch (topology)
    {
        case Topology::TRIANGLE_LIST:
            return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        case Topology::POINT_LIST:
            return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        case Topology::LINE_LIST:
            return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        case Topology::TRIANGLE_STRIP:
            return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        case Topology::LINE_STRIP:
            return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
        case Topology::PATCH_LIST:
            return VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
    }
    // Not std::unreachable(): pybind enums accept arbitrary ints.
    return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
}

// The stencil test as one value, because it is one question with several parts:
// eight separate builder fields would let half of them be set and the rest not.
struct StencilState
{
    bool enable = false;
    CompareOp compare = CompareOp::ALWAYS;
    std::uint32_t reference = 0;
    StencilOp pass_op = StencilOp::KEEP;
    StencilOp fail_op = StencilOp::KEEP;
    StencilOp depth_fail_op = StencilOp::KEEP;
    std::uint32_t read_mask = 0xFF;
    std::uint32_t write_mask = 0xFF;

    VkStencilOpState to_vk_state() const
    {
        return {
            .failOp = to_vk(fail_op),
            .passOp = to_vk(pass_op),
            .depthFailOp = to_vk(depth_fail_op),
            .compareOp = to_vk(compare),
            .compareMask = read_mask,
            .writeMask = write_mask,
            .reference = reference};
    }
};

// The colour-attachment state: how a fragment combines with what is already
// there, and which channels it may touch at all.
struct BlendState
{
    bool enable = false;
    BlendMode mode = BlendMode::ALPHA;
    VkColorComponentFlags write_mask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                                       VK_COLOR_COMPONENT_A_BIT;

    bool operator==(const BlendState&) const = default;
};

// One attachment's deviation from the pipeline-wide BlendState. Every field is
// optional so that two calls naming the same attachment merge instead of
// overwriting each other.
struct BlendOverride
{
    std::optional<bool> enable;
    std::optional<BlendMode> mode;
    std::optional<VkColorComponentFlags> write_mask;

    BlendState applied_to(BlendState base) const
    {
        base.enable = enable.value_or(base.enable);
        base.mode = mode.value_or(base.mode);
        base.write_mask = write_mask.value_or(base.write_mask);
        return base;
    }
};

// A specialization constant: a value baked into the SPIR-V at pipeline creation
// instead of read from a buffer at draw time. The driver folds it, so a
// constant loop count unrolls and a constant `false` deletes the branch behind
// it. One shader therefore serves several pipelines that differ by a number —
// quality levels, a workgroup size, a kernel radius.
//
// int/float/bool in one variant rather than three overloads, because SPIR-V
// stores all three as four bytes and only the interpretation differs. bool must
// be checked before int in the binding: Python's bool IS an int.
struct SpecConstant
{
    std::uint32_t id = 0;
    // The four bytes handed to Vulkan. Kept pre-encoded so the value's type is
    // resolved once, at the call, and never re-guessed at pipeline creation.
    std::uint32_t bytes = 0;
};

// Builds the VkSpecializationInfo for one stage. The data block and the map
// entries must outlive vkCreate*Pipelines, so this returns them together and the
// caller keeps the whole thing alive across the call — the classic trap here is
// returning a VkSpecializationInfo whose pointers dangle immediately.
struct SpecializationBlock
{
    std::vector<VkSpecializationMapEntry> entries;
    std::vector<std::uint32_t> data;
    VkSpecializationInfo info{};

    explicit SpecializationBlock(const std::vector<SpecConstant>& constants)
    {
        entries.reserve(constants.size());
        data.reserve(constants.size());
        for (const SpecConstant& c : constants)
        {
            entries.push_back(
                {.constantID = c.id,
                 .offset = static_cast<std::uint32_t>(data.size() * sizeof(std::uint32_t)),
                 .size = sizeof(std::uint32_t)});
            data.push_back(c.bytes);
        }
        info = {
            .mapEntryCount = static_cast<std::uint32_t>(entries.size()),
            .pMapEntries = entries.data(),
            .dataSize = data.size() * sizeof(std::uint32_t),
            .pData = data.data()};
    }

    // Null when nothing was specialized, which is what pSpecializationInfo
    // wants — an empty-but-present block is legal and pointless.
    const VkSpecializationInfo* get() const
    {
        return entries.empty() ? nullptr : &info;
    }
};

// Everything needed to rebuild a Pipeline's VkPipeline handle in place — the
// hot-reload mechanism. `shaders` are the modules it was built from (the
// watcher matches a changed file against these to decide what to rebuild, and
// holding them keeps a Pipeline's shaders alive for as long as the pipeline).
// `recreate` re-runs the pipeline creation against a fresh device state: it
// captured the fixed-function state and the (unchanged) pipeline layout by
// value and calls shader->get() at call time, so a ShaderModule::replace()d
// module is picked up automatically. Empty for a default-constructed Pipeline.
struct PipelineDesc
{
    std::vector<std::shared_ptr<ShaderModule>> shaders;
    std::function<std::expected<VkPipeline, Error>(Context&)> recreate;
};

class Pipeline
{
public:
    // Maps binding index -> VkDescriptorType so DescriptorSet knows what type to write
    using BindingTypeMap = std::unordered_map<uint32_t, VkDescriptorType>;

    Pipeline(
        std::shared_ptr<Context> context,
        VkPipeline pipeline,
        VkPipelineLayout layout,
        std::vector<VkDescriptorSetLayout> descLayouts = {},
        std::map<uint32_t, BindingTypeMap> bindingTypes = {},
        VkShaderStageFlags pushConstantStages = 0,
        VkPipelineBindPoint bindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        PipelineDesc desc = {})
        : context_(context),
          pipeline_(pipeline),
          layout_(layout),
          desc_layouts_(std::move(descLayouts)),
          binding_types_(std::move(bindingTypes)),
          push_constant_stages_(pushConstantStages),
          bind_point_(bindPoint),
          desc_(std::move(desc))
    {
    }

    // Carried on the Pipeline so command recording doesn't hardcode
    // VK_PIPELINE_BIND_POINT_GRAPHICS. Compute pipelines (0.6) then need no
    // change at the call sites.
    VkPipelineBindPoint bind_point() const
    {
        return bind_point_;
    }

    // The builder already knows which stages the push constant range covers, so
    // push_constants() doesn't need the caller to repeat it — and can't be given
    // a mismatched one.
    VkShaderStageFlags push_constant_stages() const
    {
        return push_constant_stages_;
    }

    ~Pipeline()
    {
        destroy();
    }

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    Pipeline(Pipeline&& other) noexcept
        : context_(std::move(other.context_)),
          pipeline_(other.pipeline_),
          layout_(other.layout_),
          desc_layouts_(std::move(other.desc_layouts_)),
          binding_types_(std::move(other.binding_types_)),
          push_constant_stages_(other.push_constant_stages_),
          bind_point_(other.bind_point_),
          desc_(std::move(other.desc_))
    {
        other.pipeline_ = VK_NULL_HANDLE;
        other.layout_ = VK_NULL_HANDLE;
    }

    Pipeline& operator=(Pipeline&& other) noexcept
    {
        if (this != &other)
        {
            destroy();
            context_ = std::move(other.context_);
            pipeline_ = other.pipeline_;
            layout_ = other.layout_;
            desc_layouts_ = std::move(other.desc_layouts_);
            binding_types_ = std::move(other.binding_types_);
            push_constant_stages_ = other.push_constant_stages_;
            bind_point_ = other.bind_point_;
            desc_ = std::move(other.desc_);

            other.pipeline_ = VK_NULL_HANDLE;
            other.layout_ = VK_NULL_HANDLE;
        }
        return *this;
    }

    VkPipeline get() const
    {
        return pipeline_;
    }
    VkPipelineLayout layout() const
    {
        return layout_;
    }

    // Every stage this pipeline was built from, in pipeline order.
    //
    // Returned as the live list rather than a precomputed "what does this pipeline
    // write" map, and that is the point: the tracker asks each module for its
    // reflection at RECORD time, so a hot-reload replace() is picked up with
    // nothing to invalidate. It is the same trick desc_.recreate uses for the
    // handles, applied to what the handles mean.
    const std::vector<std::shared_ptr<ShaderModule>>& shaders() const
    {
        return desc_.shaders;
    }

    // ── Hot reload ────────────────────────────────────────────────────────────

    // True when this pipeline was built from `module`. The watcher asks this to
    // decide which pipelines a changed shader file forces to rebuild.
    bool uses(const ShaderModule* module) const
    {
        return std::ranges::any_of(
            desc_.shaders, [module](const std::shared_ptr<ShaderModule>& s) { return s.get() == module; });
    }

    // Recreate the VkPipeline from the captured description — the hot-reload
    // path, main thread only. The modules it names may have been
    // ShaderModule::replace()d with fresh handles; recreate() reads them via
    // ->get() now. On success the old VkPipeline retires through the deletion
    // queue (an in-flight frame may still have it bound) and pipeline_ becomes
    // the new handle, which deferred bind_pipeline lambdas pick up on their next
    // replay — no re-recording. On FAILURE pipeline_ is left untouched, so a
    // shader typo keeps the last good pipeline rendering. layout_/desc_layouts_
    // are never rebuilt: bindings come from builder calls, not reflection, so
    // descriptor sets and push-constant ranges stay valid across a reload.
    std::expected<void, Error> rebuild()
    {
        if (!desc_.recreate)
        {
            return std::unexpected(err_shader("This pipeline was not built with a rebuildable description"));
        }
        auto fresh = desc_.recreate(*context_);
        if (!fresh)
        {
            return std::unexpected(fresh.error());
        }
        if (pipeline_ != VK_NULL_HANDLE)
        {
            context_->defer_destroy([vk = &context_->vk(), device = context_->device(), old = pipeline_]
                                    { vk->vkDestroyPipeline(device, old, nullptr); });
        }
        pipeline_ = fresh.value();
        return {};
    }

    VkDescriptorSetLayout descriptor_set_layout(uint32_t setIndex) const
    {
        if (setIndex < desc_layouts_.size())
        {
            return desc_layouts_[setIndex];
        }
        return VK_NULL_HANDLE;
    }

    uint32_t descriptor_set_layout_count() const
    {
        return static_cast<uint32_t>(desc_layouts_.size());
    }

    const BindingTypeMap& binding_types(uint32_t setIndex) const
    {
        static const BindingTypeMap empty;
        auto it = binding_types_.find(setIndex);
        if (it != binding_types_.end())
        {
            return it->second;
        }
        return empty;
    }

private:
    // One teardown for the destructor and move-assignment; it used to be written
    // out twice and the two copies had already started to drift.
    // Deferred: an in-flight frame may still be executing with this pipeline
    // bound.
    void destroy()
    {
        if (!context_)
        {
            return;
        }
        context_->defer_destroy(
            [vk = &context_->vk(),
             device = context_->device(),
             pipeline = pipeline_,
             layout = layout_,
             desc_layouts = desc_layouts_]
            {
                if (pipeline != VK_NULL_HANDLE)
                {
                    vk->vkDestroyPipeline(device, pipeline, nullptr);
                }
                if (layout != VK_NULL_HANDLE)
                {
                    vk->vkDestroyPipelineLayout(device, layout, nullptr);
                }
                for (auto dl : desc_layouts)
                {
                    if (dl != VK_NULL_HANDLE)
                    {
                        vk->vkDestroyDescriptorSetLayout(device, dl, nullptr);
                    }
                }
            });
    }

public:
    // Which Context this object belongs to. Multi-context (0.15) made "a
    // resource from the other Context" a reachable mistake, and its symptom
    // without a check is a driver crash or a validation message from Vulkan
    // rather than from bazalt; the binding layer compares owners at record time.
    const Context* owner() const
    {
        return context_.get();
    }

private:
    std::shared_ptr<Context> context_;
    VkPipeline pipeline_;
    VkPipelineLayout layout_;
    std::vector<VkDescriptorSetLayout> desc_layouts_;
    std::map<uint32_t, BindingTypeMap> binding_types_;
    VkShaderStageFlags push_constant_stages_ = 0;
    VkPipelineBindPoint bind_point_ = VK_PIPELINE_BIND_POINT_GRAPHICS;
    PipelineDesc desc_;
};

// The layout plumbing shared by both pipeline builders: descriptor bindings,
// push-constant ranges, and the Vk objects they become. Internal — Python only
// ever sees the two builders, each of which owns one of these.
class PipelineLayoutBuilder
{
public:
    void add_binding(
        uint32_t binding,
        VkShaderStageFlags stageFlags,
        VkDescriptorType descriptorType,
        uint32_t setIndex)
    {
        auto& bindings = descriptor_bindings_[setIndex];

        auto it = std::ranges::find(bindings, binding, &VkDescriptorSetLayoutBinding::binding);
        if (it != bindings.end())
        {
            it->stageFlags |= stageFlags;
            return;
        }

        bindings.push_back(
            {.binding = binding,
             .descriptorType = descriptorType,
             .descriptorCount = 1,
             .stageFlags = stageFlags,
             .pImmutableSamplers = nullptr});
    }

    void add_push_constant(uint32_t size, VkShaderStageFlags stageFlags)
    {
        push_constant_ranges_.push_back({.stageFlags = stageFlags, .offset = 0, .size = size});
    }

    // One VkDescriptorSetLayout per set index up to the highest one used; gap
    // set indices get an empty layout so shader set numbers stay meaningful.
    // Partially created layouts are the caller's guard's problem.
    std::expected<void, Error> create_set_layouts(
        Context& context,
        std::vector<VkDescriptorSetLayout>& layouts,
        std::map<uint32_t, Pipeline::BindingTypeMap>& bindingTypes) const
    {
        if (descriptor_bindings_.empty())
        {
            return {};
        }

        // Parenthesised to dodge the max() macro from <windows.h>.
        const uint32_t maxSetIndex = (std::ranges::max)(descriptor_bindings_ | std::views::keys);

        for (uint32_t s = 0; s <= maxSetIndex; s++)
        {
            auto it = descriptor_bindings_.find(s);
            const bool has_bindings = it != descriptor_bindings_.end() && !it->second.empty();

            VkDescriptorSetLayoutCreateInfo layoutInfo{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .bindingCount = has_bindings ? static_cast<uint32_t>(it->second.size()) : 0,
                .pBindings = has_bindings ? it->second.data() : nullptr};

            VkDescriptorSetLayout layout;
            if (auto e = check(
                    context.vk().vkCreateDescriptorSetLayout(context.device(), &layoutInfo, nullptr, &layout),
                    "create descriptor set layout for set " + std::to_string(s)))
            {
                return std::unexpected(*e);
            }
            layouts.push_back(layout);

            if (has_bindings)
            {
                Pipeline::BindingTypeMap btm;
                for (const auto& b : it->second)
                {
                    btm[b.binding] = b.descriptorType;
                }
                bindingTypes[s] = std::move(btm);
            }
        }
        return {};
    }

    std::expected<VkPipelineLayout, Error> create_layout(
        Context& context,
        const std::vector<VkDescriptorSetLayout>& layouts) const
    {
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = static_cast<uint32_t>(layouts.size()),
            .pSetLayouts = layouts.empty() ? nullptr : layouts.data(),
            .pushConstantRangeCount = static_cast<uint32_t>(push_constant_ranges_.size()),
            .pPushConstantRanges = push_constant_ranges_.data()};

        VkPipelineLayout pipelineLayout;
        if (auto e = check(
                context.vk().vkCreatePipelineLayout(context.device(), &pipelineLayoutInfo, nullptr, &pipelineLayout),
                "create pipeline layout"))
        {
            return std::unexpected(*e);
        }
        return pipelineLayout;
    }

    VkShaderStageFlags push_constant_stages() const
    {
        return std::ranges::fold_left(
            push_constant_ranges_ | std::views::transform(&VkPushConstantRange::stageFlags),
            VkShaderStageFlags{0},
            std::bit_or{});
    }

private:
    std::vector<VkPushConstantRange> push_constant_ranges_;
    std::map<uint32_t, std::vector<VkDescriptorSetLayoutBinding>> descriptor_bindings_;
};

class GraphicsPipelineBuilder
{
public:
    GraphicsPipelineBuilder(Context& context)
        : context_(context)
    {
    }

    // So the binding layer can register a freshly built pipeline with the
    // hot-reload watcher without a second Context handle.
    Context& context()
    {
        return context_;
    }

    // Chained setters use C++23 deducing this: the object parameter's value
    // category is forwarded, so a chain on a temporary builder moves instead of
    // pinning an lvalue. The pybind layer binds these through lambdas — an
    // explicit object parameter turns &GraphicsPipelineBuilder::vertex_shader into a
    // plain function-pointer type that .def() would misread.

    template <typename Self>
    Self&& vertex_shader(this Self&& self, std::shared_ptr<ShaderModule> shader)
    {
        self.vertex_shader_ = std::move(shader);
        return std::forward<Self>(self);
    }

    template <typename Self>
    Self&& fragment_shader(this Self&& self, std::shared_ptr<ShaderModule> shader)
    {
        self.fragment_shader_ = std::move(shader);
        return std::forward<Self>(self);
    }

    // One verb per stage rather than a `shader(module, stage)` taking the stage as
    // an argument: a module already knows its own stage, so that argument could
    // disagree with it, and vertex_shader/fragment_shader set the pattern in 0.2.
    // The two tessellation stages are set together or not at all — see build().
    template <typename Self>
    Self&& tess_control_shader(this Self&& self, std::shared_ptr<ShaderModule> shader)
    {
        self.tess_control_shader_ = std::move(shader);
        return std::forward<Self>(self);
    }

    template <typename Self>
    Self&& tess_evaluation_shader(this Self&& self, std::shared_ptr<ShaderModule> shader)
    {
        self.tess_evaluation_shader_ = std::move(shader);
        return std::forward<Self>(self);
    }

    template <typename Self>
    Self&& geometry_shader(this Self&& self, std::shared_ptr<ShaderModule> shader)
    {
        self.geometry_shader_ = std::move(shader);
        return std::forward<Self>(self);
    }

    // How many vertices the vertex buffer groups into one patch — the INPUT size,
    // which is why it belongs on the pipeline and not in the shader. The control
    // shader's own `layout(vertices = N) out` is its OUTPUT count, a different
    // number, and nothing here can derive one from the other.
    template <typename Self>
    Self&& patch_control_points(this Self&& self, std::uint32_t count)
    {
        self.patch_control_points_ = count;
        return std::forward<Self>(self);
    }

    template <typename Self>
    Self&& vertex_format(this Self&& self, const std::vector<VertexFormat>& formats)
    {
        self.formats_ = formats;
        return std::forward<Self>(self);
    }

    // The attributes of a SECOND vertex buffer, advanced once per instance
    // instead of once per vertex. `cmd.bind_vertex_buffer(instances, binding=1)`
    // feeds it, and `draw(n, instances=k)` runs the geometry k times with a
    // different slice of it each time — the mesh stays in one buffer and only
    // the per-object data repeats.
    //
    // A separate verb rather than a kwarg on vertex_format: this declares a
    // different binding, not a variant of the same one. Locations continue after
    // the vertex attributes (vertex_format of 3 puts the first instance
    // attribute at location 3), which is the same "location is the index in the
    // list" rule the vertex side already follows. A mat4 is four FLOAT4s and so
    // takes four locations.
    template <typename Self>
    Self&& instance_format(this Self&& self, const std::vector<VertexFormat>& formats)
    {
        self.instance_formats_ = formats;
        return std::forward<Self>(self);
    }

    // write=False keeps the test but stops the pass from updating the depth
    // buffer, which is the condition for a correct transparency pass: sorted
    // transparent geometry must test against the opaque depth without occluding
    // its own siblings. compare= replaces the LESS_OR_EQUAL that used to be
    // hard-coded — GREATER is a reversed-depth buffer, ALWAYS a full-screen pass.
    template <typename Self>
    Self&& depth_test(this Self&& self, bool enable, bool write = true, CompareOp compare = CompareOp::LESS_OR_EQUAL)
    {
        self.depth_test_ = enable;
        self.depth_write_ = write;
        self.depth_compare_ = compare;
        return std::forward<Self>(self);
    }

    // The stencil test, in one verb. Two passes make an outline: the first
    // writes the object's silhouette with
    // `stencil_test(True, compare=ALWAYS, ref=1, pass_op=REPLACE)`, the second
    // draws a scaled copy with `stencil_test(True, compare=NOT_EQUAL, ref=1)`
    // and `depth_test(False)`, so only the pixels around the object survive.
    //
    // Needs a target whose depth attachment has a stencil aspect
    // (`depth=bz.Format.DEPTH_STENCIL`).
    //
    // Front and back faces get the same state. Separate states are a rare case
    // and would double eight parameters on one verb, which is worse than the
    // ceiling; the upgrade path is a face= kwarg, and it is additive.
    template <typename Self>
    Self&& stencil_test(
        this Self&& self,
        bool enable,
        CompareOp compare = CompareOp::ALWAYS,
        std::uint32_t ref = 0,
        StencilOp pass_op = StencilOp::KEEP,
        StencilOp fail_op = StencilOp::KEEP,
        StencilOp depth_fail_op = StencilOp::KEEP,
        std::uint32_t read_mask = 0xFF,
        std::uint32_t write_mask = 0xFF)
    {
        self.stencil_.enable = enable;
        self.stencil_.compare = compare;
        self.stencil_.reference = ref;
        self.stencil_.pass_op = pass_op;
        self.stencil_.fail_op = fail_op;
        self.stencil_.depth_fail_op = depth_fail_op;
        self.stencil_.read_mask = read_mask;
        self.stencil_.write_mask = write_mask;
        return std::forward<Self>(self);
    }

    template <typename Self>
    Self&& cull_mode(this Self&& self, CullMode mode, FrontFace frontFace)
    {
        self.cull_mode_ = mode;
        self.front_face_ = frontFace;
        return std::forward<Self>(self);
    }

    // mode= is a kwarg on the existing verb, not a second method: the question
    // "how does this blend" has one answer per pipeline.
    //
    // attachment= narrows the answer to one colour attachment of an MRT target,
    // and everything without an override keeps what the plain call set. The
    // overrides are per FIELD (optional each), so blend(attachment=1) and
    // color_mask(attachment=1) compose in either order — a resolution that read
    // the default at call time would make the result depend on which line came
    // first, which is exactly the kind of rule nobody remembers at 3am.
    template <typename Self>
    Self&& blend(this Self&& self, bool enable, BlendMode mode = BlendMode::ALPHA, int attachment = -1)
    {
        if (attachment < 0)
        {
            self.blend_.enable = enable;
            self.blend_.mode = mode;
        }
        else
        {
            auto& o = self.blend_overrides_[static_cast<std::uint32_t>(attachment)];
            o.enable = enable;
            o.mode = mode;
        }
        return std::forward<Self>(self);
    }

    // Which channels this pipeline writes. A g-buffer pass that must not touch
    // the alpha of an attachment it shares, or a depth-prepass-style colour
    // write of nothing at all (all four false).
    template <typename Self>
    Self&& color_mask(this Self&& self, bool red, bool green, bool blue, bool alpha, int attachment = -1)
    {
        VkColorComponentFlags mask = 0;
        if (red)
            mask |= VK_COLOR_COMPONENT_R_BIT;
        if (green)
            mask |= VK_COLOR_COMPONENT_G_BIT;
        if (blue)
            mask |= VK_COLOR_COMPONENT_B_BIT;
        if (alpha)
            mask |= VK_COLOR_COMPONENT_A_BIT;

        if (attachment < 0)
        {
            self.blend_.write_mask = mask;
        }
        else
        {
            self.blend_overrides_[static_cast<std::uint32_t>(attachment)].write_mask = mask;
        }
        return std::forward<Self>(self);
    }

    // Clamp depth to the view volume instead of clipping the primitive. What a
    // shadow-map pass wants: geometry between the light and the near plane still
    // has to cast, and clipping it away is a hole in the shadow. Needs the
    // DEPTH_CLAMP feature, which is why the Feature existed with nothing using
    // it until now.
    template <typename Self>
    Self&& depth_clamp(this Self&& self, bool enable)
    {
        self.depth_clamp_ = enable;
        return std::forward<Self>(self);
    }

    // Turn a fragment's alpha into a coverage mask on an MSAA target: cutout
    // foliage and hair get antialiased edges from the same `discard`-free
    // shader, without sorting. Does nothing on a single-sample target, which is
    // Vulkan's rule and not ours.
    template <typename Self>
    Self&& alpha_to_coverage(this Self&& self, bool enable)
    {
        self.alpha_to_coverage_ = enable;
        return std::forward<Self>(self);
    }

    template <typename Self>
    Self&& polygon_mode(this Self&& self, PolygonMode mode)
    {
        self.polygon_mode_ = mode;
        return std::forward<Self>(self);
    }

    // Width in pixels of a LINE polygon mode or a LINE_LIST topology. Anything
    // other than 1.0 needs the WIDE_LINES feature — build() rejects it
    // otherwise — because a driver is free to support exactly one width.
    // A wireframe at 1.0 nearly disappears on a HiDPI display, which is what
    // this is for.
    template <typename Self>
    Self&& line_width(this Self&& self, float width)
    {
        self.line_width_ = width;
        return std::forward<Self>(self);
    }

    // Offset every depth value this pipeline writes. The fix for shadow acne:
    // a shadow map compared against itself self-shadows at grazing angles, and
    // pushing the depth away by a constant plus a slope-scaled term separates
    // the surface from its own shadow.
    //
    // slope scales with the polygon's depth gradient, which is what makes one
    // setting work at every angle. The bias clamp stays 0: a non-zero clamp is
    // the depthBiasClamp feature, and nothing here needs it.
    template <typename Self>
    Self&& depth_bias(this Self&& self, float constant, float slope = 0.0f)
    {
        self.depth_bias_constant_ = constant;
        self.depth_bias_slope_ = slope;
        return std::forward<Self>(self);
    }

    template <typename Self>
    Self&& topology(this Self&& self, Topology topology)
    {
        self.topology_ = topology;
        return std::forward<Self>(self);
    }

    // Per-sample fragment shading on an MSAA target: the fragment shader runs once
    // per sample instead of once per pixel, cleaning up interior/specular aliasing
    // that plain MSAA (edge coverage only) leaves behind. Needs the
    // SAMPLE_RATE_SHADING feature — build() rejects it otherwise. min_fraction
    // (0..1) is the minimum fraction of samples shaded uniquely.
    template <typename Self>
    Self&& sample_shading(this Self&& self, bool enable, float min_fraction = 1.0f)
    {
        self.sample_shading_ = enable;
        self.min_sample_shading_ = min_fraction;
        return std::forward<Self>(self);
    }

    // Debug name applied to the VkPipeline (validation diagnostics). No-op
    // without VK_EXT_debug_utils — see Context::set_debug_name.
    template <typename Self>
    Self&& name(this Self&& self, std::string name)
    {
        self.name_ = std::move(name);
        return std::forward<Self>(self);
    }

    template <typename Self>
    Self&& push_constant(this Self&& self, uint32_t size, ShaderStage stage)
    {
        self.layout_.add_push_constant(size, static_cast<VkShaderStageFlags>(to_vk(stage)));
        return std::forward<Self>(self);
    }

    // A specialization constant for one stage. It takes a stage for the same
    // reason every other graphics declarator does: each stage is a separate
    // SPIR-V module and a constant id means whatever that module says it means.
    // The same id may legitimately carry a different value in each stage.
    //
    // Keyed by stage rather than sorted into per-stage vectors by an if/else. The
    // old form tested FRAGMENT and sent everything else to the vertex list, so a
    // TESS_CONTROL constant would have been baked into the vertex shader — and a
    // switch would only have moved that bug into a default case. A map has no
    // illegal state: a bucket no module claims is simply never read.
    template <typename Self>
    Self&& constant(this Self&& self, std::uint32_t id, std::uint32_t bytes, ShaderStage stage)
    {
        self.constants_[stage].push_back({id, bytes});
        return std::forward<Self>(self);
    }

    template <typename Self>
    Self&& uniform_buffer(this Self&& self, uint32_t binding, ShaderStage stage, uint32_t set)
    {
        self.layout_.add_binding(
            binding, static_cast<VkShaderStageFlags>(to_vk(stage)), VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, set);
        return std::forward<Self>(self);
    }

    template <typename Self>
    Self&& storage_buffer(this Self&& self, uint32_t binding, ShaderStage stage, uint32_t set)
    {
        self.layout_.add_binding(
            binding, static_cast<VkShaderStageFlags>(to_vk(stage)), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, set);
        return std::forward<Self>(self);
    }

    template <typename Self>
    Self&& texture(this Self&& self, uint32_t binding, ShaderStage stage, uint32_t set)
    {
        self.layout_.add_binding(
            binding, static_cast<VkShaderStageFlags>(to_vk(stage)), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, set);
        return std::forward<Self>(self);
    }

    // A read/write image addressed by coordinate — the graphics counterpart of
    // the compute declarator that has existed since 0.9. A fragment shader that
    // does imageStore was unreachable without it, which is a different thing
    // from the automatic barriers it still does not get: the tracker cannot see
    // writes it has no reflection for, so a storage image written by a graphics
    // pipeline needs `cmd.barrier(image, ...)` exactly like an SSBO written the
    // same way (tech debt #3).
    template <typename Self>
    Self&& storage_image(this Self&& self, uint32_t binding, ShaderStage stage, uint32_t set)
    {
        self.layout_.add_binding(
            binding, static_cast<VkShaderStageFlags>(to_vk(stage)), VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, set);
        return std::forward<Self>(self);
    }

    // Build the pipeline with explicit color/depth formats (decoupled from renderer)
    //
    // A short sequence of named steps. The ~300-line monolith this replaces mixed
    // descriptor-layout creation, vertex-input translation and fixed state into
    // one scroll, with the cleanup loop copy-pasted into every failure branch.
    std::expected<std::shared_ptr<Pipeline>, Error> build(
        std::vector<VkFormat> colorFormats,
        VkFormat depthFormat,
        VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT,
        std::uint32_t view_mask = 0)
    {
        if (!vertex_shader_)
        {
            return std::unexpected(err_shader("A vertex shader must be provided"));
        }
        if (sample_shading_ && !context_.supports(Feature::SAMPLE_RATE_SHADING))
        {
            return std::unexpected(err_shader(
                "sample_shading requires the SAMPLE_RATE_SHADING feature. Create the "
                "Context with features=[bz.Feature.SAMPLE_RATE_SHADING] (or optional=[...])"));
        }
        // Anything but FILL is the fillModeNonSolid feature, which desktop
        // drivers all have and some mobile ones do not — so it goes through the
        // same negotiation as every other optional capability rather than
        // silently producing a driver-dependent pipeline.
        if (polygon_mode_ != PolygonMode::FILL && !context_.supports(Feature::WIREFRAME))
        {
            return std::unexpected(err_shader(
                "polygon_mode requires the WIREFRAME feature. Create the Context with "
                "features=[bz.Feature.WIREFRAME] (or optional=[...])"));
        }
        // Attachments that blend differently need independentBlend. It is one
        // more case of "core Vulkan" and "no feature bit" being different
        // claims: without it every element of pAttachments must be identical,
        // and the driver is entitled to reject the pipeline.
        if (!blend_overrides_.empty() && !context_.supports(Feature::INDEPENDENT_BLEND))
        {
            for (std::uint32_t i = 0; i < colorFormats.size(); ++i)
            {
                const auto it = blend_overrides_.find(i);
                if (it != blend_overrides_.end() && it->second.applied_to(blend_) != blend_)
                {
                    return std::unexpected(err_shader(
                        "blend(attachment=) / color_mask(attachment=) that differs from the "
                        "pipeline-wide setting requires the INDEPENDENT_BLEND feature. Create "
                        "the Context with features=[bz.Feature.INDEPENDENT_BLEND] (or "
                        "optional=[...])"));
                }
            }
        }
        // The pipeline reads the stencil's existence off its target, so asking
        // for the test against a target that has no stencil aspect is caught
        // here instead of at the first draw that quietly does nothing.
        if (stencil_.enable && !has_stencil(depthFormat))
        {
            return std::unexpected(err_shader(
                "stencil_test needs a target with a stencil attachment. Build the "
                "RenderTarget with depth=bz.Format.DEPTH_STENCIL"));
        }
        if (depth_clamp_ && !context_.supports(Feature::DEPTH_CLAMP))
        {
            return std::unexpected(err_shader(
                "depth_clamp requires the DEPTH_CLAMP feature. Create the Context with "
                "features=[bz.Feature.DEPTH_CLAMP] (or optional=[...])"));
        }
        if (line_width_ != 1.0f && !context_.supports(Feature::WIDE_LINES))
        {
            return std::unexpected(err_shader(
                "line_width other than 1.0 requires the WIDE_LINES feature. Create the "
                "Context with features=[bz.Feature.WIDE_LINES] (or optional=[...])"));
        }
        // A fragment shader is optional only when there is nothing to shade:
        // a depth-only pass (shadow maps) rasterizes straight into the depth
        // attachment and is valid Vulkan without one.
        if (!fragment_shader_ && !colorFormats.empty())
        {
            return std::unexpected(err_shader(
                "A fragment shader must be provided when the target has colour "
                "attachments (only depth-only targets can omit it)"));
        }
        // Tessellation is a PAIR of stages with a fixed-function tessellator
        // between them, so one without the other is not a partial pipeline, it is
        // an invalid one. Caught here rather than by the layers, because "which of
        // the two did I forget" is the useful half of the message.
        if (static_cast<bool>(tess_control_shader_) != static_cast<bool>(tess_evaluation_shader_))
        {
            return std::unexpected(err_shader(
                "tessellation needs BOTH stages: set tess_control_shader and "
                "tess_evaluation_shader together, or neither"));
        }
        // Not redundant with the same check inside compile_shader, and not a
        // second spelling of it either: this one catches a module compiled on a
        // Context that HAS the feature and then built into a pipeline on one that
        // does not. Both call the same function, so the two cannot disagree about
        // which feature a stage needs or about what the message says.
        for (const std::shared_ptr<ShaderModule>* slot :
             {&tess_control_shader_, &tess_evaluation_shader_, &geometry_shader_})
        {
            if (*slot)
            {
                if (auto e = ShaderCompiler::check_stage_supported(context_, (*slot)->stage()); !e)
                {
                    return std::unexpected(e.error());
                }
            }
        }
        // A graphics shader that WRITES a descriptor needs a feature bit, and which
        // one depends on its stage. Asked of the reflection rather than of the
        // declarators, so declaring a storage image and only reading it costs
        // nothing — the gate fires on what the shader does, not on what the pipeline
        // could do.
        //
        // written_bindings rather than writes(): a module whose scan came out
        // `writes_unknown` (foreign SPIR-V) claims to write everything, and demanding
        // the feature for every .spv shader would break callers who never write at
        // all. Such a module reaching this point behaves as it did before 0.19 — the
        // layers report it — which is the honest trade for a diagnostic.
        for (const std::shared_ptr<ShaderModule>* slot :
             {&vertex_shader_, &tess_control_shader_, &tess_evaluation_shader_, &geometry_shader_, &fragment_shader_})
        {
            if (!*slot || (*slot)->reflection().written_bindings.empty())
            {
                continue;
            }
            const bool fragment = (*slot)->stage() == ShaderStage::FRAGMENT;
            const Feature needed = fragment ? Feature::FRAGMENT_STORES : Feature::VERTEX_STAGE_STORES;
            if (!context_.supports(needed))
            {
                return std::unexpected(err_shader(
                    std::format(
                        "the {} shader writes a storage buffer or image, which requires the {} feature. "
                        "Create the Context with features=[bz.Feature.{}] (or optional=[...])",
                        ShaderCompiler::stage_name((*slot)->stage()),
                        feature_name(needed),
                        feature_name(needed))));
            }
        }
        // The two halves of one statement: a patch has no meaning without stages to
        // tessellate it, and a tessellation pipeline has nothing to read without
        // patches. Both spellings of the mistake get the same explanation.
        if (tess_control_shader_ && topology_ != Topology::PATCH_LIST)
        {
            return std::unexpected(err_shader(
                "a tessellation pipeline must draw patches. Add "
                "topology(bz.Topology.PATCH_LIST) and patch_control_points(n)"));
        }
        if (topology_ == Topology::PATCH_LIST && !tess_control_shader_)
        {
            return std::unexpected(err_shader(
                "Topology.PATCH_LIST is only valid with tessellation shaders. Set "
                "tess_control_shader and tess_evaluation_shader, or pick another topology"));
        }
        if (tess_control_shader_)
        {
            const std::uint32_t max_patch = context_.max_patch_control_points();
            if (patch_control_points_ == 0 || patch_control_points_ > max_patch)
            {
                return std::unexpected(err_shader(
                    std::format(
                        "patch_control_points must be between 1 and {} on this GPU, not {} — it is "
                        "how many vertices of the vertex buffer make one patch (3 for a triangle "
                        "patch, 4 for a quad)",
                        max_patch,
                        patch_control_points_)));
            }
        }

        // Descriptor set layouts and the pipeline layout are created ONCE and
        // reused across every hot-reload rebuild: they come from the builder's
        // binding/push-constant calls, not the shader source. Owned by guards
        // until the Pipeline exists.
        std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
        std::map<uint32_t, Pipeline::BindingTypeMap> allBindingTypes;
        ScopeGuard cleanup_layouts(
            [&]
            {
                for (auto dl : descriptorSetLayouts)
                {
                    context_.vk().vkDestroyDescriptorSetLayout(context_.device(), dl, nullptr);
                }
            });
        if (auto r = layout_.create_set_layouts(context_, descriptorSetLayouts, allBindingTypes); !r)
        {
            return std::unexpected(r.error());
        }

        auto layout = layout_.create_layout(context_, descriptorSetLayouts);
        if (!layout)
        {
            return std::unexpected(layout.error());
        }
        VkPipelineLayout pipelineLayout = layout.value();
        ScopeGuard cleanup_pipeline_layout(
            [&] { context_.vk().vkDestroyPipelineLayout(context_.device(), pipelineLayout, nullptr); });

        // The rebuildable slice of state: everything vkCreateGraphicsPipelines
        // needs except the layout. Copied into the recreate closure below so a
        // hot reload re-runs create_pipeline_ against fresh shader handles.
        GraphicsState state{
            .vertex = vertex_shader_,
            .fragment = fragment_shader_,
            .tess_control = tess_control_shader_,
            .tess_evaluation = tess_evaluation_shader_,
            .geometry = geometry_shader_,
            .formats = formats_,
            .instance_formats = instance_formats_,
            .constants = constants_,
            .depth_test = depth_test_,
            .depth_write = depth_write_,
            .depth_compare = depth_compare_,
            .stencil = stencil_,
            .cull_mode = cull_mode_,
            .front_face = front_face_,
            .polygon_mode = polygon_mode_,
            .depth_clamp = depth_clamp_,
            .line_width = line_width_,
            .depth_bias_constant = depth_bias_constant_,
            .depth_bias_slope = depth_bias_slope_,
            .blend = blend_,
            .blend_overrides = blend_overrides_,
            .alpha_to_coverage = alpha_to_coverage_,
            .topology = topology_,
            .patch_control_points = patch_control_points_,
            .color_formats = std::move(colorFormats),
            .depth_format = depthFormat,
            .samples = samples,
            .view_mask = view_mask,
            .sample_shading = sample_shading_,
            .min_sample_shading = min_sample_shading_};

        auto pipeline = create_pipeline_(context_, state, pipelineLayout);
        if (!pipeline)
        {
            return std::unexpected(pipeline.error());
        }
        if (!name_.empty())
        {
            context_.set_debug_name(VK_OBJECT_TYPE_PIPELINE, reinterpret_cast<std::uint64_t>(pipeline.value()), name_);
        }

        // Everything now belongs to the Pipeline.
        cleanup_layouts.release();
        cleanup_pipeline_layout.release();

        PipelineDesc desc;
        // Every stage the pipeline has, because Pipeline::uses() matches against
        // exactly this list: a module missing from it is a shader the watcher
        // recompiles and whose pipeline is then never rebuilt, so the edit appears
        // to do nothing. A loop rather than five ifs for that reason.
        for (const std::shared_ptr<ShaderModule>* slot :
             {&vertex_shader_, &tess_control_shader_, &tess_evaluation_shader_, &geometry_shader_, &fragment_shader_})
        {
            if (*slot)
            {
                desc.shaders.push_back(*slot);
            }
        }
        desc.recreate = [state = std::move(state), pipelineLayout](Context& c)
        { return create_pipeline_(c, state, pipelineLayout); };

        return std::make_shared<Pipeline>(
            context_.shared_from_this(),
            pipeline.value(),
            pipelineLayout,
            std::move(descriptorSetLayouts),
            std::move(allBindingTypes),
            layout_.push_constant_stages(),
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            std::move(desc));
    }

    // Convenience overload: a RenderTarget already knows its own formats, so the
    // caller shouldn't have to dig them out. This is what replaces build(renderer).
    std::expected<std::shared_ptr<Pipeline>, Error> build(const RenderTarget& target)
    {
        std::vector<VkFormat> colorFormats;
        colorFormats.reserve(target.color_count());
        for (std::uint32_t i = 0; i < target.color_count(); ++i)
        {
            colorFormats.push_back(target.color_format(i));
        }
        // The sample count and multiview mask come off the target too, so a
        // pipeline built for an MSAA or multiview target is automatically matched —
        // no separate knob. (A multiview pipeline's viewMask must equal the pass's.)
        return build(std::move(colorFormats), target.depth_format(), target.samples(), target.view_mask());
    }

private:
    // ── build() steps ─────────────────────────────────────────────────────────

    // The rebuildable fixed-function + shader state, minus the pipeline layout
    // (created once in build() and reused). Copied by value into the Pipeline's
    // recreate closure; the shader shared_ptrs are read via ->get() inside
    // create_pipeline_, which is exactly the hot-reload swap point.
    struct GraphicsState
    {
        std::shared_ptr<ShaderModule> vertex;
        std::shared_ptr<ShaderModule> fragment;
        std::shared_ptr<ShaderModule> tess_control;
        std::shared_ptr<ShaderModule> tess_evaluation;
        std::shared_ptr<ShaderModule> geometry;
        std::vector<VertexFormat> formats;
        std::vector<VertexFormat> instance_formats;
        // Part of the rebuildable state, not of the ShaderModule: a hot reload
        // recompiles the source and must re-apply the same values, or the
        // reloaded pipeline would quietly differ from the one it replaces.
        std::map<ShaderStage, std::vector<SpecConstant>> constants;
        bool depth_test = false;
        bool depth_write = true;
        CompareOp depth_compare = CompareOp::LESS_OR_EQUAL;
        StencilState stencil;
        CullMode cull_mode = CullMode::BACK;
        FrontFace front_face = FrontFace::COUNTER_CLOCKWISE;
        PolygonMode polygon_mode = PolygonMode::FILL;
        bool depth_clamp = false;
        float line_width = 1.0f;
        float depth_bias_constant = 0.0f;
        float depth_bias_slope = 0.0f;
        BlendState blend;
        std::map<std::uint32_t, BlendOverride> blend_overrides;
        bool alpha_to_coverage = false;
        Topology topology = Topology::TRIANGLE_LIST;
        std::uint32_t patch_control_points = 0;
        std::vector<VkFormat> color_formats;
        VkFormat depth_format = VK_FORMAT_UNDEFINED;
        VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
        std::uint32_t view_mask = 0;
        bool sample_shading = false;
        float min_sample_shading = 1.0f;
    };

    struct VertexInput
    {
        // Binding 0 is the per-vertex buffer, binding 1 the per-instance one.
        // Either may be absent (a shader that builds its geometry from
        // gl_VertexIndex declares no vertex attributes at all).
        std::vector<VkVertexInputBindingDescription> bindings;
        std::vector<VkVertexInputAttributeDescription> attributes;
    };

    // Assemble the VkPipeline from the rebuildable state plus a ready pipeline
    // layout. Static and reading only its arguments, so build() and rebuild()
    // share one code path. shader_stages_ reads s.vertex/s.fragment->get() here
    // — after a ShaderModule::replace() that returns the new handle.
    static std::expected<VkPipeline, Error> create_pipeline_(
        Context& context,
        const GraphicsState& s,
        VkPipelineLayout pipelineLayout)
    {
        std::vector<SpecializationBlock> specs;
        const std::vector<VkPipelineShaderStageCreateInfo> shaderStages = shader_stages_(s, specs);

        // Vertex input — the CreateInfo points into vertexInput, so it lives here.
        const VertexInput vertexInput = vertex_input_(s);
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        if (!vertexInput.attributes.empty())
        {
            vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInput.bindings.size());
            vertexInputInfo.pVertexBindingDescriptions = vertexInput.bindings.data();
            vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInput.attributes.size());
            vertexInputInfo.pVertexAttributeDescriptions = vertexInput.attributes.data();
        }

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .topology = to_vk(s.topology),
            .primitiveRestartEnable = VK_FALSE};

        // Only read when the pipeline has tessellation stages, so it costs a
        // struct on the stack and nothing else. patchControlPoints is the INPUT
        // patch size — how the vertex buffer groups into patches — and build()
        // has already validated it against maxTessellationPatchSize.
        const VkPipelineTessellationStateCreateInfo tessellationState{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .patchControlPoints = s.patch_control_points};

        // Dynamic State
        std::vector<VkDynamicState> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamicState{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data()};

        VkPipelineViewportStateCreateInfo viewportState{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .viewportCount = 1,
            .pViewports = nullptr,
            .scissorCount = 1,
            .pScissors = nullptr};

        const VkPipelineRasterizationStateCreateInfo rasterizer = rasterization_state_(s);

        // rasterizationSamples must match the sample count of the target this
        // pipeline draws into — build(target) reads it off the target so the two
        // never drift. sample_shading (per-sample fragment execution) is an opt-in
        // quality knob on top, gated on the SAMPLE_RATE_SHADING feature in build().
        VkPipelineMultisampleStateCreateInfo multisampling{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .rasterizationSamples = s.samples,
            .sampleShadingEnable = s.sample_shading ? VK_TRUE : VK_FALSE,
            .minSampleShading = s.min_sample_shading,
            .pSampleMask = nullptr,
            .alphaToCoverageEnable = s.alpha_to_coverage ? VK_TRUE : VK_FALSE,
            .alphaToOneEnable = VK_FALSE};

        // One blend state per colour attachment: the pipeline-wide one, with any
        // per-attachment override folded in.
        std::vector<VkPipelineColorBlendAttachmentState> blendAttachments;
        blendAttachments.reserve(s.color_formats.size());
        for (std::uint32_t i = 0; i < s.color_formats.size(); ++i)
        {
            const auto it = s.blend_overrides.find(i);
            blendAttachments.push_back(
                color_blend_attachment_(it == s.blend_overrides.end() ? s.blend : it->second.applied_to(s.blend)));
        }

        VkPipelineColorBlendStateCreateInfo colorBlending{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .logicOpEnable = VK_FALSE,
            .logicOp = VK_LOGIC_OP_COPY,
            .attachmentCount = static_cast<uint32_t>(blendAttachments.size()),
            .pAttachments = blendAttachments.data(),
            .blendConstants = {0.0f, 0.0f, 0.0f, 0.0f}};

        const VkPipelineDepthStencilStateCreateInfo depthStencil = depth_stencil_state_(s);

        // Dynamic Rendering Info
        VkPipelineRenderingCreateInfo pipelineRenderingCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
            .pNext = nullptr,
            // Multiview: must equal the viewMask of the pass this pipeline draws into
            // (build(target) reads it off the target — a MultiviewTarget lights one
            // bit per layer, everything else is 0).
            .viewMask = s.view_mask,
            .colorAttachmentCount = static_cast<uint32_t>(s.color_formats.size()),
            .pColorAttachmentFormats = s.color_formats.empty() ? nullptr : s.color_formats.data(),
            .depthAttachmentFormat = s.depth_format,
            // Derived from the depth format, never asked for: a target either has
            // a stencil aspect or it does not, and a pipeline that disagreed with
            // its target here would fail to render with no useful message.
            .stencilAttachmentFormat = has_stencil(s.depth_format) ? s.depth_format : VK_FORMAT_UNDEFINED};

        VkGraphicsPipelineCreateInfo pipelineInfo{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext = &pipelineRenderingCreateInfo,
            .flags = 0,
            .stageCount = static_cast<uint32_t>(shaderStages.size()),
            .pStages = shaderStages.data(),
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssembly,
            .pTessellationState = s.tess_control ? &tessellationState : nullptr,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depthStencil,
            .pColorBlendState = &colorBlending,
            .pDynamicState = &dynamicState,
            .layout = pipelineLayout,
            .renderPass = VK_NULL_HANDLE, // Dynamic rendering
            .subpass = 0,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1};

        VkPipeline graphicsPipeline;
        // ErrorCode::Shader, not Initialization: a pipeline that fails to build is
        // almost always a shader/state mismatch the caller can fix and retry, and
        // hot reload (0.8) depends on catching exactly this as recoverable.
        if (auto e = check(
                context.vk().vkCreateGraphicsPipelines(
                    context.device(), context.pipeline_cache(), 1, &pipelineInfo, nullptr, &graphicsPipeline),
                "create graphics pipeline",
                ErrorCode::Shader))
        {
            return std::unexpected(*e);
        }
        return graphicsPipeline;
    }

    // Every stage this pipeline has, walked in Vulkan's own pipeline order. One
    // (depth-only) to five. Vertex is mandatory; build() enforces which of the
    // others are legal together.
    //
    // The stage bit is to_vk(module->stage()) and no longer a literal per slot.
    // The literals were correct while there were exactly two slots and became a
    // liability at five: a module set on the wrong verb would have been announced
    // to Vulkan as the stage the slot expected rather than the stage it is.
    //
    // The specialization blocks are the caller's, not ours: their data has to stay
    // alive until vkCreateGraphicsPipelines has read it, and a local here would
    // dangle the moment this function returns.
    static std::vector<VkPipelineShaderStageCreateInfo> shader_stages_(
        const GraphicsState& s,
        std::vector<SpecializationBlock>& specs)
    {
        std::vector<VkPipelineShaderStageCreateInfo> stages;
        stages.reserve(5);
        // reserve() before the first emplace_back is load-bearing. Each block's
        // `info` points into that block's own entries/data vectors, and while a
        // vector move does carry those heap buffers along, not reallocating at all
        // costs one line and removes the question entirely.
        specs.reserve(5);

        for (const std::shared_ptr<ShaderModule>* slot :
             {&s.vertex, &s.tess_control, &s.tess_evaluation, &s.geometry, &s.fragment})
        {
            const std::shared_ptr<ShaderModule>& module = *slot;
            if (!module)
            {
                continue;
            }
            const auto it = s.constants.find(module->stage());
            specs.emplace_back(it != s.constants.end() ? it->second : std::vector<SpecConstant>{});
            stages.push_back(
                {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                 .pNext = nullptr,
                 .flags = 0,
                 .stage = to_vk(module->stage()),
                 .module = module->get(),
                 .pName = entry_name_(*module),
                 .pSpecializationInfo = specs.back().get()});
        }
        return stages;
    }

    // The SPIR-V keeps the entry point under the name it was compiled with, so a
    // module built with entry_point="VSMain" declares VSMain and a stage that
    // says "main" fails pipeline creation. The name lives on the module for this
    // reason as much as for the hot reload.
    //
    // The pointer stays valid because GraphicsState holds the module by
    // shared_ptr and create_pipeline_ consumes the stages before returning.
    static const char* entry_name_(const ShaderModule& module)
    {
        return module.entry_point().empty() ? "main" : module.entry_point().c_str();
    }

    // Translates the two VertexFormat lists into binding + attribute
    // descriptions with packed offsets. Both bindings are built by the same
    // loop: per-vertex and per-instance data differ by the input rate and by
    // nothing else, so a second copy of this would only be a second place for
    // the stride to be wrong.
    static VertexInput vertex_input_(const GraphicsState& s)
    {
        VertexInput result;
        std::uint32_t location = 0;

        const auto add_binding =
            [&](std::uint32_t binding, const std::vector<VertexFormat>& formats, VkVertexInputRate rate)
        {
            if (formats.empty())
            {
                return;
            }
            std::uint32_t offset = 0;
            for (const VertexFormat format : formats)
            {
                const VertexFormatInfo info = vertex_format_info(format);
                result.attributes.push_back(
                    {.location = location++, .binding = binding, .format = info.vk, .offset = offset});
                offset += info.size;
            }
            result.bindings.push_back({.binding = binding, .stride = offset, .inputRate = rate});
        };

        add_binding(0, s.formats, VK_VERTEX_INPUT_RATE_VERTEX);
        add_binding(1, s.instance_formats, VK_VERTEX_INPUT_RATE_INSTANCE);
        return result;
    }

    static VkPipelineRasterizationStateCreateInfo rasterization_state_(const GraphicsState& s)
    {
        VkCullModeFlags vkCullMode = VK_CULL_MODE_NONE;
        if (s.cull_mode == CullMode::BACK)
            vkCullMode = VK_CULL_MODE_BACK_BIT;
        else if (s.cull_mode == CullMode::FRONT)
            vkCullMode = VK_CULL_MODE_FRONT_BIT;
        else if (s.cull_mode == CullMode::FRONT_AND_BACK)
            vkCullMode = VK_CULL_MODE_FRONT_AND_BACK;

        return {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .depthClampEnable = s.depth_clamp ? VK_TRUE : VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = to_vk(s.polygon_mode),
            .cullMode = vkCullMode,
            .frontFace = s.front_face == FrontFace::CLOCKWISE ? VK_FRONT_FACE_CLOCKWISE
                                                              : VK_FRONT_FACE_COUNTER_CLOCKWISE,
            // Enabled by having a bias, so depth_bias(0) builds the same pipeline
            // as no call at all instead of a no-op the driver still honours.
            .depthBiasEnable = (s.depth_bias_constant != 0.0f || s.depth_bias_slope != 0.0f) ? VK_TRUE : VK_FALSE,
            .depthBiasConstantFactor = s.depth_bias_constant,
            .depthBiasClamp = 0.0f,
            .depthBiasSlopeFactor = s.depth_bias_slope,
            .lineWidth = s.line_width};
    }

    static VkPipelineColorBlendAttachmentState color_blend_attachment_(const BlendState& s)
    {
        // Blending off is ONE/ZERO — the source replaces the destination — so the
        // mode is read only when it is on.
        VkBlendFactor src_color = VK_BLEND_FACTOR_ONE;
        VkBlendFactor dst_color = VK_BLEND_FACTOR_ZERO;
        VkBlendFactor src_alpha = VK_BLEND_FACTOR_ONE;
        VkBlendFactor dst_alpha = VK_BLEND_FACTOR_ZERO;

        if (s.enable)
        {
            switch (s.mode)
            {
                case BlendMode::ALPHA:
                    src_color = VK_BLEND_FACTOR_SRC_ALPHA;
                    dst_color = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                    dst_alpha = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                    break;
                case BlendMode::ADDITIVE:
                    // Nothing scales down and nothing is subtracted, so draw
                    // order stops mattering — the point of additive.
                    src_color = VK_BLEND_FACTOR_ONE;
                    dst_color = VK_BLEND_FACTOR_ONE;
                    dst_alpha = VK_BLEND_FACTOR_ONE;
                    break;
                case BlendMode::PREMULTIPLIED:
                    // The colour already carries its alpha, so only the
                    // destination is attenuated.
                    src_color = VK_BLEND_FACTOR_ONE;
                    dst_color = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                    dst_alpha = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                    break;
            }
        }

        return {
            .blendEnable = s.enable ? VK_TRUE : VK_FALSE,
            .srcColorBlendFactor = src_color,
            .dstColorBlendFactor = dst_color,
            .colorBlendOp = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = src_alpha,
            .dstAlphaBlendFactor = dst_alpha,
            .alphaBlendOp = VK_BLEND_OP_ADD,
            .colorWriteMask = s.write_mask};
    }

    static VkPipelineDepthStencilStateCreateInfo depth_stencil_state_(const GraphicsState& s)
    {
        return {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .depthTestEnable = s.depth_test ? VK_TRUE : VK_FALSE,
            // Gated on depth_test as well: Vulkan lets a pipeline write depth
            // with the test off, and depth_test(False) has always meant "this
            // pass has nothing to do with depth".
            .depthWriteEnable = (s.depth_test && s.depth_write) ? VK_TRUE : VK_FALSE,
            .depthCompareOp = to_vk(s.depth_compare),
            .depthBoundsTestEnable = VK_FALSE,
            .stencilTestEnable = s.stencil.enable ? VK_TRUE : VK_FALSE,
            // Front and back carry the same state — see stencil_test().
            .front = s.stencil.to_vk_state(),
            .back = s.stencil.to_vk_state(),
            .minDepthBounds = 0.0f,
            .maxDepthBounds = 1.0f};
    }

    Context& context_;
    std::shared_ptr<ShaderModule> vertex_shader_;
    std::shared_ptr<ShaderModule> fragment_shader_;
    std::shared_ptr<ShaderModule> tess_control_shader_;
    std::shared_ptr<ShaderModule> tess_evaluation_shader_;
    std::shared_ptr<ShaderModule> geometry_shader_;
    std::vector<VertexFormat> formats_;
    std::vector<VertexFormat> instance_formats_;
    std::map<ShaderStage, std::vector<SpecConstant>> constants_;
    bool depth_test_ = false;
    bool depth_write_ = true;
    CompareOp depth_compare_ = CompareOp::LESS_OR_EQUAL;
    StencilState stencil_;
    CullMode cull_mode_ = CullMode::BACK;
    FrontFace front_face_ = FrontFace::COUNTER_CLOCKWISE;
    PolygonMode polygon_mode_ = PolygonMode::FILL;
    bool depth_clamp_ = false;
    float line_width_ = 1.0f;
    float depth_bias_constant_ = 0.0f;
    float depth_bias_slope_ = 0.0f;
    BlendState blend_;
    std::map<std::uint32_t, BlendOverride> blend_overrides_;
    bool alpha_to_coverage_ = false;
    Topology topology_ = Topology::TRIANGLE_LIST;
    std::uint32_t patch_control_points_ = 0;
    bool sample_shading_ = false;
    float min_sample_shading_ = 1.0f;
    std::string name_;
    PipelineLayoutBuilder layout_;
};

// Compute pipelines get their own builder instead of extra methods on the
// graphics one: a single builder where vertex_shader() and a compute shader
// coexist has illegal states, and the split is what lets storage_buffer()
// and push_constant() drop the stage argument — compute has exactly one stage.
class ComputePipelineBuilder
{
public:
    ComputePipelineBuilder(Context& context)
        : context_(context)
    {
    }

    Context& context()
    {
        return context_;
    }

    template <typename Self>
    Self&& shader(this Self&& self, std::shared_ptr<ShaderModule> shader)
    {
        self.shader_ = std::move(shader);
        return std::forward<Self>(self);
    }

    template <typename Self>
    Self&& uniform_buffer(this Self&& self, uint32_t binding, uint32_t set)
    {
        self.layout_.add_binding(binding, VK_SHADER_STAGE_COMPUTE_BIT, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, set);
        return std::forward<Self>(self);
    }

    template <typename Self>
    Self&& storage_buffer(this Self&& self, uint32_t binding, uint32_t set)
    {
        self.layout_.add_binding(binding, VK_SHADER_STAGE_COMPUTE_BIT, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, set);
        return std::forward<Self>(self);
    }

    // A read/write image the shader accesses by coordinate (imageLoad/imageStore).
    template <typename Self>
    Self&& storage_image(this Self&& self, uint32_t binding, uint32_t set)
    {
        self.layout_.add_binding(binding, VK_SHADER_STAGE_COMPUTE_BIT, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, set);
        return std::forward<Self>(self);
    }

    template <typename Self>
    Self&& push_constant(this Self&& self, uint32_t size)
    {
        self.layout_.add_push_constant(size, VK_SHADER_STAGE_COMPUTE_BIT);
        return std::forward<Self>(self);
    }

    // No stage argument, for the same reason nothing else here has one.
    template <typename Self>
    Self&& constant(this Self&& self, std::uint32_t id, std::uint32_t bytes)
    {
        self.constants_.push_back({id, bytes});
        return std::forward<Self>(self);
    }

    template <typename Self>
    Self&& name(this Self&& self, std::string name)
    {
        self.name_ = std::move(name);
        return std::forward<Self>(self);
    }

    // No target argument: compute has no attachments.
    std::expected<std::shared_ptr<Pipeline>, Error> build()
    {
        if (!shader_)
        {
            return std::unexpected(err_shader("A compute shader must be provided"));
        }

        // Layouts once, reused across hot-reload rebuilds (see graphics build()).
        std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
        std::map<uint32_t, Pipeline::BindingTypeMap> allBindingTypes;
        ScopeGuard cleanup_layouts(
            [&]
            {
                for (auto dl : descriptorSetLayouts)
                {
                    context_.vk().vkDestroyDescriptorSetLayout(context_.device(), dl, nullptr);
                }
            });
        if (auto r = layout_.create_set_layouts(context_, descriptorSetLayouts, allBindingTypes); !r)
        {
            return std::unexpected(r.error());
        }

        auto layout = layout_.create_layout(context_, descriptorSetLayouts);
        if (!layout)
        {
            return std::unexpected(layout.error());
        }
        VkPipelineLayout pipelineLayout = layout.value();
        ScopeGuard cleanup_pipeline_layout(
            [&] { context_.vk().vkDestroyPipelineLayout(context_.device(), pipelineLayout, nullptr); });

        auto pipeline = create_pipeline_(context_, shader_, pipelineLayout, constants_);
        if (!pipeline)
        {
            return std::unexpected(pipeline.error());
        }
        if (!name_.empty())
        {
            context_.set_debug_name(VK_OBJECT_TYPE_PIPELINE, reinterpret_cast<std::uint64_t>(pipeline.value()), name_);
        }

        cleanup_layouts.release();
        cleanup_pipeline_layout.release();

        PipelineDesc desc;
        desc.shaders.push_back(shader_);
        desc.recreate = [shader = shader_, pipelineLayout, constants = constants_](Context& c)
        { return create_pipeline_(c, shader, pipelineLayout, constants); };

        return std::make_shared<Pipeline>(
            context_.shared_from_this(),
            pipeline.value(),
            pipelineLayout,
            std::move(descriptorSetLayouts),
            std::move(allBindingTypes),
            layout_.push_constant_stages(),
            VK_PIPELINE_BIND_POINT_COMPUTE,
            std::move(desc));
    }

private:
    // Static so build() and rebuild() share it; reads shader->get() at call
    // time, the hot-reload swap point.
    static std::expected<VkPipeline, Error> create_pipeline_(
        Context& context,
        const std::shared_ptr<ShaderModule>& shader,
        VkPipelineLayout pipelineLayout,
        const std::vector<SpecConstant>& constants)
    {
        const SpecializationBlock spec(constants);
        VkComputePipelineCreateInfo pipelineInfo{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage =
                {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                 .pNext = nullptr,
                 .flags = 0,
                 .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                 .module = shader->get(),
                 // Same rule as the graphics stages: the name the module was
                 // compiled with, so an HLSL CSMain works here too.
                 .pName = shader->entry_point().empty() ? "main" : shader->entry_point().c_str(),
                 .pSpecializationInfo = spec.get()},
            .layout = pipelineLayout,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1};

        VkPipeline computePipeline;
        // ErrorCode::Shader for the same reason as graphics: a pipeline that
        // fails to build is a shader/state mismatch the caller can fix and
        // retry, and hot reload (0.8) depends on catching exactly that.
        if (auto e = check(
                context.vk().vkCreateComputePipelines(
                    context.device(), context.pipeline_cache(), 1, &pipelineInfo, nullptr, &computePipeline),
                "create compute pipeline",
                ErrorCode::Shader))
        {
            return std::unexpected(*e);
        }
        return computePipeline;
    }

    Context& context_;
    std::shared_ptr<ShaderModule> shader_;
    std::vector<SpecConstant> constants_;
    std::string name_;
    PipelineLayoutBuilder layout_;
};
