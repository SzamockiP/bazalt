#pragma once
#include <volk.h>
#include <algorithm>
#include <functional>
#include <vector>
#include <memory>
#include <expected>
#include <array>
#include <map>
#include <ranges>
#include <unordered_map>
#include <utility>
#include "ShaderCompiler.hpp"
#include "Context.hpp"
#include "RenderTarget.hpp"
#include "ScopeGuard.hpp"

// Renamed from Format: this describes a vertex attribute, and `Format` is needed
// for pixel formats in 0.5.
enum class VertexFormat
{
    FLOAT2,
    FLOAT3,
    FLOAT4
};

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
    LINE_LIST
};

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
    }
    // Not std::unreachable(): pybind enums accept arbitrary ints.
    return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
}

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
            context_->defer_destroy([device = context_->device(), old = pipeline_]
                                    { vkDestroyPipeline(device, old, nullptr); });
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
            [device = context_->device(), pipeline = pipeline_, layout = layout_, desc_layouts = desc_layouts_]
            {
                if (pipeline != VK_NULL_HANDLE)
                {
                    vkDestroyPipeline(device, pipeline, nullptr);
                }
                if (layout != VK_NULL_HANDLE)
                {
                    vkDestroyPipelineLayout(device, layout, nullptr);
                }
                for (auto dl : desc_layouts)
                {
                    if (dl != VK_NULL_HANDLE)
                    {
                        vkDestroyDescriptorSetLayout(device, dl, nullptr);
                    }
                }
            });
    }

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
                    vkCreateDescriptorSetLayout(context.device(), &layoutInfo, nullptr, &layout),
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
                vkCreatePipelineLayout(context.device(), &pipelineLayoutInfo, nullptr, &pipelineLayout),
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

    template <typename Self>
    Self&& vertex_format(this Self&& self, const std::vector<VertexFormat>& formats)
    {
        self.formats_ = formats;
        return std::forward<Self>(self);
    }

    template <typename Self>
    Self&& depth_test(this Self&& self, bool enable)
    {
        self.depth_test_ = enable;
        return std::forward<Self>(self);
    }

    template <typename Self>
    Self&& cull_mode(this Self&& self, CullMode mode, FrontFace frontFace)
    {
        self.cull_mode_ = mode;
        self.front_face_ = frontFace;
        return std::forward<Self>(self);
    }

    template <typename Self>
    Self&& blend(this Self&& self, bool enable)
    {
        self.blend_enable_ = enable;
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

    // Build the pipeline with explicit color/depth formats (decoupled from renderer)
    //
    // A short sequence of named steps. The ~300-line monolith this replaces mixed
    // descriptor-layout creation, vertex-input translation and fixed state into
    // one scroll, with the cleanup loop copy-pasted into every failure branch.
    std::expected<std::shared_ptr<Pipeline>, Error> build(
        std::vector<VkFormat> colorFormats,
        VkFormat depthFormat,
        VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT)
    {
        if (!vertex_shader_)
        {
            return std::unexpected(err_shader("A vertex shader must be provided"));
        }
        if (sample_shading_ && !context_.supports(Feature::SAMPLE_RATE_SHADING))
        {
            return std::unexpected(err_shader(
                "sample_shading requires the SAMPLE_RATE_SHADING feature; create the "
                "Context with features=[bz.Feature.SAMPLE_RATE_SHADING] (or optional=[...])"));
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
                    vkDestroyDescriptorSetLayout(context_.device(), dl, nullptr);
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
        ScopeGuard cleanup_pipeline_layout([&]
                                           { vkDestroyPipelineLayout(context_.device(), pipelineLayout, nullptr); });

        // The rebuildable slice of state: everything vkCreateGraphicsPipelines
        // needs except the layout. Copied into the recreate closure below so a
        // hot reload re-runs create_pipeline_ against fresh shader handles.
        GraphicsState state{
            .vertex = vertex_shader_,
            .fragment = fragment_shader_,
            .formats = formats_,
            .depth_test = depth_test_,
            .cull_mode = cull_mode_,
            .front_face = front_face_,
            .blend_enable = blend_enable_,
            .topology = topology_,
            .color_formats = std::move(colorFormats),
            .depth_format = depthFormat,
            .samples = samples,
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
        desc.shaders.push_back(vertex_shader_);
        if (fragment_shader_)
        {
            desc.shaders.push_back(fragment_shader_);
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
        // The sample count comes off the target too, so a pipeline built for an
        // MSAA target is automatically multisample-matched — no separate knob.
        return build(std::move(colorFormats), target.depth_format(), target.samples());
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
        std::vector<VertexFormat> formats;
        bool depth_test = false;
        CullMode cull_mode = CullMode::BACK;
        FrontFace front_face = FrontFace::COUNTER_CLOCKWISE;
        bool blend_enable = false;
        Topology topology = Topology::TRIANGLE_LIST;
        std::vector<VkFormat> color_formats;
        VkFormat depth_format = VK_FORMAT_UNDEFINED;
        VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
        bool sample_shading = false;
        float min_sample_shading = 1.0f;
    };

    struct VertexInput
    {
        VkVertexInputBindingDescription binding{};
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
        const std::vector<VkPipelineShaderStageCreateInfo> shaderStages = shader_stages_(s);

        // Vertex input — the CreateInfo points into vertexInput, so it lives here.
        const VertexInput vertexInput = vertex_input_(s);
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        if (!vertexInput.attributes.empty())
        {
            vertexInputInfo.vertexBindingDescriptionCount = 1;
            vertexInputInfo.pVertexBindingDescriptions = &vertexInput.binding;
            vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInput.attributes.size());
            vertexInputInfo.pVertexAttributeDescriptions = vertexInput.attributes.data();
        }

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .topology = to_vk(s.topology),
            .primitiveRestartEnable = VK_FALSE};

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
            .alphaToCoverageEnable = VK_FALSE,
            .alphaToOneEnable = VK_FALSE};

        // One blend state per colour attachment (identical for now; per-target
        // blend control can arrive additively).
        const std::vector<VkPipelineColorBlendAttachmentState> blendAttachments(
            s.color_formats.size(), color_blend_attachment_(s));

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
            .viewMask = 0,
            .colorAttachmentCount = static_cast<uint32_t>(s.color_formats.size()),
            .pColorAttachmentFormats = s.color_formats.empty() ? nullptr : s.color_formats.data(),
            .depthAttachmentFormat = s.depth_format != VK_FORMAT_UNDEFINED ? s.depth_format : VK_FORMAT_UNDEFINED,
            .stencilAttachmentFormat = VK_FORMAT_UNDEFINED};

        VkGraphicsPipelineCreateInfo pipelineInfo{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext = &pipelineRenderingCreateInfo,
            .flags = 0,
            .stageCount = static_cast<uint32_t>(shaderStages.size()),
            .pStages = shaderStages.data(),
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssembly,
            .pTessellationState = nullptr,
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
                vkCreateGraphicsPipelines(
                    context.device(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline),
                "create graphics pipeline",
                ErrorCode::Shader))
        {
            return std::unexpected(*e);
        }
        return graphicsPipeline;
    }

    // 1 stage (depth-only) or 2. The fragment stage is optional exactly when
    // the target has no colour attachments — build() enforces that pairing.
    static std::vector<VkPipelineShaderStageCreateInfo> shader_stages_(const GraphicsState& s)
    {
        std::vector<VkPipelineShaderStageCreateInfo> stages;
        stages.push_back(
            {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
             .pNext = nullptr,
             .flags = 0,
             .stage = VK_SHADER_STAGE_VERTEX_BIT,
             .module = s.vertex->get(),
             .pName = "main",
             .pSpecializationInfo = nullptr});
        if (s.fragment)
        {
            stages.push_back(
                {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                 .pNext = nullptr,
                 .flags = 0,
                 .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                 .module = s.fragment->get(),
                 .pName = "main",
                 .pSpecializationInfo = nullptr});
        }
        return stages;
    }

    // Translates the VertexFormat list into a binding + attribute descriptions
    // with packed offsets.
    static VertexInput vertex_input_(const GraphicsState& s)
    {
        VertexInput result;
        result.binding = {
            .binding = 0,
            .stride = 0, // accumulated below
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX};

        result.attributes.resize(s.formats.size());
        uint32_t offset = 0;
        for (size_t i = 0; i < s.formats.size(); i++)
        {
            result.attributes[i].binding = 0;
            result.attributes[i].location = static_cast<uint32_t>(i);
            result.attributes[i].offset = offset;

            switch (s.formats[i])
            {
                case VertexFormat::FLOAT2:
                    result.attributes[i].format = VK_FORMAT_R32G32_SFLOAT;
                    offset += 8;
                    break;
                case VertexFormat::FLOAT3:
                    result.attributes[i].format = VK_FORMAT_R32G32B32_SFLOAT;
                    offset += 12;
                    break;
                case VertexFormat::FLOAT4:
                    result.attributes[i].format = VK_FORMAT_R32G32B32A32_SFLOAT;
                    offset += 16;
                    break;
            }
        }
        result.binding.stride = offset;
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
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = vkCullMode,
            .frontFace = s.front_face == FrontFace::CLOCKWISE ? VK_FRONT_FACE_CLOCKWISE
                                                              : VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .depthBiasEnable = VK_FALSE,
            .depthBiasConstantFactor = 0.0f,
            .depthBiasClamp = 0.0f,
            .depthBiasSlopeFactor = 0.0f,
            .lineWidth = 1.0f};
    }

    static VkPipelineColorBlendAttachmentState color_blend_attachment_(const GraphicsState& s)
    {
        return {
            .blendEnable = s.blend_enable ? VK_TRUE : VK_FALSE,
            .srcColorBlendFactor = s.blend_enable ? VK_BLEND_FACTOR_SRC_ALPHA : VK_BLEND_FACTOR_ONE,
            .dstColorBlendFactor = s.blend_enable ? VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA : VK_BLEND_FACTOR_ZERO,
            .colorBlendOp = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = s.blend_enable ? VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA : VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = VK_BLEND_OP_ADD,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                              VK_COLOR_COMPONENT_A_BIT};
    }

    static VkPipelineDepthStencilStateCreateInfo depth_stencil_state_(const GraphicsState& s)
    {
        return {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .depthTestEnable = s.depth_test ? VK_TRUE : VK_FALSE,
            .depthWriteEnable = s.depth_test ? VK_TRUE : VK_FALSE,
            .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
            .depthBoundsTestEnable = VK_FALSE,
            .stencilTestEnable = VK_FALSE,
            .front = {},
            .back = {},
            .minDepthBounds = 0.0f,
            .maxDepthBounds = 1.0f};
    }

    Context& context_;
    std::shared_ptr<ShaderModule> vertex_shader_;
    std::shared_ptr<ShaderModule> fragment_shader_;
    std::vector<VertexFormat> formats_;
    bool depth_test_ = false;
    CullMode cull_mode_ = CullMode::BACK;
    FrontFace front_face_ = FrontFace::COUNTER_CLOCKWISE;
    bool blend_enable_ = false;
    Topology topology_ = Topology::TRIANGLE_LIST;
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
    // Compute-only for now: fragment-shader storage images would need the tracker
    // to see writes it can't (no reflection), so they wait for a manual image
    // barrier — see the ResourceTracker ceiling note.
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
                    vkDestroyDescriptorSetLayout(context_.device(), dl, nullptr);
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
        ScopeGuard cleanup_pipeline_layout([&]
                                           { vkDestroyPipelineLayout(context_.device(), pipelineLayout, nullptr); });

        auto pipeline = create_pipeline_(context_, shader_, pipelineLayout);
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
        desc.recreate = [shader = shader_, pipelineLayout](Context& c)
        { return create_pipeline_(c, shader, pipelineLayout); };

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
        VkPipelineLayout pipelineLayout)
    {
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
                 .pName = "main",
                 .pSpecializationInfo = nullptr},
            .layout = pipelineLayout,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1};

        VkPipeline computePipeline;
        // ErrorCode::Shader for the same reason as graphics: a pipeline that
        // fails to build is a shader/state mismatch the caller can fix and
        // retry, and hot reload (0.8) depends on catching exactly that.
        if (auto e = check(
                vkCreateComputePipelines(context.device(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline),
                "create compute pipeline",
                ErrorCode::Shader))
        {
            return std::unexpected(*e);
        }
        return computePipeline;
    }

    Context& context_;
    std::shared_ptr<ShaderModule> shader_;
    std::string name_;
    PipelineLayoutBuilder layout_;
};
