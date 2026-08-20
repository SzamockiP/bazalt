#include "Pipeline.hpp"

#include <format>
#include <string>
#include <utility>

VkStencilOpState StencilState::to_vk_state() const
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

BlendState BlendOverride::applied_to(BlendState base) const
{
    base.enable = enable.value_or(base.enable);
    base.equation = equation.value_or(base.equation);
    base.write_mask = write_mask.value_or(base.write_mask);
    return base;
}

SpecializationBlock::SpecializationBlock(const std::vector<SpecConstant>& constants)
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

// ── Pipeline ──────────────────────────────────────────────────────────────────

Pipeline::Pipeline(
    std::shared_ptr<Context> context,
    VkPipeline pipeline,
    VkPipelineLayout layout,
    std::vector<VkDescriptorSetLayout> descLayouts,
    std::map<uint32_t, BindingTypeMap> bindingTypes,
    VkShaderStageFlags pushConstantStages,
    VkPipelineBindPoint bindPoint,
    PipelineDesc desc)
    : context_(std::move(context)),
      pipeline_(pipeline),
      layout_(layout),
      desc_layouts_(std::move(descLayouts)),
      binding_types_(std::move(bindingTypes)),
      push_constant_stages_(pushConstantStages),
      bind_point_(bindPoint),
      desc_(std::move(desc))
{
}

Pipeline::~Pipeline()
{
    destroy();
}

std::expected<void, Error> Pipeline::rebuild()
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

VkDescriptorSetLayout Pipeline::descriptor_set_layout(uint32_t setIndex) const
{
    if (setIndex < desc_layouts_.size())
    {
        return desc_layouts_[setIndex];
    }
    return VK_NULL_HANDLE;
}

const Pipeline::BindingTypeMap& Pipeline::binding_types(uint32_t setIndex) const
{
    static const BindingTypeMap empty;
    auto it = binding_types_.find(setIndex);
    if (it != binding_types_.end())
    {
        return it->second;
    }
    return empty;
}

void Pipeline::destroy()
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
            for (auto* dl : desc_layouts)
            {
                if (dl != VK_NULL_HANDLE)
                {
                    vk->vkDestroyDescriptorSetLayout(device, dl, nullptr);
                }
            }
        });
}

// ── PipelineLayoutBuilder ─────────────────────────────────────────────────────

void PipelineLayoutBuilder::add_binding(
    uint32_t binding,
    VkShaderStageFlags stageFlags,
    VkDescriptorType descriptorType,
    uint32_t setIndex,
    uint32_t count,
    std::optional<bool> update_after_bind)
{
    auto& bindings = descriptor_bindings_[setIndex];
    if (update_after_bind)
    {
        update_after_bind_[{setIndex, binding}] = *update_after_bind;
    }

    auto it = std::ranges::find(bindings, binding, &VkDescriptorSetLayoutBinding::binding);
    if (it != bindings.end())
    {
        // Declaring one binding twice is how a resource read by two stages is
        // spelled, so the stages merge. Two different counts are not that:
        // one of the two numbers is wrong and the layout can only hold one, so
        // it is a user error rather than something to merge. The builder verbs
        // chain and have no error channel, so the diagnosis waits for build().
        if (it->descriptorCount != count)
        {
            error_ = err_shader(
                std::format(
                    "binding {} of set {} is declared twice with different counts ({} and {}). "
                    "Declare it once per stage with the same count, or use two bindings.",
                    binding,
                    setIndex,
                    it->descriptorCount,
                    count));
        }
        it->stageFlags |= stageFlags;
        return;
    }

    bindings.push_back(
        {.binding = binding,
         .descriptorType = descriptorType,
         .descriptorCount = count,
         .stageFlags = stageFlags,
         .pImmutableSamplers = nullptr});
}

uint32_t PipelineLayoutBuilder::max_descriptor_count() const
{
    uint32_t largest = 1;
    for (const auto& [set, bindings] : descriptor_bindings_)
    {
        for (const auto& b : bindings)
        {
            largest = (std::ranges::max)(largest, b.descriptorCount);
        }
    }
    return largest;
}

std::expected<void, Error> PipelineLayoutBuilder::create_set_layouts(
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

        // Per-binding flags for the array bindings in this set, empty when it
        // has none. Must outlive vkCreateDescriptorSetLayout, hence this scope.
        std::vector<VkDescriptorBindingFlags> binding_flags;
        VkDescriptorSetLayoutBindingFlagsCreateInfo flags_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO, .pNext = nullptr};
        VkDescriptorSetLayoutCreateFlags layout_flags = 0;
        if (has_bindings)
        {
            binding_flags = binding_flags_for(context, s, it->second);
            if (std::ranges::any_of(binding_flags, [](auto f) { return f != 0; }))
            {
                flags_info.bindingCount = static_cast<uint32_t>(binding_flags.size());
                flags_info.pBindingFlags = binding_flags.data();
                layout_flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
            }
            else
            {
                binding_flags.clear();
            }
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = binding_flags.empty() ? nullptr : &flags_info,
            .flags = layout_flags,
            .bindingCount = has_bindings ? static_cast<uint32_t>(it->second.size()) : 0,
            .pBindings = has_bindings ? it->second.data() : nullptr};

        VkDescriptorSetLayout layout = nullptr;
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
                btm[b.binding] = {.type = b.descriptorType, .count = b.descriptorCount};
            }
            bindingTypes[s] = std::move(btm);
        }
    }
    return {};
}

std::expected<VkPipelineLayout, Error> PipelineLayoutBuilder::create_layout(
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

    VkPipelineLayout pipelineLayout = nullptr;
    if (auto e = check(
            context.vk().vkCreatePipelineLayout(context.device(), &pipelineLayoutInfo, nullptr, &pipelineLayout),
            "create pipeline layout"))
    {
        return std::unexpected(*e);
    }
    return pipelineLayout;
}

VkShaderStageFlags PipelineLayoutBuilder::push_constant_stages() const
{
    return std::ranges::fold_left(
        push_constant_ranges_ | std::views::transform(&VkPushConstantRange::stageFlags),
        VkShaderStageFlags{0},
        std::bit_or{});
}

// PARTIALLY_BOUND on every array: a slot nobody wrote is the normal case for a
// 500-texture array, and without the flag reading the SET at all is undefined
// rather than reading an unwritten slot. descriptorIndexing guarantees it.
//
// UPDATE_AFTER_BIND is the one the caller can name. The default is "an array
// yes, a single descriptor no", because that is what each is FOR: an array
// exists to be rewritten while the scene runs, and a plain binding is written
// once at setup in nearly every program. Both defaults can be wrong, so both
// are overridable — a static 500-texture atlas wants it off, and a single
// texture swapped between frames wants it on.
//
// Off is the cheaper side, which is why it is the default for the common case:
// an update-after-bind descriptor is counted against a separate limit
// (maxPerStageDescriptorUpdateAfterBind*, usually larger) and some
// implementations place it differently. Nothing here is measured, so the knob
// exists rather than a claim.
//
// The flag is only set where the device enabled the bit for that descriptor
// type. descriptorIndexing guarantees it for sampled images, storage images and
// storage buffers, but NOT for uniform buffers, so this reads what actually
// stuck rather than assuming. Without it the array is still usable; it just has
// to be written before the first frame that binds the set.
std::vector<VkDescriptorBindingFlags> PipelineLayoutBuilder::binding_flags_for(
    Context& context,
    uint32_t setIndex,
    const std::vector<VkDescriptorSetLayoutBinding>& bindings) const
{
    const VkPhysicalDeviceVulkan12Features& v12 = context.negotiated_features().v12;
    std::vector<VkDescriptorBindingFlags> flags;
    flags.reserve(bindings.size());
    for (const auto& b : bindings)
    {
        VkDescriptorBindingFlags f = 0;
        if (b.descriptorCount > 1)
        {
            f |= VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
        }

        bool wants = b.descriptorCount > 1;
        if (auto it = update_after_bind_.find({setIndex, b.binding}); it != update_after_bind_.end())
        {
            wants = it->second;
        }
        const bool device_has = (b.descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
                                 v12.descriptorBindingSampledImageUpdateAfterBind) ||
                                (b.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE &&
                                 v12.descriptorBindingStorageImageUpdateAfterBind) ||
                                (b.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER &&
                                 v12.descriptorBindingStorageBufferUpdateAfterBind) ||
                                (b.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER &&
                                 v12.descriptorBindingUniformBufferUpdateAfterBind);
        if (wants && device_has)
        {
            f |= VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
        }
        flags.push_back(f);
    }
    return flags;
}

std::expected<void, Error> check_descriptor_arrays(Context& context, const PipelineLayoutBuilder& layout)
{
    if (layout.error())
    {
        return std::unexpected(*layout.error());
    }
    if (layout.max_descriptor_count() > 1 && !context.supports(Feature::BINDLESS))
    {
        return std::unexpected(err_unsupported(
            "count > 1 on a binding declarator requires the BINDLESS feature. Create the "
            "Context with optional=[bz.Feature.BINDLESS] and check ctx.supports() before "
            "you declare the array."));
    }
    // Same gate, because the descriptorBinding*UpdateAfterBind bits are turned on
    // only for a Context that asked for BINDLESS. Silently ignoring the request
    // would leave a caller rewriting descriptors that a submit still reads, which
    // is the undefined behaviour they asked to be rid of.
    if (layout.wants_update_after_bind() && !context.supports(Feature::BINDLESS))
    {
        return std::unexpected(err_unsupported(
            "update_after_bind=True requires the BINDLESS feature. Create the Context with "
            "optional=[bz.Feature.BINDLESS], or write the descriptor before the first frame "
            "that binds the set."));
    }
    return {};
}

namespace
{

    // The identical front half of both build() methods: the descriptor set layouts
    // and the pipeline layout. Created once and reused across every hot-reload
    // rebuild, because they come from the builder's binding/push-constant calls,
    // not the shader source. The struct owns the handles — every failure branch
    // after create() unwinds them — until release() hands them to the Pipeline.
    struct PipelineLayouts
    {
        explicit PipelineLayouts(Context& context)
            : context_(context)
        {
        }

        PipelineLayouts(const PipelineLayouts&) = delete;
        PipelineLayouts& operator=(const PipelineLayouts&) = delete;

        ~PipelineLayouts()
        {
            if (released_)
            {
                return;
            }
            for (auto* dl : set_layouts)
            {
                context_.vk().vkDestroyDescriptorSetLayout(context_.device(), dl, nullptr);
            }
            if (pipeline_layout != VK_NULL_HANDLE)
            {
                context_.vk().vkDestroyPipelineLayout(context_.device(), pipeline_layout, nullptr);
            }
        }

        std::expected<void, Error> create(const PipelineLayoutBuilder& builder)
        {
            if (auto r = builder.create_set_layouts(context_, set_layouts, binding_types); !r)
            {
                return std::unexpected(r.error());
            }
            auto layout = builder.create_layout(context_, set_layouts);
            if (!layout)
            {
                return std::unexpected(layout.error());
            }
            pipeline_layout = layout.value();
            return {};
        }

        // Call once the handles have found their owner.
        void release()
        {
            released_ = true;
        }

        std::vector<VkDescriptorSetLayout> set_layouts;
        std::map<uint32_t, Pipeline::BindingTypeMap> binding_types;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;

    private:
        Context& context_;
        bool released_ = false;
    };

} // namespace

// ── GraphicsPipelineBuilder ───────────────────────────────────────────────────

GraphicsPipelineBuilder& GraphicsPipelineBuilder::stencil_test(
    bool enable,
    CompareOp compare,
    std::uint32_t ref,
    StencilOp pass_op,
    StencilOp fail_op,
    StencilOp depth_fail_op,
    std::uint32_t read_mask,
    std::uint32_t write_mask,
    Face face)
{
    stencil_.enable = enable;
    for (StencilState* side : {&stencil_, &stencil_back_})
    {
        const bool is_front = side == &stencil_;
        if (face == Face::FRONT && !is_front)
        {
            continue;
        }
        if (face == Face::BACK && is_front)
        {
            continue;
        }
        side->compare = compare;
        side->reference = ref;
        side->pass_op = pass_op;
        side->fail_op = fail_op;
        side->depth_fail_op = depth_fail_op;
        side->read_mask = read_mask;
        side->write_mask = write_mask;
    }
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::blend(bool enable, BlendEquation equation, int attachment)
{
    if (attachment < 0)
    {
        blend_.enable = enable;
        blend_.equation = equation;
    }
    else
    {
        auto& o = blend_overrides_[static_cast<std::uint32_t>(attachment)];
        o.enable = enable;
        o.equation = equation;
    }
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::color_mask(
    bool red,
    bool green,
    bool blue,
    bool alpha,
    int attachment)
{
    VkColorComponentFlags mask = 0;
    if (red)
    {
        mask |= VK_COLOR_COMPONENT_R_BIT;
    }
    if (green)
    {
        mask |= VK_COLOR_COMPONENT_G_BIT;
    }
    if (blue)
    {
        mask |= VK_COLOR_COMPONENT_B_BIT;
    }
    if (alpha)
    {
        mask |= VK_COLOR_COMPONENT_A_BIT;
    }

    if (attachment < 0)
    {
        blend_.write_mask = mask;
    }
    else
    {
        blend_overrides_[static_cast<std::uint32_t>(attachment)].write_mask = mask;
    }
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::uniform_buffer(
    uint32_t binding,
    ShaderStage stage,
    uint32_t set,
    uint32_t count,
    std::optional<bool> update_after_bind)
{
    layout_.add_binding(
        binding,
        static_cast<VkShaderStageFlags>(to_vk(stage)),
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        set,
        count,
        update_after_bind);
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::storage_buffer(
    uint32_t binding,
    ShaderStage stage,
    uint32_t set,
    uint32_t count,
    std::optional<bool> update_after_bind)
{
    layout_.add_binding(
        binding,
        static_cast<VkShaderStageFlags>(to_vk(stage)),
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        set,
        count,
        update_after_bind);
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::texture(
    uint32_t binding,
    ShaderStage stage,
    uint32_t set,
    uint32_t count,
    std::optional<bool> update_after_bind)
{
    layout_.add_binding(
        binding,
        static_cast<VkShaderStageFlags>(to_vk(stage)),
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        set,
        count,
        update_after_bind);
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::storage_image(
    uint32_t binding,
    ShaderStage stage,
    uint32_t set,
    uint32_t count,
    std::optional<bool> update_after_bind)
{
    layout_.add_binding(
        binding,
        static_cast<VkShaderStageFlags>(to_vk(stage)),
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        set,
        count,
        update_after_bind);
    return *this;
}

std::expected<std::shared_ptr<Pipeline>, Error> GraphicsPipelineBuilder::build(
    std::vector<VkFormat> colorFormats,
    VkFormat depthFormat,
    VkSampleCountFlagBits samples,
    std::uint32_t view_mask)
{
    if (!vertex_shader_)
    {
        return std::unexpected(err_shader("A vertex shader must be provided"));
    }
    if (auto e = check_descriptor_arrays(context_, layout_); !e)
    {
        return std::unexpected(e.error());
    }
    if (sample_shading_ && !context_.supports(Feature::SAMPLE_RATE_SHADING))
    {
        return std::unexpected(err_unsupported(
            "sample_shading requires the SAMPLE_RATE_SHADING feature. Create the "
            "Context with features=[bz.Feature.SAMPLE_RATE_SHADING] (or optional=[...])"));
    }
    // Anything but FILL is the fillModeNonSolid feature, which desktop
    // drivers all have and some mobile ones do not — so it goes through the
    // same negotiation as every other optional capability rather than
    // silently producing a driver-dependent pipeline.
    if (polygon_mode_ != PolygonMode::FILL && !context_.supports(Feature::WIREFRAME))
    {
        return std::unexpected(err_unsupported(
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
                return std::unexpected(err_unsupported(
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
        return std::unexpected(err_unsupported(
            "depth_clamp requires the DEPTH_CLAMP feature. Create the Context with "
            "features=[bz.Feature.DEPTH_CLAMP] (or optional=[...])"));
    }
    if (line_width_ != 1.0f && !context_.supports(Feature::WIDE_LINES))
    {
        return std::unexpected(err_unsupported(
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
            return std::unexpected(err_unsupported(
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
    // VUID-VkPipelineInputAssemblyStateCreateInfo-topology-06252: a restart
    // index only means anything to a topology that runs primitives together.
    // A fan counts — it ends at the sentinel and the next fan picks a new
    // centre — which is why the list here is not just the two strips.
    if (primitive_restart_ && topology_ != Topology::TRIANGLE_STRIP && topology_ != Topology::LINE_STRIP &&
        topology_ != Topology::TRIANGLE_FAN)
    {
        return std::unexpected(err_shader(
            "topology(..., restart=True) needs a strip or a fan. A restart index ends the "
            "primitive being run together, and a list has none to end — use "
            "Topology.TRIANGLE_STRIP, Topology.LINE_STRIP or Topology.TRIANGLE_FAN"));
    }
    // Metal has no triangle fan, so a portability driver may say no. Same
    // shape as every other portability gate: ask by capability, and name the
    // shape that works everywhere.
    if (topology_ == Topology::TRIANGLE_FAN && !context_.supports(Feature::TRIANGLE_FANS))
    {
        return std::unexpected(err_unsupported(
            "Topology.TRIANGLE_FAN needs the TRIANGLE_FANS feature, which this driver does not "
            "offer — Metal has no fan, so MoltenVK reports it missing. Ask "
            "ctx.supports(bz.Feature.TRIANGLE_FANS), and emit the same shape as an indexed "
            "Topology.TRIANGLE_LIST where it answers False."));
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

    PipelineLayouts layouts(context_);
    if (auto r = layouts.create(layout_); !r)
    {
        return std::unexpected(r.error());
    }

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
        .stencil_back = stencil_back_,
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
        .primitive_restart = primitive_restart_,
        .patch_control_points = patch_control_points_,
        .color_formats = std::move(colorFormats),
        .depth_format = depthFormat,
        .samples = samples,
        .view_mask = view_mask,
        .sample_shading = sample_shading_,
        .min_sample_shading = min_sample_shading_};

    auto pipeline = create_pipeline_(context_, state, layouts.pipeline_layout);
    if (!pipeline)
    {
        return std::unexpected(pipeline.error());
    }
    if (!name_.empty())
    {
        context_.set_debug_name(VK_OBJECT_TYPE_PIPELINE, reinterpret_cast<std::uint64_t>(pipeline.value()), name_);
    }

    // Everything now belongs to the Pipeline.
    layouts.release();

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
    desc.recreate = [state = std::move(state), pipelineLayout = layouts.pipeline_layout](Context& c)
    { return create_pipeline_(c, state, pipelineLayout); };

    return std::make_shared<Pipeline>(
        context_.shared_from_this(),
        pipeline.value(),
        layouts.pipeline_layout,
        std::move(layouts.set_layouts),
        std::move(layouts.binding_types),
        layout_.push_constant_stages(),
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        std::move(desc));
}

std::expected<std::shared_ptr<Pipeline>, Error> GraphicsPipelineBuilder::build(const RenderTarget& target)
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

// ── build() steps ─────────────────────────────────────────────────────────────

std::expected<VkPipeline, Error> GraphicsPipelineBuilder::create_pipeline_(
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
        .primitiveRestartEnable = s.primitive_restart ? VK_TRUE : VK_FALSE};

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

    VkPipeline graphicsPipeline = nullptr;
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

std::vector<VkPipelineShaderStageCreateInfo> GraphicsPipelineBuilder::shader_stages_(
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

const char* GraphicsPipelineBuilder::entry_name_(const ShaderModule& module)
{
    return module.entry_point().empty() ? "main" : module.entry_point().c_str();
}

GraphicsPipelineBuilder::VertexInput GraphicsPipelineBuilder::vertex_input_(const GraphicsState& s)
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

VkPipelineRasterizationStateCreateInfo GraphicsPipelineBuilder::rasterization_state_(const GraphicsState& s)
{
    VkCullModeFlags vkCullMode = VK_CULL_MODE_NONE;
    if (s.cull_mode == CullMode::BACK)
    {
        vkCullMode = VK_CULL_MODE_BACK_BIT;
    }
    else if (s.cull_mode == CullMode::FRONT)
    {
        vkCullMode = VK_CULL_MODE_FRONT_BIT;
    }
    else if (s.cull_mode == CullMode::FRONT_AND_BACK)
    {
        vkCullMode = VK_CULL_MODE_FRONT_AND_BACK;
    }

    return {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .depthClampEnable = s.depth_clamp ? VK_TRUE : VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = to_vk(s.polygon_mode),
        .cullMode = vkCullMode,
        .frontFace = s.front_face == FrontFace::CLOCKWISE ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE,
        // Enabled by having a bias, so depth_bias(0) builds the same pipeline
        // as no call at all instead of a no-op the driver still honours.
        .depthBiasEnable = (s.depth_bias_constant != 0.0f || s.depth_bias_slope != 0.0f) ? VK_TRUE : VK_FALSE,
        .depthBiasConstantFactor = s.depth_bias_constant,
        .depthBiasClamp = 0.0f,
        .depthBiasSlopeFactor = s.depth_bias_slope,
        .lineWidth = s.line_width};
}

VkPipelineColorBlendAttachmentState GraphicsPipelineBuilder::color_blend_attachment_(const BlendState& s)
{
    // Blending off is ONE/ZERO ADD — the source replaces the destination —
    // so the equation is read only when it is on. Vulkan ignores these
    // fields with blendEnable false; writing them anyway keeps
    // blend(False, ...) from depending on that.
    const BlendEquation eq = s.enable ? s.equation
                                      : BlendEquation{
                                            .src_color = BlendFactor::ONE,
                                            .dst_color = BlendFactor::ZERO,
                                            .color_op = BlendOp::ADD,
                                            .src_alpha = BlendFactor::ONE,
                                            .dst_alpha = BlendFactor::ZERO,
                                            .alpha_op = BlendOp::ADD};

    return {
        .blendEnable = s.enable ? VK_TRUE : VK_FALSE,
        .srcColorBlendFactor = to_vk(eq.src_color),
        .dstColorBlendFactor = to_vk(eq.dst_color),
        .colorBlendOp = to_vk(eq.color_op),
        .srcAlphaBlendFactor = to_vk(eq.src_alpha),
        .dstAlphaBlendFactor = to_vk(eq.dst_alpha),
        .alphaBlendOp = to_vk(eq.alpha_op),
        .colorWriteMask = s.write_mask};
}

VkPipelineDepthStencilStateCreateInfo GraphicsPipelineBuilder::depth_stencil_state_(const GraphicsState& s)
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
        .back = s.stencil_back.to_vk_state(),
        .minDepthBounds = 0.0f,
        .maxDepthBounds = 1.0f};
}

// ── ComputePipelineBuilder ────────────────────────────────────────────────────

ComputePipelineBuilder& ComputePipelineBuilder::uniform_buffer(
    uint32_t binding,
    uint32_t set,
    uint32_t count,
    std::optional<bool> update_after_bind)
{
    layout_.add_binding(
        binding, VK_SHADER_STAGE_COMPUTE_BIT, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, set, count, update_after_bind);
    return *this;
}

ComputePipelineBuilder& ComputePipelineBuilder::storage_buffer(
    uint32_t binding,
    uint32_t set,
    uint32_t count,
    std::optional<bool> update_after_bind)
{
    layout_.add_binding(
        binding, VK_SHADER_STAGE_COMPUTE_BIT, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, set, count, update_after_bind);
    return *this;
}

ComputePipelineBuilder& ComputePipelineBuilder::texture(
    uint32_t binding,
    uint32_t set,
    uint32_t count,
    std::optional<bool> update_after_bind)
{
    layout_.add_binding(
        binding, VK_SHADER_STAGE_COMPUTE_BIT, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, set, count, update_after_bind);
    return *this;
}

ComputePipelineBuilder& ComputePipelineBuilder::storage_image(
    uint32_t binding,
    uint32_t set,
    uint32_t count,
    std::optional<bool> update_after_bind)
{
    layout_.add_binding(
        binding, VK_SHADER_STAGE_COMPUTE_BIT, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, set, count, update_after_bind);
    return *this;
}

std::expected<std::shared_ptr<Pipeline>, Error> ComputePipelineBuilder::build()
{
    if (!shader_)
    {
        return std::unexpected(err_shader("A compute shader must be provided"));
    }
    if (auto e = check_descriptor_arrays(context_, layout_); !e)
    {
        return std::unexpected(e.error());
    }

    PipelineLayouts layouts(context_);
    if (auto r = layouts.create(layout_); !r)
    {
        return std::unexpected(r.error());
    }

    auto pipeline = create_pipeline_(context_, shader_, layouts.pipeline_layout, constants_);
    if (!pipeline)
    {
        return std::unexpected(pipeline.error());
    }
    if (!name_.empty())
    {
        context_.set_debug_name(VK_OBJECT_TYPE_PIPELINE, reinterpret_cast<std::uint64_t>(pipeline.value()), name_);
    }

    layouts.release();

    PipelineDesc desc;
    desc.shaders.push_back(shader_);
    desc.recreate = [shader = shader_, pipelineLayout = layouts.pipeline_layout, constants = constants_](Context& c)
    { return create_pipeline_(c, shader, pipelineLayout, constants); };

    return std::make_shared<Pipeline>(
        context_.shared_from_this(),
        pipeline.value(),
        layouts.pipeline_layout,
        std::move(layouts.set_layouts),
        std::move(layouts.binding_types),
        layout_.push_constant_stages(),
        VK_PIPELINE_BIND_POINT_COMPUTE,
        std::move(desc));
}

std::expected<VkPipeline, Error> ComputePipelineBuilder::create_pipeline_(
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

    VkPipeline computePipeline = nullptr;
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
