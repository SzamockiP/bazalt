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
    // Integer vectors (`in uvecN`, no conversion). UINT4 is what skinning joint
    // indices need — UBYTE4_NORM carried the weights and nothing carried the
    // indices (0.23). UBYTE4_UINT is the same four joints in a quarter the size.
    UINT2,
    UINT3,
    UINT4,
    UBYTE4_UINT,
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
        case VertexFormat::UINT2:
            return {VK_FORMAT_R32G32_UINT, 8};
        case VertexFormat::UINT3:
            return {VK_FORMAT_R32G32B32_UINT, 12};
        case VertexFormat::UINT4:
            return {VK_FORMAT_R32G32B32A32_UINT, 16};
        case VertexFormat::UBYTE4_UINT:
            return {VK_FORMAT_R8G8B8A8_UINT, 4};
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

// Which side of a triangle a stencil state applies to. Vulkan keeps two
// VkStencilOpStates and one enable bit, and this names the pair rather than
// splitting stencil_test into two verbs of eight parameters each.
enum class Face
{
    FRONT_AND_BACK,
    FRONT,
    BACK
};

enum class Topology
{
    TRIANGLE_LIST,
    POINT_LIST,
    LINE_LIST,
    // Strips: each new vertex extends the primitive instead of starting one, so
    // a quad is 4 vertices instead of 6. One strip per draw unless the caller
    // asks for restart (topology(..., restart=true)), which is opt-in because it
    // changes what an index buffer's largest value means.
    TRIANGLE_STRIP,
    LINE_STRIP,
    // Every triangle shares the FIRST vertex: 0,1,2 then 0,2,3 then 0,3,4. A
    // circle, a pie slice, a convex polygon.
    //
    // The one topology that is not universal. Metal has no fan at all, so a
    // portability driver may refuse it, and Feature::TRIANGLE_FANS is how you
    // ask. Where it answers False the same shape is an indexed TRIANGLE_LIST,
    // which is what a portable renderer should emit anyway.
    TRIANGLE_FAN,
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
    PREMULTIPLIED,
    // src * dst. Darkening overlays: ambient occlusion, baked shadows, tinted
    // glass. The one common mode the first three could not spell (0.23).
    MULTIPLY
};

// What each side of the blend is multiplied by. The named modes above are four
// points in this space; these are the axes, for the fifth thing somebody wants
// (0.23). 1:1 with VkBlendFactor, minus two families that need more API than a
// row: the constant-colour factors want a blend_constants() verb, and the SRC1_*
// ones want dual-source output declared in the shader.
enum class BlendFactor
{
    ZERO,
    ONE,
    SRC_COLOR,
    ONE_MINUS_SRC_COLOR,
    DST_COLOR,
    ONE_MINUS_DST_COLOR,
    SRC_ALPHA,
    ONE_MINUS_SRC_ALPHA,
    DST_ALPHA,
    ONE_MINUS_DST_ALPHA,
    // min(src.a, 1 - dst.a) on the colour channels, 1 on alpha. The
    // order-independent-ish additive trick.
    SRC_ALPHA_SATURATE
};

// How the two scaled sides are combined. ADD is what every named mode uses;
// MIN and MAX ignore the factors entirely and are what a depth-peel or a
// "keep the brightest" pass wants.
enum class BlendOp
{
    ADD,
    SUBTRACT,         // src - dst
    REVERSE_SUBTRACT, // dst - src
    MIN,
    MAX
};

inline constexpr VkBlendFactor to_vk(BlendFactor factor)
{
    switch (factor)
    {
        case BlendFactor::ZERO:
            return VK_BLEND_FACTOR_ZERO;
        case BlendFactor::ONE:
            return VK_BLEND_FACTOR_ONE;
        case BlendFactor::SRC_COLOR:
            return VK_BLEND_FACTOR_SRC_COLOR;
        case BlendFactor::ONE_MINUS_SRC_COLOR:
            return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case BlendFactor::DST_COLOR:
            return VK_BLEND_FACTOR_DST_COLOR;
        case BlendFactor::ONE_MINUS_DST_COLOR:
            return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        case BlendFactor::SRC_ALPHA:
            return VK_BLEND_FACTOR_SRC_ALPHA;
        case BlendFactor::ONE_MINUS_SRC_ALPHA:
            return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        case BlendFactor::DST_ALPHA:
            return VK_BLEND_FACTOR_DST_ALPHA;
        case BlendFactor::ONE_MINUS_DST_ALPHA:
            return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case BlendFactor::SRC_ALPHA_SATURATE:
            return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    }
    // Not std::unreachable(): a pybind enum accepts any int, so a forged value
    // has to land on something legal.
    return VK_BLEND_FACTOR_ONE;
}

inline constexpr VkBlendOp to_vk(BlendOp op)
{
    switch (op)
    {
        case BlendOp::ADD:
            return VK_BLEND_OP_ADD;
        case BlendOp::SUBTRACT:
            return VK_BLEND_OP_SUBTRACT;
        case BlendOp::REVERSE_SUBTRACT:
            return VK_BLEND_OP_REVERSE_SUBTRACT;
        case BlendOp::MIN:
            return VK_BLEND_OP_MIN;
        case BlendOp::MAX:
            return VK_BLEND_OP_MAX;
    }
    return VK_BLEND_OP_ADD;
}

// The whole blend equation, colour and alpha. A BlendMode resolves into one of
// these, and so do the factor arguments — so the pipeline stores an equation
// and nothing downstream has to know which spelling produced it.
struct BlendEquation
{
    BlendFactor src_color = BlendFactor::SRC_ALPHA;
    BlendFactor dst_color = BlendFactor::ONE_MINUS_SRC_ALPHA;
    BlendOp color_op = BlendOp::ADD;
    BlendFactor src_alpha = BlendFactor::ONE;
    BlendFactor dst_alpha = BlendFactor::ONE_MINUS_SRC_ALPHA;
    BlendOp alpha_op = BlendOp::ADD;

    bool operator==(const BlendEquation&) const = default;
};

// The four named modes, written out. This is the only place a preset means
// anything: past here everything is an equation.
inline constexpr BlendEquation blend_equation_for(BlendMode mode)
{
    switch (mode)
    {
        case BlendMode::ALPHA:
            return {
                BlendFactor::SRC_ALPHA,
                BlendFactor::ONE_MINUS_SRC_ALPHA,
                BlendOp::ADD,
                BlendFactor::ONE,
                BlendFactor::ONE_MINUS_SRC_ALPHA,
                BlendOp::ADD};
        case BlendMode::ADDITIVE:
            // Nothing scales down and nothing is subtracted, so draw order
            // stops mattering — the point of additive.
            return {BlendFactor::ONE, BlendFactor::ONE, BlendOp::ADD, BlendFactor::ONE, BlendFactor::ONE, BlendOp::ADD};
        case BlendMode::PREMULTIPLIED:
            // The colour already carries its alpha, so only the destination is
            // attenuated.
            return {
                BlendFactor::ONE,
                BlendFactor::ONE_MINUS_SRC_ALPHA,
                BlendOp::ADD,
                BlendFactor::ONE,
                BlendFactor::ONE_MINUS_SRC_ALPHA,
                BlendOp::ADD};
        case BlendMode::MULTIPLY:
            // dst * src + 0: the framebuffer is scaled by the fragment. White
            // leaves it alone, black removes it — an AO or shadow overlay.
            // Alpha keeps the destination's coverage.
            return {
                BlendFactor::DST_COLOR,
                BlendFactor::ZERO,
                BlendOp::ADD,
                BlendFactor::ZERO,
                BlendFactor::ONE,
                BlendOp::ADD};
    }
    return {};
}

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

// Which pixels a primitive produces fragments for (0.30). OFF is ordinary
// Vulkan: a pixel gets a fragment when the primitive covers its sample point,
// so a triangle that crosses a pixel without reaching the sample produces
// nothing there.
//
// The other two are the two directions to be wrong in on purpose:
// OVERESTIMATE covers every pixel the primitive TOUCHES, so nothing that the
// primitive reaches is ever missed and some pixels are shaded that a strict
// test would not; UNDERESTIMATE covers only the pixels the primitive contains
// ENTIRELY, so every fragment is a pixel fully inside and some are dropped.
// The first is for building a grid or a coverage mask with no holes, the
// second for an occlusion test that must not claim more than it can prove.
//
// Both need Feature::CONSERVATIVE_RASTER, and UNDERESTIMATE additionally needs
// a device whose primitiveUnderestimation property is true — build() refuses
// where it is not, because the layers do.
enum class ConservativeRaster
{
    OFF,
    OVERESTIMATE,
    UNDERESTIMATE
};

inline constexpr VkConservativeRasterizationModeEXT to_vk(ConservativeRaster mode)
{
    switch (mode)
    {
        case ConservativeRaster::OVERESTIMATE:
            return VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT;
        case ConservativeRaster::UNDERESTIMATE:
            return VK_CONSERVATIVE_RASTERIZATION_MODE_UNDERESTIMATE_EXT;
        case ConservativeRaster::OFF:
            return VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT;
    }
    return VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT;
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
        case Topology::TRIANGLE_FAN:
            return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
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

    VkStencilOpState to_vk_state() const;
};

// The colour-attachment state: how a fragment combines with what is already
// there, and which channels it may touch at all.
struct BlendState
{
    bool enable = false;
    BlendEquation equation = blend_equation_for(BlendMode::ALPHA);
    VkColorComponentFlags write_mask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                                       VK_COLOR_COMPONENT_A_BIT;

    bool operator==(const BlendState&) const = default;
};

// One attachment's deviation from the pipeline-wide BlendState. Every field is
// optional so that two calls naming the same attachment merge instead of
// overwriting each other. The equation is ONE field, not six: half an equation
// is not a state anybody can act on, and `blend()` always resolves a complete
// one from its own arguments.
struct BlendOverride
{
    std::optional<bool> enable;
    std::optional<BlendEquation> equation;
    std::optional<VkColorComponentFlags> write_mask;

    BlendState applied_to(BlendState base) const;
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

    explicit SpecializationBlock(const std::vector<SpecConstant>& constants);

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
    // What the layout declared for one binding. The type is what DescriptorSet
    // writes and what it refuses a mismatched setter for; the count is what it
    // bounds-checks index= against, since a descriptor array is one binding with
    // N elements and only the layout knows N.
    struct BindingInfo
    {
        VkDescriptorType type;
        std::uint32_t count;
    };

    // Maps binding index -> what was declared there
    using BindingTypeMap = std::unordered_map<uint32_t, BindingInfo>;

    Pipeline(
        std::shared_ptr<Context> context,
        VkPipeline pipeline,
        VkPipelineLayout layout,
        std::vector<VkDescriptorSetLayout> descLayouts = {},
        std::map<uint32_t, BindingTypeMap> bindingTypes = {},
        VkShaderStageFlags pushConstantStages = 0,
        VkPipelineBindPoint bindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        PipelineDesc desc = {});

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

    ~Pipeline();

    // Neither copyable nor movable, and the move half is a 0.27 deletion rather
    // than an omission: a Pipeline is always reached through the shared_ptr its
    // builder returns, so the move constructor and move assignment had no caller
    // in the whole tree. They also claimed noexcept while member-wise moving an
    // unordered_map, whose move the standard does not promise is noexcept —
    // which is how clang-tidy found them.
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    Pipeline(Pipeline&&) = delete;
    Pipeline& operator=(Pipeline&&) = delete;

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
    std::expected<void, Error> rebuild();

    VkDescriptorSetLayout descriptor_set_layout(uint32_t setIndex) const;

    const BindingTypeMap& binding_types(uint32_t setIndex) const;

private:
    // One teardown for the destructor and move-assignment; it used to be written
    // out twice and the two copies had already started to drift.
    // Deferred: an in-flight frame may still be executing with this pipeline
    // bound.
    void destroy();

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

class PipelineLayoutBuilder;

// The two things a build() has to say about count= before it creates anything:
// what a chaining declarator could not report, and whether this device can do
// descriptor arrays at all.
//
// count > 1 is refused without BINDLESS even though a fixed-size array indexed by
// a dynamically uniform expression is core Vulkan. That is deliberately stricter
// than the spec: without descriptorIndexing an unwritten slot is undefined, an
// index that differs per invocation is undefined, and a descriptor cannot be
// rewritten while an earlier frame reads it. Allowing the declaration anyway would
// mean a texture array that works here and returns garbage on the next machine,
// which is the failure mode this library exists to remove. The escape hatch is
// count=1 and one binding per texture, exactly as before.
std::expected<void, Error> check_descriptor_arrays(Context& context, const PipelineLayoutBuilder& layout);

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
        uint32_t setIndex,
        uint32_t count = 1,
        std::optional<bool> update_after_bind = std::nullopt);

    // The largest count any binding declared, so build() can gate arrays on
    // Feature.BINDLESS before it creates anything.
    uint32_t max_descriptor_count() const;

    // Whether anything asked for update-after-bind by name. Same gate as an
    // array: the bits behind it only exist on a Context that enabled BINDLESS.
    bool wants_update_after_bind() const
    {
        return std::ranges::any_of(update_after_bind_, [](const auto& e) { return e.second; });
    }

    const std::optional<Error>& error() const
    {
        return error_;
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
        std::map<uint32_t, Pipeline::BindingTypeMap>& bindingTypes) const;

    std::expected<VkPipelineLayout, Error> create_layout(
        Context& context,
        const std::vector<VkDescriptorSetLayout>& layouts) const;

    VkShaderStageFlags push_constant_stages() const;

private:
    // The flags each binding of one set needs, parallel to the set's binding list
    // because that is the shape VkDescriptorSetLayoutBindingFlagsCreateInfo takes.
    // Which flag lands on which binding, and why, is at the definition.
    std::vector<VkDescriptorBindingFlags> binding_flags_for(
        Context& context,
        uint32_t setIndex,
        const std::vector<VkDescriptorSetLayoutBinding>& bindings) const;

    std::vector<VkPushConstantRange> push_constant_ranges_;
    std::map<uint32_t, std::vector<VkDescriptorSetLayoutBinding>> descriptor_bindings_;
    // Only the bindings whose caller named it; everything else takes the default
    // in binding_flags_for. Keyed by (set, binding), because a binding index means
    // nothing without its set.
    std::map<std::pair<uint32_t, uint32_t>, bool> update_after_bind_;
    // A user error found by a chaining verb, carried to build(). See add_binding.
    std::optional<Error> error_;
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

    // Chained setters return *this: nothing ever chains on a temporary builder,
    // because the only way to reach one is ctx.graphics_pipeline() /
    // compute_pipeline(), which hand out shared_ptrs.

    GraphicsPipelineBuilder& vertex_shader(std::shared_ptr<ShaderModule> shader)
    {
        vertex_shader_ = std::move(shader);
        return *this;
    }

    GraphicsPipelineBuilder& fragment_shader(std::shared_ptr<ShaderModule> shader)
    {
        fragment_shader_ = std::move(shader);
        return *this;
    }

    // One verb per stage rather than a `shader(module, stage)` taking the stage as
    // an argument: a module already knows its own stage, so that argument could
    // disagree with it, and vertex_shader/fragment_shader set the pattern in 0.2.
    // The two tessellation stages are set together or not at all — see build().
    GraphicsPipelineBuilder& tess_control_shader(std::shared_ptr<ShaderModule> shader)
    {
        tess_control_shader_ = std::move(shader);
        return *this;
    }

    GraphicsPipelineBuilder& tess_evaluation_shader(std::shared_ptr<ShaderModule> shader)
    {
        tess_evaluation_shader_ = std::move(shader);
        return *this;
    }

    GraphicsPipelineBuilder& geometry_shader(std::shared_ptr<ShaderModule> shader)
    {
        geometry_shader_ = std::move(shader);
        return *this;
    }

    // How many vertices the vertex buffer groups into one patch — the INPUT size,
    // which is why it belongs on the pipeline and not in the shader. The control
    // shader's own `layout(vertices = N) out` is its OUTPUT count, a different
    // number, and nothing here can derive one from the other.
    GraphicsPipelineBuilder& patch_control_points(std::uint32_t count)
    {
        patch_control_points_ = count;
        return *this;
    }

    GraphicsPipelineBuilder& vertex_format(const std::vector<VertexFormat>& formats)
    {
        formats_ = formats;
        return *this;
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
    GraphicsPipelineBuilder& instance_format(const std::vector<VertexFormat>& formats)
    {
        instance_formats_ = formats;
        return *this;
    }

    // write=False keeps the test but stops the pass from updating the depth
    // buffer, which is the condition for a correct transparency pass: sorted
    // transparent geometry must test against the opaque depth without occluding
    // its own siblings. compare= replaces the LESS_OR_EQUAL that used to be
    // hard-coded — GREATER is a reversed-depth buffer, ALWAYS a full-screen pass.
    GraphicsPipelineBuilder& depth_test(bool enable, bool write = true, CompareOp compare = CompareOp::LESS_OR_EQUAL)
    {
        depth_test_ = enable;
        depth_write_ = write;
        depth_compare_ = compare;
        return *this;
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
    // Both faces share the state unless face= says otherwise (0.25, the upgrade
    // path the 0.17 entry named). Two calls spell a two-sided test — one per
    // face — which is what a shadow-volume pass wants: the same test increments
    // on one side and decrements on the other.
    //
    // `enable` is NOT per face. Vulkan has one stencilTestEnable for the
    // pipeline and two op-states, so any call sets the bit, and the last
    // `enable` wins. Pretending otherwise would invent a state the hardware does
    // not have.
    GraphicsPipelineBuilder& stencil_test(
        bool enable,
        CompareOp compare = CompareOp::ALWAYS,
        std::uint32_t ref = 0,
        StencilOp pass_op = StencilOp::KEEP,
        StencilOp fail_op = StencilOp::KEEP,
        StencilOp depth_fail_op = StencilOp::KEEP,
        std::uint32_t read_mask = 0xFF,
        std::uint32_t write_mask = 0xFF,
        Face face = Face::FRONT_AND_BACK);

    // front_face defaults to the value the builder already starts with, so
    // cull_mode(CullMode::NONE) is spellable. Culling nothing makes the winding
    // meaningless, and requiring an argument that means nothing is how a caller
    // ends up picking one at random and being wrong later — the 0.24 rule that a
    // default belongs on every call that asks for the value.
    GraphicsPipelineBuilder& cull_mode(CullMode mode, FrontFace frontFace = FrontFace::COUNTER_CLOCKWISE)
    {
        cull_mode_ = mode;
        front_face_ = frontFace;
        return *this;
    }

    // One verb for the question "how does this blend", whether the answer is a
    // named mode or a hand-written equation: the binding layer resolves both
    // spellings into a BlendEquation before it gets here, so there is one path
    // and no way for the two to disagree.
    //
    // attachment= narrows the answer to one colour attachment of an MRT target,
    // and everything without an override keeps what the plain call set. The
    // overrides are per FIELD (optional each), so blend(attachment=1) and
    // color_mask(attachment=1) compose in either order — a resolution that read
    // the default at call time would make the result depend on which line came
    // first, which is exactly the kind of rule nobody remembers at 3am.
    GraphicsPipelineBuilder& blend(
        bool enable,
        BlendEquation equation = blend_equation_for(BlendMode::ALPHA),
        int attachment = -1);

    // Which channels this pipeline writes. A g-buffer pass that must not touch
    // the alpha of an attachment it shares, or a depth-prepass-style colour
    // write of nothing at all (all four false).
    GraphicsPipelineBuilder& color_mask(bool red, bool green, bool blue, bool alpha, int attachment = -1);

    // Clamp depth to the view volume instead of clipping the primitive. What a
    // shadow-map pass wants: geometry between the light and the near plane still
    // has to cast, and clipping it away is a hole in the shadow. Needs the
    // DEPTH_CLAMP feature, which is why the Feature existed with nothing using
    // it until now.
    GraphicsPipelineBuilder& depth_clamp(bool enable)
    {
        depth_clamp_ = enable;
        return *this;
    }

    // Turn a fragment's alpha into a coverage mask on an MSAA target: cutout
    // foliage and hair get antialiased edges from the same `discard`-free
    // shader, without sorting. Does nothing on a single-sample target, which is
    // Vulkan's rule and not ours.
    GraphicsPipelineBuilder& alpha_to_coverage(bool enable)
    {
        alpha_to_coverage_ = enable;
        return *this;
    }

    GraphicsPipelineBuilder& polygon_mode(PolygonMode mode)
    {
        polygon_mode_ = mode;
        return *this;
    }

    // Width in pixels of a LINE polygon mode or a LINE_LIST topology. Anything
    // other than 1.0 needs the WIDE_LINES feature — build() rejects it
    // otherwise — because a driver is free to support exactly one width.
    // A wireframe at 1.0 nearly disappears on a HiDPI display, which is what
    // this is for.
    GraphicsPipelineBuilder& line_width(float width)
    {
        line_width_ = width;
        return *this;
    }

    // Offset every depth value this pipeline writes. The fix for shadow acne:
    // a shadow map compared against itself self-shadows at grazing angles, and
    // pushing the depth away by a constant plus a slope-scaled term separates
    // the surface from its own shadow.
    //
    // slope scales with the polygon's depth gradient, which is what makes one
    // setting work at every angle. The bias clamp stays 0: a non-zero clamp is
    // the depthBiasClamp feature, and nothing here needs it.
    GraphicsPipelineBuilder& depth_bias(float constant, float slope = 0.0f)
    {
        depth_bias_constant_ = constant;
        depth_bias_slope_ = slope;
        return *this;
    }

    // Rasterize by what the primitive touches rather than by where the samples
    // land — see ConservativeRaster for what each mode covers. Per pipeline,
    // never per device: Feature::CONSERVATIVE_RASTER only makes this call
    // legal, so every other pipeline on the same Context rasterizes normally
    // and pays nothing.
    //
    // extra_overestimation pushes the covered area further out, in pixels, on
    // top of what OVERESTIMATE already covers. It is the knob for a grid whose
    // cells must catch a surface that only grazes them. The driver holds it to
    // limits.max_extra_overestimation and rounds it down to a multiple of
    // limits.extra_overestimation_granularity, so build() refuses a value
    // above the maximum rather than letting it be silently clamped. It means
    // nothing to the other two modes and build() refuses it there too.
    GraphicsPipelineBuilder& conservative_raster(ConservativeRaster mode, float extra_overestimation = 0.0f)
    {
        conservative_raster_ = mode;
        extra_overestimation_ = extra_overestimation;
        return *this;
    }

    // restart= turns the largest representable index (0xFFFF or 0xFFFFFFFF, by
    // the index type) into "end this strip and start another", so one draw can
    // carry many strips. Opt-in rather than always on, because it takes that
    // value away from being an index: a mesh that happens to use it is silently
    // cut in two, and the caller who asked for restart is the one who knows.
    //
    // A kwarg on topology() rather than a verb of its own — it is meaningless
    // without a strip topology, and Vulkan rejects it on a list.
    GraphicsPipelineBuilder& topology(Topology topology, bool restart = false)
    {
        topology_ = topology;
        primitive_restart_ = restart;
        return *this;
    }

    // Per-sample fragment shading on an MSAA target: the fragment shader runs once
    // per sample instead of once per pixel, cleaning up interior/specular aliasing
    // that plain MSAA (edge coverage only) leaves behind. Needs the
    // SAMPLE_RATE_SHADING feature — build() rejects it otherwise. min_fraction
    // (0..1) is the minimum fraction of samples shaded uniquely.
    GraphicsPipelineBuilder& sample_shading(bool enable, float min_fraction = 1.0f)
    {
        sample_shading_ = enable;
        min_sample_shading_ = min_fraction;
        return *this;
    }

    // Debug name applied to the VkPipeline (validation diagnostics). No-op
    // without VK_EXT_debug_utils — see Context::set_debug_name.
    GraphicsPipelineBuilder& name(std::string name)
    {
        name_ = std::move(name);
        return *this;
    }

    GraphicsPipelineBuilder& push_constant(uint32_t size, ShaderStage stage)
    {
        layout_.add_push_constant(size, static_cast<VkShaderStageFlags>(to_vk(stage)));
        return *this;
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
    GraphicsPipelineBuilder& constant(std::uint32_t id, std::uint32_t bytes, ShaderStage stage)
    {
        constants_[stage].push_back({id, bytes});
        return *this;
    }

    GraphicsPipelineBuilder& uniform_buffer(
        uint32_t binding,
        ShaderStage stage,
        uint32_t set,
        uint32_t count,
        std::optional<bool> update_after_bind);

    GraphicsPipelineBuilder& storage_buffer(
        uint32_t binding,
        ShaderStage stage,
        uint32_t set,
        uint32_t count,
        std::optional<bool> update_after_bind);

    // count > 1 declares a descriptor ARRAY — one binding holding N textures, the
    // shader picking one per draw or per fragment. It is a kwarg rather than a
    // second declarator because a bindless texture and a plain one differ by how
    // many there are, which is the definition of a variant (rule 1).
    GraphicsPipelineBuilder& texture(
        uint32_t binding,
        ShaderStage stage,
        uint32_t set,
        uint32_t count,
        std::optional<bool> update_after_bind);

    // A read/write image addressed by coordinate — the graphics counterpart of
    // the compute declarator that has existed since 0.9. A fragment shader that
    // does imageStore was unreachable without it, which is a different thing
    // from the automatic barriers it still does not get: the tracker cannot see
    // writes it has no reflection for, so a storage image written by a graphics
    // pipeline needs `cmd.barrier(image, ...)` exactly like an SSBO written the
    // same way (tech debt #3).
    GraphicsPipelineBuilder& storage_image(
        uint32_t binding,
        ShaderStage stage,
        uint32_t set,
        uint32_t count,
        std::optional<bool> update_after_bind);

    // Build the pipeline with explicit color/depth formats (decoupled from renderer)
    //
    // A short sequence of named steps. The ~300-line monolith this replaces mixed
    // descriptor-layout creation, vertex-input translation and fixed state into
    // one scroll, with the cleanup loop copy-pasted into every failure branch.
    std::expected<std::shared_ptr<Pipeline>, Error> build(
        std::vector<VkFormat> colorFormats,
        VkFormat depthFormat,
        VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT,
        std::uint32_t view_mask = 0);

    // Convenience overload: a RenderTarget already knows its own formats, so the
    // caller shouldn't have to dig them out. This is what replaces build(renderer).
    std::expected<std::shared_ptr<Pipeline>, Error> build(const RenderTarget& target);

private:
    // ── build() steps ─────────────────────────────────────────────────────────
    //
    // The bodies live in Pipeline.cpp.

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
        StencilState stencil_back;
        CullMode cull_mode = CullMode::BACK;
        FrontFace front_face = FrontFace::COUNTER_CLOCKWISE;
        PolygonMode polygon_mode = PolygonMode::FILL;
        bool depth_clamp = false;
        ConservativeRaster conservative_raster = ConservativeRaster::OFF;
        float extra_overestimation = 0.0f;
        float line_width = 1.0f;
        float depth_bias_constant = 0.0f;
        float depth_bias_slope = 0.0f;
        BlendState blend;
        std::map<std::uint32_t, BlendOverride> blend_overrides;
        bool alpha_to_coverage = false;
        Topology topology = Topology::TRIANGLE_LIST;
        bool primitive_restart = false;
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
        VkPipelineLayout pipelineLayout);

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
        std::vector<SpecializationBlock>& specs);

    // The SPIR-V keeps the entry point under the name it was compiled with, so a
    // module built with entry_point="VSMain" declares VSMain and a stage that
    // says "main" fails pipeline creation. The name lives on the module for this
    // reason as much as for the hot reload.
    //
    // The pointer stays valid because GraphicsState holds the module by
    // shared_ptr and create_pipeline_ consumes the stages before returning.
    static const char* entry_name_(const ShaderModule& module);

    // Translates the two VertexFormat lists into binding + attribute
    // descriptions with packed offsets. Both bindings are built by the same
    // loop: per-vertex and per-instance data differ by the input rate and by
    // nothing else, so a second copy of this would only be a second place for
    // the stride to be wrong.
    static VertexInput vertex_input_(const GraphicsState& s);

    static VkPipelineRasterizationStateCreateInfo rasterization_state_(const GraphicsState& s);

    static VkPipelineColorBlendAttachmentState color_blend_attachment_(const BlendState& s);

    static VkPipelineDepthStencilStateCreateInfo depth_stencil_state_(const GraphicsState& s);

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
    // The back face's op-state. Kept beside the front one rather than inside it,
    // because the enable bit they share belongs to neither.
    StencilState stencil_back_;
    CullMode cull_mode_ = CullMode::BACK;
    FrontFace front_face_ = FrontFace::COUNTER_CLOCKWISE;
    PolygonMode polygon_mode_ = PolygonMode::FILL;
    bool depth_clamp_ = false;
    ConservativeRaster conservative_raster_ = ConservativeRaster::OFF;
    float extra_overestimation_ = 0.0f;
    float line_width_ = 1.0f;
    float depth_bias_constant_ = 0.0f;
    float depth_bias_slope_ = 0.0f;
    BlendState blend_;
    std::map<std::uint32_t, BlendOverride> blend_overrides_;
    bool alpha_to_coverage_ = false;
    Topology topology_ = Topology::TRIANGLE_LIST;
    bool primitive_restart_ = false;
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

    ComputePipelineBuilder& shader(std::shared_ptr<ShaderModule> shader)
    {
        shader_ = std::move(shader);
        return *this;
    }

    ComputePipelineBuilder& uniform_buffer(
        uint32_t binding,
        uint32_t set,
        uint32_t count,
        std::optional<bool> update_after_bind);

    // count > 1 is a descriptor array, same as on the graphics builder.
    ComputePipelineBuilder& storage_buffer(
        uint32_t binding,
        uint32_t set,
        uint32_t count,
        std::optional<bool> update_after_bind);

    // A sampled image in a compute shader. Missing until 0.21 for no reason
    // anyone wrote down: set_image, the descriptor pool and the barrier tracker
    // all handled a COMBINED_IMAGE_SAMPLER on a compute set already, so the gap
    // was the declaration and nothing else. Without it a compute shader could
    // only reach an image as a storage image -- no filtering, no mip selection,
    // no address mode.
    ComputePipelineBuilder& texture(
        uint32_t binding,
        uint32_t set,
        uint32_t count,
        std::optional<bool> update_after_bind);

    // A read/write image the shader accesses by coordinate (imageLoad/imageStore).
    ComputePipelineBuilder& storage_image(
        uint32_t binding,
        uint32_t set,
        uint32_t count,
        std::optional<bool> update_after_bind);

    ComputePipelineBuilder& push_constant(uint32_t size)
    {
        layout_.add_push_constant(size, VK_SHADER_STAGE_COMPUTE_BIT);
        return *this;
    }

    // No stage argument, for the same reason nothing else here has one.
    ComputePipelineBuilder& constant(std::uint32_t id, std::uint32_t bytes)
    {
        constants_.push_back({id, bytes});
        return *this;
    }

    ComputePipelineBuilder& name(std::string name)
    {
        name_ = std::move(name);
        return *this;
    }

    // No target argument: compute has no attachments.
    std::expected<std::shared_ptr<Pipeline>, Error> build();

private:
    // Static so build() and rebuild() share it; reads shader->get() at call
    // time, the hot-reload swap point.
    static std::expected<VkPipeline, Error> create_pipeline_(
        Context& context,
        const std::shared_ptr<ShaderModule>& shader,
        VkPipelineLayout pipelineLayout,
        const std::vector<SpecConstant>& constants);

    Context& context_;
    std::shared_ptr<ShaderModule> shader_;
    std::vector<SpecConstant> constants_;
    std::string name_;
    PipelineLayoutBuilder layout_;
};
