#pragma once
#include <volk.h>

#include <cstdint>
#include <cstring>
#include <optional>
#include <string>

// How to read texels. Deliberately small: these knobs cover every example and
// test; per-axis address modes can be added additively if something ever needs
// them.
enum class Filter
{
    LINEAR,
    NEAREST,
};

enum class AddressMode
{
    REPEAT,
    CLAMP,
    MIRROR,
    // Everything outside 0..1 reads the border colour instead of the edge texel.
    // What a shadow map needs: with CLAMP, geometry past the edge of the map
    // takes the depth of whatever happened to be on that edge, so the shadow
    // smears across the whole scene. A white border means "nothing occludes
    // here" and the smear disappears.
    CLAMP_TO_BORDER,
};

// The texel read outside a CLAMP_TO_BORDER sampler's range. Two values, because
// the third Vulkan option (transparent black) differs from opaque black only in
// alpha, and the shadow-map case is exactly the white/black question.
enum class BorderColor
{
    OPAQUE_BLACK,
    OPAQUE_WHITE,
};

// 1:1 with VkCompareOp. Used by compare samplers (sampler2DShadow); a full
// eight-value enum costs nothing and later lets the pipeline's depth test take
// a compare op additively.
enum class CompareOp
{
    NEVER,
    LESS,
    EQUAL,
    LESS_OR_EQUAL,
    GREATER,
    NOT_EQUAL,
    GREATER_OR_EQUAL,
    ALWAYS,
};

inline constexpr VkCompareOp to_vk(CompareOp op)
{
    switch (op)
    {
        case CompareOp::NEVER:
            return VK_COMPARE_OP_NEVER;
        case CompareOp::LESS:
            return VK_COMPARE_OP_LESS;
        case CompareOp::EQUAL:
            return VK_COMPARE_OP_EQUAL;
        case CompareOp::LESS_OR_EQUAL:
            return VK_COMPARE_OP_LESS_OR_EQUAL;
        case CompareOp::GREATER:
            return VK_COMPARE_OP_GREATER;
        case CompareOp::NOT_EQUAL:
            return VK_COMPARE_OP_NOT_EQUAL;
        case CompareOp::GREATER_OR_EQUAL:
            return VK_COMPARE_OP_GREATER_OR_EQUAL;
        case CompareOp::ALWAYS:
            return VK_COMPARE_OP_ALWAYS;
    }
    // Not std::unreachable(): pybind enums accept arbitrary ints.
    return VK_COMPARE_OP_ALWAYS;
}

struct SamplerDesc
{
    Filter filter = Filter::LINEAR;
    AddressMode address_mode = AddressMode::REPEAT;
    bool anisotropy = true;
    // Engaged = compare sampler (sampler2DShadow in GLSL): reads return the
    // comparison result, and LINEAR filtering becomes hardware PCF.
    std::optional<CompareOp> compare = std::nullopt;
    BorderColor border_color = BorderColor::OPAQUE_BLACK;
    // Added to the computed mip level: negative sharpens, positive blurs. A
    // cheap blur for a reflection or a downsample chain, and the fix for a
    // texture that samples one mip too soft under anisotropy.
    float mip_lod_bias = 0.0f;

    bool operator==(const SamplerDesc&) const = default;
};

// Cache key on the Context: the whole descriptor space fits in a handful of
// bits, so identical requests share one VkSampler. Texture used to create a
// fresh sampler per texture — pure waste of a pooled driver object.
// Compare bits (4-7) are strictly additive: every pre-compare desc keeps the
// exact key it had before.
//
// The bias is a float and cannot be packed into bits, so the key became a hash
// of the whole description rather than a bit field pretending to be one. The
// cache is a handful of entries either way, and a collision here would hand back
// a sampler with the wrong filtering — so the description is compared too, which
// is why the map now stores the desc beside the handle.
inline std::uint32_t sampler_cache_key(const SamplerDesc& d)
{
    const std::uint32_t bits = static_cast<std::uint32_t>(d.filter) |
                               (static_cast<std::uint32_t>(d.address_mode) << 1) | (d.anisotropy ? 1u << 4 : 0u) |
                               (d.compare ? (1u << 5) | (static_cast<std::uint32_t>(*d.compare) << 6) : 0u) |
                               (static_cast<std::uint32_t>(d.border_color) << 10);
    std::uint32_t bias_bits = 0;
    static_assert(sizeof(bias_bits) == sizeof(d.mip_lod_bias));
    std::memcpy(&bias_bits, &d.mip_lod_bias, sizeof(bias_bits));
    return bits ^ (bias_bits * 2654435761u);
}

// A non-owning view of a cached VkSampler. The Context owns the handle and
// destroys it at teardown; Sampler objects handed to Python just keep the
// cache entry (and thus the Context) reachable.
class Sampler
{
public:
    Sampler(VkSampler handle, SamplerDesc desc)
        : handle_(handle),
          desc_(desc)
    {
    }

    VkSampler get() const
    {
        return handle_;
    }
    const SamplerDesc& desc() const
    {
        return desc_;
    }

    // Debug names ACCUMULATE, because the object they name is shared: the cache
    // keys on the description, so create_sampler(name="shadow") and
    // create_sampler(name="terrain") with the same filtering are one VkSampler.
    //
    // The alternatives are both worse. Naming only the first caller silently
    // drops a name, which is confusing exactly when you are reading validation
    // output. Putting the name in the cache key gives a named sampler its own
    // handle, which makes a debug label change what the program allocates.
    // A list of every user is true, and it is what a validation message should
    // say. Returns false when the name adds nothing, so the caller can skip the
    // Vulkan call.
    bool add_debug_name(const std::string& name)
    {
        if (name.empty() || debug_name_.find(name) != std::string::npos)
        {
            return false;
        }
        debug_name_ = debug_name_.empty() ? name : debug_name_ + " + " + name;
        return true;
    }

    const std::string& debug_name() const
    {
        return debug_name_;
    }

private:
    VkSampler handle_ = VK_NULL_HANDLE;
    SamplerDesc desc_{};
    std::string debug_name_;
};
