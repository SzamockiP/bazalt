#pragma once
#include <volk.h>

#include <cstdint>
#include <optional>

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
		case CompareOp::NEVER:            return VK_COMPARE_OP_NEVER;
		case CompareOp::LESS:             return VK_COMPARE_OP_LESS;
		case CompareOp::EQUAL:            return VK_COMPARE_OP_EQUAL;
		case CompareOp::LESS_OR_EQUAL:    return VK_COMPARE_OP_LESS_OR_EQUAL;
		case CompareOp::GREATER:          return VK_COMPARE_OP_GREATER;
		case CompareOp::NOT_EQUAL:        return VK_COMPARE_OP_NOT_EQUAL;
		case CompareOp::GREATER_OR_EQUAL: return VK_COMPARE_OP_GREATER_OR_EQUAL;
		case CompareOp::ALWAYS:           return VK_COMPARE_OP_ALWAYS;
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

	bool operator==(const SamplerDesc&) const = default;
};

// Cache key on the Context: the whole descriptor space fits in a handful of
// bits, so identical requests share one VkSampler. Texture used to create a
// fresh sampler per texture — pure waste of a pooled driver object.
// Compare bits (4-7) are strictly additive: every pre-compare desc keeps the
// exact key it had before.
constexpr std::uint32_t sampler_cache_key(const SamplerDesc& d)
{
	return static_cast<std::uint32_t>(d.filter)
	     | (static_cast<std::uint32_t>(d.address_mode) << 1)
	     | (d.anisotropy ? 1u << 3 : 0u)
	     | (d.compare ? (1u << 4) | (static_cast<std::uint32_t>(*d.compare) << 5) : 0u);
}

// A non-owning view of a cached VkSampler. The Context owns the handle and
// destroys it at teardown; Sampler objects handed to Python just keep the
// cache entry (and thus the Context) reachable.
class Sampler
{
public:
	Sampler(VkSampler handle, SamplerDesc desc) : handle_(handle), desc_(desc) {}

	VkSampler get() const { return handle_; }
	const SamplerDesc& desc() const { return desc_; }

private:
	VkSampler handle_ = VK_NULL_HANDLE;
	SamplerDesc desc_{};
};
