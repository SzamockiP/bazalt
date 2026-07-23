#pragma once
#include <volk.h>

#include <algorithm>
#include <array>
#include <format>
#include <string>
#include <string_view>

// Optional GPU capabilities, addressed by what they *do* rather than by which
// Vulkan version or extension happens to spell them on a given driver.
//
// Why this exists: Vulkan promotes extensions into core versions, so the same
// capability has two spellings depending on the driver (VK_KHR_dynamic_rendering
// is an extension on 1.2 and core in 1.3). "Which version + which extensions" is
// therefore not a user's decision — it's an implementation detail this library
// resolves per device. Exposing it would force the user to know the Vulkan
// trivia the library exists to hide.
//
// Adding a capability later is an additive enum entry, never a breaking change,
// which is why there is no need to save new features for a 2.0.
enum class Feature
{
    ANISOTROPIC_FILTERING, // samplerAnisotropy — Texture uses this
    WIREFRAME,             // fillModeNonSolid
    WIDE_LINES,            // wideLines
    DEPTH_CLAMP,           // depthClamp
    SAMPLE_RATE_SHADING,   // sampleRateShading
    MULTI_DRAW_INDIRECT,   // multiDrawIndirect
    SHADER_FLOAT64,        // shaderFloat64
};

// Every Feature above maps to a plain VkPhysicalDeviceFeatures boolean, so the
// table is a name plus a pointer-to-member. Capabilities that need a version or
// an extension (ray tracing, mesh shaders) slot into this same table with extra
// columns when there is API in bazalt to actually use them — advertising them
// before that would be a hollow promise.
struct FeatureInfo
{
    Feature feature;
    const char* name;
    VkBool32 VkPhysicalDeviceFeatures::* bit;
};

inline constexpr std::array<FeatureInfo, 7> kFeatureTable{{
    {Feature::ANISOTROPIC_FILTERING, "ANISOTROPIC_FILTERING", &VkPhysicalDeviceFeatures::samplerAnisotropy},
    {Feature::WIREFRAME, "WIREFRAME", &VkPhysicalDeviceFeatures::fillModeNonSolid},
    {Feature::WIDE_LINES, "WIDE_LINES", &VkPhysicalDeviceFeatures::wideLines},
    {Feature::DEPTH_CLAMP, "DEPTH_CLAMP", &VkPhysicalDeviceFeatures::depthClamp},
    {Feature::SAMPLE_RATE_SHADING, "SAMPLE_RATE_SHADING", &VkPhysicalDeviceFeatures::sampleRateShading},
    {Feature::MULTI_DRAW_INDIRECT, "MULTI_DRAW_INDIRECT", &VkPhysicalDeviceFeatures::multiDrawIndirect},
    {Feature::SHADER_FLOAT64, "SHADER_FLOAT64", &VkPhysicalDeviceFeatures::shaderFloat64},
}};

inline constexpr const FeatureInfo& feature_info(Feature feature)
{
    auto it = std::ranges::find(kFeatureTable, feature, &FeatureInfo::feature);
    // The table covers the enum, but the input can be forged from Python
    // (pybind enums accept arbitrary ints), so a safe fallback beats
    // std::unreachable here.
    return it != kFeatureTable.end() ? *it : kFeatureTable[0];
}

inline constexpr std::string_view feature_name(Feature feature)
{
    return feature_info(feature).name;
}

inline constexpr bool feature_available(const VkPhysicalDeviceFeatures& available, Feature feature)
{
    return available.*(feature_info(feature).bit) == VK_TRUE;
}

inline constexpr void enable_feature(VkPhysicalDeviceFeatures& features, Feature feature)
{
    features.*(feature_info(feature).bit) = VK_TRUE;
}

inline std::string api_version_string(std::uint32_t version)
{
    return std::format(
        "{}.{}.{}", VK_API_VERSION_MAJOR(version), VK_API_VERSION_MINOR(version), VK_API_VERSION_PATCH(version));
}
