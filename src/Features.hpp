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
    INDEPENDENT_BLEND,     // independentBlend — a different blend state per MRT attachment
    TESSELLATION,          // tessellationShader — the TESS_CONTROL and TESS_EVALUATION stages
    GEOMETRY_SHADER,       // geometryShader — the GEOMETRY stage
    // Writing a storage buffer or storage image from a GRAPHICS shader. Two bits
    // rather than one because Vulkan splits them by stage, and the split is real:
    // a deferred pass writing from the fragment shader needs the first, a vertex
    // shader compacting geometry needs the second.
    //
    // These exist as of 0.19 because the graphics storage_image() declarator landed
    // in 0.17 without them, so a fragment imageStore worked on the GPU and failed
    // pipeline creation. Fourth time a capability that reads like plain command
    // recording turned out to have a feature bit, after fillModeNonSolid, wideLines
    // and independentBlend.
    FRAGMENT_STORES,     // fragmentStoresAndAtomics
    VERTEX_STAGE_STORES, // vertexPipelineStoresAndAtomics — vertex, tessellation, geometry
    // The three rows below live in a pNext struct rather than in
    // VkPhysicalDeviceFeatures, which is why the table has three columns as of
    // 0.21 and had one before.
    MULTIVIEW,          // VkPhysicalDeviceVulkan11Features::multiview — target.all_layers()
    BINDLESS,           // VkPhysicalDeviceVulkan12Features::descriptorIndexing — count= arrays
    DRAW_INDIRECT_COUNT // VkPhysicalDeviceVulkan12Features::drawIndirectCount — count_buffer=
};

// The feature structs bazalt reads, as one value with no pNext links between the
// members. The chain is built inside query_device_features and torn down again on
// the way out on purpose: a Device is copied into Python and a pNext pointing at
// the member of a temporary is a dangling pointer wearing a struct.
struct DeviceFeatures
{
    VkPhysicalDeviceFeatures core{};
    VkPhysicalDeviceVulkan11Features v11{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features v12{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
};

// One query for all three structs. Both callers used to build this chain by hand
// — Context::configure_features_ and list_devices — and two hand-built chains are
// two chances to ask a different question.
//
// The entry point is a parameter because the two callers reach it differently:
// the Context uses volk's instance-level global, and list_devices holds its own
// pointer from a throwaway instance it must not bind volk to.
inline DeviceFeatures query_device_features(
    PFN_vkGetPhysicalDeviceFeatures2 get_features2,
    VkPhysicalDevice physical_device)
{
    DeviceFeatures features;
    features.v11.pNext = &features.v12;
    VkPhysicalDeviceFeatures2 features2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &features.v11};
    get_features2(physical_device, &features2);
    features.core = features2.features;
    features.v11.pNext = nullptr;
    return features;
}

// A Feature is one boolean in one of the three structs above, so the table is a
// name plus three optional pointers-to-member of which exactly one is set.
//
// Three columns rather than a union or a variant: the table stays constexpr and
// stays an aggregate initializer, and the cost of a row that sets none is a
// feature that always reports false — loud the first time anyone asks for it.
// Capabilities that need an extension (ray tracing, mesh shaders) get a fourth
// column when there is API in bazalt to actually use them; advertising them
// before that would be a hollow promise.
struct FeatureInfo
{
    Feature feature;
    const char* name;
    VkBool32 VkPhysicalDeviceFeatures::* core = nullptr;
    VkBool32 VkPhysicalDeviceVulkan11Features::* v11 = nullptr;
    VkBool32 VkPhysicalDeviceVulkan12Features::* v12 = nullptr;
};

// std::to_array, not std::array<FeatureInfo, N>: the extent was a hardcoded 8 and
// adding a row meant editing a number in a second place, which is exactly the kind
// of edit a compiler should be doing. Deduced, it cannot go stale.
inline constexpr auto kFeatureTable = std::to_array<FeatureInfo>({
    {Feature::ANISOTROPIC_FILTERING, "ANISOTROPIC_FILTERING", &VkPhysicalDeviceFeatures::samplerAnisotropy},
    {Feature::WIREFRAME, "WIREFRAME", &VkPhysicalDeviceFeatures::fillModeNonSolid},
    {Feature::WIDE_LINES, "WIDE_LINES", &VkPhysicalDeviceFeatures::wideLines},
    {Feature::DEPTH_CLAMP, "DEPTH_CLAMP", &VkPhysicalDeviceFeatures::depthClamp},
    {Feature::SAMPLE_RATE_SHADING, "SAMPLE_RATE_SHADING", &VkPhysicalDeviceFeatures::sampleRateShading},
    {Feature::MULTI_DRAW_INDIRECT, "MULTI_DRAW_INDIRECT", &VkPhysicalDeviceFeatures::multiDrawIndirect},
    {Feature::SHADER_FLOAT64, "SHADER_FLOAT64", &VkPhysicalDeviceFeatures::shaderFloat64},
    {Feature::INDEPENDENT_BLEND, "INDEPENDENT_BLEND", &VkPhysicalDeviceFeatures::independentBlend},
    {Feature::TESSELLATION, "TESSELLATION", &VkPhysicalDeviceFeatures::tessellationShader},
    {Feature::GEOMETRY_SHADER, "GEOMETRY_SHADER", &VkPhysicalDeviceFeatures::geometryShader},
    {Feature::FRAGMENT_STORES, "FRAGMENT_STORES", &VkPhysicalDeviceFeatures::fragmentStoresAndAtomics},
    {Feature::VERTEX_STAGE_STORES, "VERTEX_STAGE_STORES", &VkPhysicalDeviceFeatures::vertexPipelineStoresAndAtomics},
    {.feature = Feature::MULTIVIEW, .name = "MULTIVIEW", .v11 = &VkPhysicalDeviceVulkan11Features::multiview},
    {.feature = Feature::BINDLESS,
     .name = "BINDLESS",
     .v12 = &VkPhysicalDeviceVulkan12Features::descriptorIndexing},
    {.feature = Feature::DRAW_INDIRECT_COUNT,
     .name = "DRAW_INDIRECT_COUNT",
     .v12 = &VkPhysicalDeviceVulkan12Features::drawIndirectCount},
});

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

inline constexpr bool feature_available(const DeviceFeatures& available, Feature feature)
{
    const FeatureInfo& info = feature_info(feature);
    if (info.core)
    {
        return available.core.*(info.core) == VK_TRUE;
    }
    if (info.v11)
    {
        return available.v11.*(info.v11) == VK_TRUE;
    }
    return info.v12 && available.v12.*(info.v12) == VK_TRUE;
}

inline constexpr void enable_feature(DeviceFeatures& features, Feature feature)
{
    const FeatureInfo& info = feature_info(feature);
    if (info.core)
    {
        features.core.*(info.core) = VK_TRUE;
    }
    else if (info.v11)
    {
        features.v11.*(info.v11) = VK_TRUE;
    }
    else if (info.v12)
    {
        features.v12.*(info.v12) = VK_TRUE;
    }
}

// The descriptor-indexing bits Feature::BINDLESS turns on, enabled one by one
// because the roll-up boolean does not do it: the spec says enabling
// descriptorIndexing "does not imply the other minimum descriptor indexing
// features are also enabled".
//
// Enable-what-is-present rather than a fixed list, for the same reason
// enable_features_if_present exists. descriptorIndexing guarantees most of these
// (partially bound, runtime arrays, and update-after-bind plus non-uniform
// indexing for sampled images and storage buffers), but NOT non-uniform indexing
// for uniform buffers and storage images, and not update-after-bind for uniform
// buffers. A device that has them gets them; one that does not still gets a
// working texture array, and the layout code asks which bits actually stuck
// rather than assuming.
inline constexpr auto kDescriptorIndexingBits = std::to_array<VkBool32 VkPhysicalDeviceVulkan12Features::*>({
    &VkPhysicalDeviceVulkan12Features::runtimeDescriptorArray,
    &VkPhysicalDeviceVulkan12Features::descriptorBindingPartiallyBound,
    &VkPhysicalDeviceVulkan12Features::descriptorBindingUpdateUnusedWhilePending,
    &VkPhysicalDeviceVulkan12Features::descriptorBindingSampledImageUpdateAfterBind,
    &VkPhysicalDeviceVulkan12Features::descriptorBindingStorageImageUpdateAfterBind,
    &VkPhysicalDeviceVulkan12Features::descriptorBindingStorageBufferUpdateAfterBind,
    &VkPhysicalDeviceVulkan12Features::descriptorBindingUniformBufferUpdateAfterBind,
    &VkPhysicalDeviceVulkan12Features::shaderSampledImageArrayNonUniformIndexing,
    &VkPhysicalDeviceVulkan12Features::shaderStorageImageArrayNonUniformIndexing,
    &VkPhysicalDeviceVulkan12Features::shaderStorageBufferArrayNonUniformIndexing,
    &VkPhysicalDeviceVulkan12Features::shaderUniformBufferArrayNonUniformIndexing,
});

inline constexpr void enable_descriptor_indexing(DeviceFeatures& enabled, const DeviceFeatures& available)
{
    for (auto bit : kDescriptorIndexingBits)
    {
        enabled.v12.*bit = available.v12.*bit;
    }
}

inline std::string api_version_string(std::uint32_t version)
{
    return std::format(
        "{}.{}.{}", VK_API_VERSION_MAJOR(version), VK_API_VERSION_MINOR(version), VK_API_VERSION_PATCH(version));
}
