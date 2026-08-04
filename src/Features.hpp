#pragma once
#include <volk.h>

#include <algorithm>
#include <array>
#include <format>
#include <string>
#include <string_view>
#include <vector>

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
    // occlusionQueryPrecise — an occlusion query counts SAMPLES rather than
    // answering "something passed". Without it the spec allows any non-zero
    // value, so `query.samples` would mean two things depending on the driver:
    // this row is what lets a caller ask which one they are getting.
    PRECISE_OCCLUSION,
    // The three rows below live in a pNext struct rather than in
    // VkPhysicalDeviceFeatures, which is why the table has three columns as of
    // 0.21 and had one before.
    MULTIVIEW,           // VkPhysicalDeviceVulkan11Features::multiview — target.all_layers()
    BINDLESS,            // VkPhysicalDeviceVulkan12Features::descriptorIndexing — count= arrays
    DRAW_INDIRECT_COUNT, // VkPhysicalDeviceVulkan12Features::drawIndirectCount — count_buffer=
    // The three rows below are the OPPOSITE shape of every row above: they name
    // things full Vulkan always has and a portability driver may not. Metal has
    // no Vulkan of its own, so MoltenVK reports VK_KHR_portability_subset and
    // every restriction it applies. A device without that extension has no
    // restrictions, so these read TRUE there by construction (0.22).
    COMPARISON_SAMPLER,   // mutableComparisonSamplers — create_sampler(compare=)
    SAMPLER_MIP_LOD_BIAS, // samplerMipLodBias — create_sampler(mip_lod_bias=)
    MULTISAMPLE_ARRAYS,   // multisampleArrayImage — samples > 1 together with layers > 1
    IMAGE_VIEW_2D_ON_3D,  // imageView2DOn3DImage — target.layer(z) on a 3D image
    TRIANGLE_FANS,        // triangleFans — Topology::TRIANGLE_FAN
    // The first capability that is an EXTENSION rather than a feature bit
    // (0.25). VK_EXT_full_screen_exclusive is Win32-only in practice, so this
    // answers False everywhere else — which is what the row is for.
    EXCLUSIVE_FULLSCREEN,
    // A shader reading a buffer through its ADDRESS instead of a descriptor
    // (0.26). The reason this exists is a limit rather than a convenience:
    // maxStorageBufferRange caps what one descriptor may see at 4 GiB on the
    // desktop drivers that report the most, and an address is not a descriptor,
    // so it has no such cap. Core in Vulkan 1.2, so it needs no extension path.
    BUFFER_ADDRESS, // VkPhysicalDeviceVulkan12Features::bufferDeviceAddress
    // 64-bit integers in a shader (0.26). Its own row rather than a silent part
    // of BUFFER_ADDRESS: address arithmetic in GLSL (uint64_t casts,
    // `buffer_reference` maths) needs it, plain indexing does not.
    SHADER_INT64, // shaderInt64
    // A workgroup size the pipeline decides rather than the shader text (0.26):
    // `layout(local_size_x_id = 0)` plus .constant(). The capability is
    // maintenance4, because that is what makes the OpExecutionMode LocalSizeId
    // glslang emits legal. Core in 1.3, VK_KHR_maintenance4 on the 1.2 path —
    // the first row that needs both spellings, hence the sixth column below.
    WORKGROUP_SIZE
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

    // Only meaningful when the device reports VK_KHR_portability_subset, and the
    // flag says whether it did. Reading the struct on a full Vulkan driver would
    // report every restriction as ON, which is backwards — see feature_available.
    VkPhysicalDevicePortabilitySubsetFeaturesKHR portability{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_FEATURES_KHR};
    bool portability_subset = false;

    // Promoted into Vulkan 1.3 and available on 1.2 as VK_KHR_maintenance4, so
    // unlike the three above it is legal to ask for on two different grounds.
    // The flag records whether either held: a struct chained onto a device that
    // has neither is a question the driver never agreed to answer (0.26).
    VkPhysicalDeviceMaintenance4Features maintenance4{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES};
    bool maintenance4_queryable = false;

    // Which device extensions this GPU offers, for the rows whose capability is
    // an extension rather than a feature bit (0.25). A vector of names rather
    // than a bool per extension: the table below is the list of the ones bazalt
    // cares about, and a second list here would be a place for the two to
    // disagree.
    std::vector<std::string> extensions;
};

// One query for all three structs. Both callers used to build this chain by hand
// — Context::configure_features_ and list_devices — and two hand-built chains are
// two chances to ask a different question.
//
// The entry point is a parameter because the two callers reach it differently:
// the Context uses volk's instance-level global, and list_devices holds its own
// pointer from a throwaway instance it must not bind volk to.
// portability_subset must come from the caller: a pNext struct for an extension
// the device does not have is not a question the driver has to answer, and
// asking anyway is how you read a garbage struct as "every restriction applies".
inline DeviceFeatures query_device_features(
    PFN_vkGetPhysicalDeviceFeatures2 get_features2,
    VkPhysicalDevice physical_device,
    bool portability_subset = false,
    std::uint32_t api_version = VK_API_VERSION_1_2)
{
    DeviceFeatures features;
    features.portability_subset = portability_subset;

    // Before the feature query since 0.26, not after: which pNext structs are
    // legal to chain is partly an extension question, so the list has to exist
    // first. Nothing else about the order matters.
    std::uint32_t extension_count = 0;
    if (vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr) == VK_SUCCESS &&
        extension_count > 0)
    {
        std::vector<VkExtensionProperties> properties(extension_count);
        if (vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, properties.data()) ==
            VK_SUCCESS)
        {
            features.extensions.reserve(properties.size());
            for (const auto& property : properties)
            {
                features.extensions.emplace_back(property.extensionName);
            }
        }
    }

    features.maintenance4_queryable =
        api_version >= VK_API_VERSION_1_3 ||
        std::ranges::find(features.extensions, std::string_view(VK_KHR_MAINTENANCE_4_EXTENSION_NAME)) !=
            features.extensions.end();

    features.v11.pNext = &features.v12;
    void** tail = &features.v12.pNext;
    if (portability_subset)
    {
        *tail = &features.portability;
        tail = &features.portability.pNext;
    }
    if (features.maintenance4_queryable)
    {
        *tail = &features.maintenance4;
    }
    VkPhysicalDeviceFeatures2 features2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &features.v11};
    get_features2(physical_device, &features2);
    features.core = features2.features;
    features.v11.pNext = nullptr;
    features.v12.pNext = nullptr;
    features.portability.pNext = nullptr;
    features.maintenance4.pNext = nullptr;
    return features;
}

// The numbers a caller has to respect, as opposed to the capabilities above
// (0.26). A limit is not a Feature and must not become one: every device has a
// value for all of these, so the question is never "may I" but "how much".
//
// Which numbers are here is a judgement, not a dump. VkPhysicalDeviceLimits has
// about a hundred members and almost none of them ever change a decision; these
// are the ones that do — how big a buffer may get, and how work is spread over
// it. A field earns its place by being the reason some code takes a different
// path.
struct DeviceLimits
{
    // The most one DESCRIPTOR may see. Commonly 4 GiB - 1 on desktop drivers,
    // and the reason Feature::BUFFER_ADDRESS exists: an address is not a
    // descriptor, so a buffer read through one is not held to this.
    VkDeviceSize max_storage_buffer = 0;
    VkDeviceSize max_uniform_buffer = 0;
    // The most one VkBuffer may hold, and the most one allocation may be. Both
    // are usually far larger than the descriptor range above, which is what
    // makes that the binding limit rather than these.
    VkDeviceSize max_buffer = 0;
    VkDeviceSize max_allocation = 0;
    std::uint32_t max_push_constants = 0;

    // Compute: the shape of one workgroup, and how many of them one dispatch
    // may launch.
    std::array<std::uint32_t, 3> max_workgroup_size{};
    std::uint32_t max_workgroup_invocations = 0;
    std::uint32_t max_workgroup_memory = 0;
    std::array<std::uint32_t, 3> max_dispatch{};

    // The subgroup width, and the range it may be pinned to on a driver that
    // allows more than one. min == max on a device that has no range.
    std::uint32_t subgroup_size = 0;
    std::uint32_t min_subgroup_size = 0;
    std::uint32_t max_subgroup_size = 0;
};

// Same shape and the same reasoning as query_device_features: one query, so two
// callers cannot ask a different question. Takes the features because which
// property structs are legal to chain follows from the same extension list.
inline DeviceLimits query_device_limits(
    PFN_vkGetPhysicalDeviceProperties2 get_properties2,
    VkPhysicalDevice physical_device,
    const DeviceFeatures& features)
{
    const bool subgroup_control =
        std::ranges::find(features.extensions, std::string_view(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME)) !=
        features.extensions.end();

    VkPhysicalDeviceSubgroupProperties subgroup{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceVulkan11Properties v11{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
    VkPhysicalDeviceMaintenance4Properties maintenance4{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES};
    VkPhysicalDeviceSubgroupSizeControlProperties size_control{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};

    subgroup.pNext = &v11;
    void** tail = &v11.pNext;
    if (features.maintenance4_queryable)
    {
        *tail = &maintenance4;
        tail = &maintenance4.pNext;
    }
    if (subgroup_control)
    {
        *tail = &size_control;
    }

    VkPhysicalDeviceProperties2 props{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &subgroup};
    get_properties2(physical_device, &props);

    DeviceLimits limits;
    limits.max_storage_buffer = props.properties.limits.maxStorageBufferRange;
    limits.max_uniform_buffer = props.properties.limits.maxUniformBufferRange;
    limits.max_allocation = v11.maxMemoryAllocationSize;
    // maxBufferSize is maintenance4's own contribution. Without it the largest
    // buffer anyone can prove is the largest allocation that can back one.
    limits.max_buffer = features.maintenance4_queryable ? maintenance4.maxBufferSize : v11.maxMemoryAllocationSize;
    limits.max_push_constants = props.properties.limits.maxPushConstantsSize;

    const auto& core = props.properties.limits;
    limits.max_workgroup_size = {
        core.maxComputeWorkGroupSize[0], core.maxComputeWorkGroupSize[1], core.maxComputeWorkGroupSize[2]};
    limits.max_workgroup_invocations = core.maxComputeWorkGroupInvocations;
    limits.max_workgroup_memory = core.maxComputeSharedMemorySize;
    limits.max_dispatch = {
        core.maxComputeWorkGroupCount[0], core.maxComputeWorkGroupCount[1], core.maxComputeWorkGroupCount[2]};

    limits.subgroup_size = subgroup.subgroupSize;
    limits.min_subgroup_size = subgroup_control ? size_control.minSubgroupSize : subgroup.subgroupSize;
    limits.max_subgroup_size = subgroup_control ? size_control.maxSubgroupSize : subgroup.subgroupSize;
    return limits;
}

// A Feature is one boolean in one of the three structs above, so the table is a
// name plus three optional pointers-to-member of which exactly one is set.
//
// Four columns rather than a union or a variant: the table stays constexpr and
// stays an aggregate initializer, and the cost of a row that sets none is a
// feature that always reports false — loud the first time anyone asks for it.
//
// The fourth column arrived in 0.25 and this comment predicted it: a capability
// whose Vulkan spelling is an EXTENSION rather than a feature bit (exclusive
// fullscreen, and ray tracing or mesh shaders if they ever come) names the
// extension instead. It is deliberately not a fifth kind of thing at the call
// sites — feature_available and configure_features_ each grew one branch, and
// ctx.supports() reads the same for every row.
struct FeatureInfo
{
    Feature feature;
    const char* name;
    VkBool32 VkPhysicalDeviceFeatures::* core = nullptr;
    VkBool32 VkPhysicalDeviceVulkan11Features::* v11 = nullptr;
    VkBool32 VkPhysicalDeviceVulkan12Features::* v12 = nullptr;
    VkBool32 VkPhysicalDevicePortabilitySubsetFeaturesKHR::* portability = nullptr;
    VkBool32 VkPhysicalDeviceMaintenance4Features::* maintenance4 = nullptr;
    const char* extension = nullptr;
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
    {Feature::PRECISE_OCCLUSION, "PRECISE_OCCLUSION", &VkPhysicalDeviceFeatures::occlusionQueryPrecise},
    {.feature = Feature::MULTIVIEW, .name = "MULTIVIEW", .v11 = &VkPhysicalDeviceVulkan11Features::multiview},
    {.feature = Feature::BINDLESS, .name = "BINDLESS", .v12 = &VkPhysicalDeviceVulkan12Features::descriptorIndexing},
    {.feature = Feature::DRAW_INDIRECT_COUNT,
     .name = "DRAW_INDIRECT_COUNT",
     .v12 = &VkPhysicalDeviceVulkan12Features::drawIndirectCount},
    {.feature = Feature::COMPARISON_SAMPLER,
     .name = "COMPARISON_SAMPLER",
     .portability = &VkPhysicalDevicePortabilitySubsetFeaturesKHR::mutableComparisonSamplers},
    {.feature = Feature::SAMPLER_MIP_LOD_BIAS,
     .name = "SAMPLER_MIP_LOD_BIAS",
     .portability = &VkPhysicalDevicePortabilitySubsetFeaturesKHR::samplerMipLodBias},
    {.feature = Feature::MULTISAMPLE_ARRAYS,
     .name = "MULTISAMPLE_ARRAYS",
     .portability = &VkPhysicalDevicePortabilitySubsetFeaturesKHR::multisampleArrayImage},
    {.feature = Feature::IMAGE_VIEW_2D_ON_3D,
     .name = "IMAGE_VIEW_2D_ON_3D",
     .portability = &VkPhysicalDevicePortabilitySubsetFeaturesKHR::imageView2DOn3DImage},
    {.feature = Feature::TRIANGLE_FANS,
     .name = "TRIANGLE_FANS",
     .portability = &VkPhysicalDevicePortabilitySubsetFeaturesKHR::triangleFans},
    // The first row whose Vulkan spelling is an extension rather than a bit.
    {.feature = Feature::EXCLUSIVE_FULLSCREEN,
     .name = "EXCLUSIVE_FULLSCREEN",
     .extension = "VK_EXT_full_screen_exclusive"},
    {.feature = Feature::BUFFER_ADDRESS,
     .name = "BUFFER_ADDRESS",
     .v12 = &VkPhysicalDeviceVulkan12Features::bufferDeviceAddress},
    {Feature::SHADER_INT64, "SHADER_INT64", &VkPhysicalDeviceFeatures::shaderInt64},
    // The first row in its own struct, which is also the first capability with
    // both a core and an extension spelling of the same bit.
    // Both columns on purpose: the bit says whether the device has it, the
    // extension is how a 1.2 device is asked for it. On 1.3 the bit lives in
    // VkPhysicalDeviceVulkan13Features and the extension may not be advertised
    // at all, which enable_extension_if_present already treats as "nothing to
    // do" — so one row covers both drivers with no version test at the call site.
    {.feature = Feature::WORKGROUP_SIZE,
     .name = "WORKGROUP_SIZE",
     .maintenance4 = &VkPhysicalDeviceMaintenance4Features::maintenance4,
     .extension = VK_KHR_MAINTENANCE_4_EXTENSION_NAME},
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

// Not constexpr since 0.25: an extension row compares strings in a vector, which
// no caller evaluates at compile time anyway.
inline bool feature_available(const DeviceFeatures& available, Feature feature)
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
    if (info.v12)
    {
        return available.v12.*(info.v12) == VK_TRUE;
    }
    if (info.portability)
    {
        // Absent extension, no restriction. A portability row asks "may this
        // device still do the thing full Vulkan always does", so a driver that
        // never claimed to be a subset answers yes without being asked.
        return !available.portability_subset || available.portability.*(info.portability) == VK_TRUE;
    }
    if (info.maintenance4)
    {
        // Unasked is not the same as absent here, the opposite way round from a
        // portability row: nothing was chained because the device offered
        // neither spelling, so the struct holds its zero-initialised default and
        // reading it would be reading nothing.
        return available.maintenance4_queryable && available.maintenance4.*(info.maintenance4) == VK_TRUE;
    }
    if (info.extension)
    {
        return std::ranges::find(available.extensions, std::string_view(info.extension)) != available.extensions.end();
    }
    return false;
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
    else if (info.portability)
    {
        features.portability.*(info.portability) = VK_TRUE;
    }
    else if (info.maintenance4)
    {
        features.maintenance4.*(info.maintenance4) = VK_TRUE;
        // create_device_ reads this to decide whether to chain the struct at
        // all, so the bit and the permission to send it travel together.
        features.maintenance4_queryable = true;
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
