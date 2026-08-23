#pragma once
#include <volk.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

#include "Error.hpp"
#include "Features.hpp"
#include "ScopeGuard.hpp"

// The stable identity of a GPU: the same 16 bytes for the same card across
// instances, drivers and runs. Names are not identities (two identical cards
// share one), and indices are not either (enumeration order is a driver's
// business), so this is what Context matches a chosen Device against.
using DeviceUUID = std::array<std::uint8_t, VK_UUID_SIZE>;

// A GPU the machine can offer, as inert data.
//
// Deliberately holds NO VkPhysicalDevice and NO VkInstance. A physical-device
// handle only means anything inside the instance that enumerated it, and
// list_devices() destroys its instance before returning — so a Device that
// carried one would be a dangling handle wearing a nice name. Everything here
// is a copy, which is also why a Device stays valid for a Context created
// minutes later, or for no Context at all.
struct Device
{
    std::string name;
    // "discrete", "integrated", "virtual", "cpu" or "other" — the plain word,
    // not VkPhysicalDeviceType, because picking a GPU is the one moment a user
    // wants to read rather than decode.
    std::string type;
    std::uint32_t api_version = 0;
    DeviceUUID uuid{};

    DeviceFeatures features{};
    // The same object ctx.limits answers with, asked before there is a Context
    // — so "which of these cards can hold the file" is answerable at the moment
    // the card is being chosen. `memory_bytes` used to live here beside it,
    // which made the numbers about a GPU come from two places depending on
    // whether a Context existed yet; it is limits.device_memory now (0.26).
    DeviceLimits limits{};

    // Same question as ctx.supports(), asked before there is a Context — so a
    // caller can pick the card that can do the job instead of finding out at
    // create time. Reads the copied feature table through the same helper the
    // Context path uses.
    bool supports(Feature feature) const
    {
        return feature_available(features, feature);
    }
};

inline const char* device_type_name(VkPhysicalDeviceType type)
{
    switch (type)
    {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            return "discrete";
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            return "integrated";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            return "virtual";
        case VK_PHYSICAL_DEVICE_TYPE_CPU:
            return "cpu";
        default:
            return "other";
    }
}

// Read the UUID of an already-enumerated physical device. Used twice: once
// while listing, once inside Context to match a chosen Device against the
// devices its own instance can see.
inline DeviceUUID device_uuid(PFN_vkGetPhysicalDeviceProperties2 get_properties2, VkPhysicalDevice physical_device)
{
    VkPhysicalDeviceIDProperties id{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES, .pNext = nullptr};
    VkPhysicalDeviceProperties2 props2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &id};
    get_properties2(physical_device, &props2);

    DeviceUUID uuid{};
    std::copy(std::begin(id.deviceUUID), std::end(id.deviceUUID), uuid.begin());
    return uuid;
}

// Enumerate every GPU on this machine, without creating a device — the step
// that has to happen BEFORE a Context exists, since choosing the GPU is the
// whole point.
//
// Vulkan has no way to enumerate physical devices without a VkInstance, so this
// builds a bare one (no layers, no extensions) and destroys it on the way out.
// The entry points are fetched through vkGetInstanceProcAddr rather than
// volkLoadInstance on purpose: volk's loader is global, so binding it to this
// throwaway instance would silently redirect a live Context's instance-level
// calls at an instance that is about to be destroyed. Same pattern
// bindings/Targets.cpp already uses for vkCreateWin32SurfaceKHR.
inline std::expected<std::vector<Device>, Error> list_devices()
{
    if (volkInitialize() != VK_SUCCESS)
    {
        return std::unexpected(err_no_vulkan_loader());
    }

    // 1.2 is bazalt's baseline anyway, and it is what makes the Properties2 /
    // Features2 chains below (UUID, multiview) legal to ask for.
    VkApplicationInfo app_info{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = nullptr,
        .pApplicationName = "bazalt.list_devices",
        .applicationVersion = 0,
        .pEngineName = "bazalt",
        .engineVersion = 0,
        .apiVersion = VK_API_VERSION_1_2};
    // A portability driver -- MoltenVK is the one that exists -- is invisible to a
    // plain vkCreateInstance, which fails with VK_ERROR_INCOMPATIBLE_DRIVER rather
    // than returning zero devices. Context does not need this code because
    // vk-bootstrap does it there. This instance is built by hand, so it asks for
    // itself, and it asks only when the loader offers the extension: enabling an
    // absent extension is a hard failure, and on Windows and Linux it is absent.
    std::uint32_t extension_count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);
    std::vector<VkExtensionProperties> extensions(extension_count);
    vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, extensions.data());

    const bool portability = std::ranges::any_of(
        extensions,
        [](const VkExtensionProperties& extension)
        { return std::string_view(extension.extensionName) == VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME; });

    const char* portability_extension = VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
    VkInstanceCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = portability ? VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR : VkInstanceCreateFlags{0},
        .pApplicationInfo = &app_info,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = nullptr,
        .enabledExtensionCount = portability ? 1u : 0u,
        .ppEnabledExtensionNames = portability ? &portability_extension : nullptr};

    VkInstance instance = VK_NULL_HANDLE;
    if (auto e = check(vkCreateInstance(&create_info, nullptr, &instance), "create instance to list devices"))
    {
        return std::unexpected(*e);
    }

    auto load = [instance](const char* symbol) { return vkGetInstanceProcAddr(instance, symbol); };
    auto destroy_instance = (PFN_vkDestroyInstance)load("vkDestroyInstance");
    auto enumerate = (PFN_vkEnumeratePhysicalDevices)load("vkEnumeratePhysicalDevices");
    auto get_properties2 = (PFN_vkGetPhysicalDeviceProperties2)load("vkGetPhysicalDeviceProperties2");
    auto get_features2 = (PFN_vkGetPhysicalDeviceFeatures2)load("vkGetPhysicalDeviceFeatures2");
    auto get_memory = (PFN_vkGetPhysicalDeviceMemoryProperties)load("vkGetPhysicalDeviceMemoryProperties");
    auto enumerate_device_extensions =
        (PFN_vkEnumerateDeviceExtensionProperties)load("vkEnumerateDeviceExtensionProperties");

    ScopeGuard cleanup(
        [&]
        {
            if (destroy_instance)
            {
                destroy_instance(instance, nullptr);
            }
        });

    if (!enumerate || !get_properties2 || !get_features2 || !get_memory)
    {
        return std::unexpected(err_init(
            "Vulkan: the loader on this machine does not expose the 1.1 device-query "
            "entry points, so bazalt cannot list GPUs. Updating the graphics driver "
            "usually fixes this."));
    }

    std::uint32_t count = 0;
    if (auto e = check(enumerate(instance, &count, nullptr), "enumerate physical devices"))
    {
        return std::unexpected(*e);
    }
    std::vector<VkPhysicalDevice> handles(count);
    if (count > 0)
    {
        if (auto e = check(enumerate(instance, &count, handles.data()), "enumerate physical devices"))
        {
            return std::unexpected(*e);
        }
    }

    std::vector<Device> devices;
    devices.reserve(handles.size());
    for (VkPhysicalDevice handle : handles)
    {
        VkPhysicalDeviceIDProperties id{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES, .pNext = nullptr};
        VkPhysicalDeviceProperties2 props2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &id};
        get_properties2(handle, &props2);

        VkPhysicalDeviceMemoryProperties memory{};
        get_memory(handle, &memory);
        std::uint64_t device_local = 0;
        for (std::uint32_t i = 0; i < memory.memoryHeapCount; ++i)
        {
            if (memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            {
                device_local += memory.memoryHeaps[i].size;
            }
        }

        Device device;
        device.name = props2.properties.deviceName;
        device.type = device_type_name(props2.properties.deviceType);
        device.api_version = props2.properties.apiVersion;
        std::copy(std::begin(id.deviceUUID), std::end(id.deviceUUID), device.uuid.begin());
        // Asked per device, because the portability rows read the opposite way for
        // a device that is a subset and one that is not (see feature_available).
        // A loader too old to expose the entry point simply reports no subset,
        // which is what every non-portability driver reports anyway.
        bool portability_subset = false;
        if (enumerate_device_extensions)
        {
            std::uint32_t device_extension_count = 0;
            enumerate_device_extensions(handle, nullptr, &device_extension_count, nullptr);
            std::vector<VkExtensionProperties> device_extensions(device_extension_count);
            enumerate_device_extensions(handle, nullptr, &device_extension_count, device_extensions.data());
            portability_subset = std::ranges::any_of(
                device_extensions,
                [](const VkExtensionProperties& extension)
                { return std::string_view(extension.extensionName) == VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME; });
        }
        device.features = query_device_features(
            get_features2, enumerate_device_extensions, handle, portability_subset, props2.properties.apiVersion);
        device.limits = query_device_limits(get_properties2, handle, device.features);
        device.limits.device_memory = device_local;
        devices.push_back(std::move(device));
    }

    return devices;
}
