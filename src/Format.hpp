#pragma once
#include <volk.h>

#include <cstdint>

// Pixel formats. The name became available in 0.4, when the vertex-attribute
// enum was renamed to VertexFormat exactly so that pixels could have this one.
//
// One entry per format the library actually honours end-to-end (creation,
// upload, readback, numpy round trip). Growing this enum is additive; every
// consumer goes through format_info(), whose switch has no default, so a new
// entry that misses a consumer is a compile error, not a silent fallback.
enum class Format
{
    RGBA8,      // R8G8B8A8_UNORM — data. The default for arrays and render targets.
    RGBA8_SRGB, // R8G8B8A8_SRGB — pictures. What load_image() decodes into.
    BGRA8,      // B8G8R8A8_UNORM — the common swapchain byte order.
    R8,
    RG8,
    R16F,
    RGBA16F,
    R32F,
    RGBA32F,
    D32F, // 32-bit float depth.
};

struct FormatInfo
{
    VkFormat vk;
    std::uint32_t bytes_per_pixel;
    std::uint32_t channels;
    const char* numpy_dtype; // as understood by py::dtype(...)
    bool depth;
};

// No default on the switch: adding a Format must break compilation everywhere
// the table is consulted. The fallback AFTER the switch is separate and
// deliberate — pybind enums accept arbitrary ints (bz.Format(999) constructs),
// so the "impossible" path is reachable from Python and must not be UB.
constexpr FormatInfo format_info(Format f)
{
    switch (f)
    {
        case Format::RGBA8:
            return {VK_FORMAT_R8G8B8A8_UNORM, 4, 4, "uint8", false};
        case Format::RGBA8_SRGB:
            return {VK_FORMAT_R8G8B8A8_SRGB, 4, 4, "uint8", false};
        case Format::BGRA8:
            return {VK_FORMAT_B8G8R8A8_UNORM, 4, 4, "uint8", false};
        case Format::R8:
            return {VK_FORMAT_R8_UNORM, 1, 1, "uint8", false};
        case Format::RG8:
            return {VK_FORMAT_R8G8_UNORM, 2, 2, "uint8", false};
        case Format::R16F:
            return {VK_FORMAT_R16_SFLOAT, 2, 1, "float16", false};
        case Format::RGBA16F:
            return {VK_FORMAT_R16G16B16A16_SFLOAT, 8, 4, "float16", false};
        case Format::R32F:
            return {VK_FORMAT_R32_SFLOAT, 4, 1, "float32", false};
        case Format::RGBA32F:
            return {VK_FORMAT_R32G32B32A32_SFLOAT, 16, 4, "float32", false};
        case Format::D32F:
            return {VK_FORMAT_D32_SFLOAT, 4, 1, "float32", true};
    }
    return {VK_FORMAT_R8G8B8A8_UNORM, 4, 4, "uint8", false};
}

constexpr const char* format_name(Format f)
{
    switch (f)
    {
        case Format::RGBA8:
            return "RGBA8";
        case Format::RGBA8_SRGB:
            return "RGBA8_SRGB";
        case Format::BGRA8:
            return "BGRA8";
        case Format::R8:
            return "R8";
        case Format::RG8:
            return "RG8";
        case Format::R16F:
            return "R16F";
        case Format::RGBA16F:
            return "RGBA16F";
        case Format::R32F:
            return "R32F";
        case Format::RGBA32F:
            return "RGBA32F";
        case Format::D32F:
            return "D32F";
    }
    return "RGBA8";
}
