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
    // An unsigned integer target (`uvec`/`uint` in the shader, no filtering and
    // no conversion). What an id buffer is made of: render an object index and
    // read one pixel back to know what the cursor is over.
    R32_UINT,
    // Packed HDR colour in 4 bytes instead of 8. Half the bandwidth of RGBA16F
    // for a bloom or a light-accumulation target, and it has no alpha, which is
    // what makes it cheap.
    R11G11B10F,
    // Depth AND stencil. The exact VkFormat is chosen per device — see
    // depth_stencil_format() on the Context — because the spec guarantees only
    // that ONE of D24_UNORM_S8_UINT / D32_SFLOAT_S8_UINT is supported, and which
    // one it is is Vulkan trivia this library exists to hide. Not readable and
    // not sampleable: a combined format has no single numpy dtype.
    DEPTH_STENCIL,
};

struct FormatInfo
{
    // VK_FORMAT_UNDEFINED means "resolved per device" — ask the Context, do not
    // pass this to Vulkan. DEPTH_STENCIL is the only such entry today.
    VkFormat vk;
    std::uint32_t bytes_per_pixel;
    std::uint32_t channels;
    const char* numpy_dtype; // as understood by py::dtype(...); empty = not readable
    bool depth;
};

// The aspect an image of this format is addressed through. Views, barriers,
// copies and attachment infos must all agree, so they all read it from here —
// the 0.13 lesson about the view and the barrier coming from one source, now
// applied to the aspect as well.
inline constexpr VkImageAspectFlags aspect_mask_for(VkFormat format)
{
    switch (format)
    {
        case VK_FORMAT_D32_SFLOAT:
        case VK_FORMAT_D16_UNORM:
            return VK_IMAGE_ASPECT_DEPTH_BIT;
        case VK_FORMAT_D24_UNORM_S8_UINT:
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
        case VK_FORMAT_D16_UNORM_S8_UINT:
            return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        default:
            return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}

inline constexpr bool has_stencil(VkFormat format)
{
    return (aspect_mask_for(format) & VK_IMAGE_ASPECT_STENCIL_BIT) != 0;
}

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
        case Format::R32_UINT:
            return {VK_FORMAT_R32_UINT, 4, 1, "uint32", false};
        case Format::R11G11B10F:
            // Packed: three channels in one 32-bit word, so there is no numpy
            // dtype that describes a pixel. Render into it, sample it, and read
            // it back only after a shader has unpacked it into something else.
            return {VK_FORMAT_B10G11R11_UFLOAT_PACK32, 4, 3, "", false};
        case Format::DEPTH_STENCIL:
            return {VK_FORMAT_UNDEFINED, 4, 1, "", true};
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
        case Format::R32_UINT:
            return "R32_UINT";
        case Format::R11G11B10F:
            return "R11G11B10F";
        case Format::DEPTH_STENCIL:
            return "DEPTH_STENCIL";
    }
    return "RGBA8";
}
