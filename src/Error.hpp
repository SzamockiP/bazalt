#pragma once
#include <volk.h>

#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

// Unified error type for the whole library.
//
// Every fallible operation returns std::expected<T, Error>. The ErrorCode maps
// 1:1 onto a Python exception class at the pybind boundary (see
// bindings/Common.hpp), so
// the code a call site picks decides what the user is able to catch.
//
// The distinction that matters is *recoverability*: ShaderError must be
// catchable on its own, or a typo in a shader kills the application and hot
// reload is pointless.

enum class ErrorCode
{
    // Fatal — the Context is unusable afterwards.
    Initialization, // no Vulkan, no suitable GPU, a required feature is missing
    DeviceLost,     // VK_ERROR_DEVICE_LOST

    // Sometimes recoverable — free something and retry.
    OutOfMemory,

    // Recoverable — the caller can fix the input and try again.
    Shader,   // compilation/linking; carries path + line
    Window,   // GLFW; carries the real platform diagnostic
    Resource, // missing file, bad format, exhausted pool
};

struct Error
{
    ErrorCode code = ErrorCode::Initialization;
    std::string message;

    // Preserved when the error came from a Vulkan call. A failed vkCreate*
    // used to collapse to a bare string, losing which VkResult it was — that
    // makes driver-specific failures undiagnosable from a bug report.
    VkResult result = VK_SUCCESS;

    // ErrorCode::Shader only.
    std::string path;
    int line = -1;
};

inline constexpr std::string_view vk_result_name(VkResult result)
{
    switch (result)
    {
        case VK_SUCCESS:
            return "VK_SUCCESS";
        case VK_NOT_READY:
            return "VK_NOT_READY";
        case VK_TIMEOUT:
            return "VK_TIMEOUT";
        case VK_INCOMPLETE:
            return "VK_INCOMPLETE";
        case VK_SUBOPTIMAL_KHR:
            return "VK_SUBOPTIMAL_KHR";
        case VK_ERROR_OUT_OF_HOST_MEMORY:
            return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:
            return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED:
            return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST:
            return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED:
            return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT:
            return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT:
            return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT:
            return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER:
            return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS:
            return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED:
            return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_FRAGMENTED_POOL:
            return "VK_ERROR_FRAGMENTED_POOL";
        case VK_ERROR_OUT_OF_POOL_MEMORY:
            return "VK_ERROR_OUT_OF_POOL_MEMORY";
        case VK_ERROR_INVALID_EXTERNAL_HANDLE:
            return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
        case VK_ERROR_FRAGMENTATION:
            return "VK_ERROR_FRAGMENTATION";
        case VK_ERROR_SURFACE_LOST_KHR:
            return "VK_ERROR_SURFACE_LOST_KHR";
        case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR:
            return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
        case VK_ERROR_OUT_OF_DATE_KHR:
            return "VK_ERROR_OUT_OF_DATE_KHR";
        case VK_ERROR_INCOMPATIBLE_DISPLAY_KHR:
            return "VK_ERROR_INCOMPATIBLE_DISPLAY_KHR";
        case VK_ERROR_VALIDATION_FAILED_EXT:
            return "VK_ERROR_VALIDATION_FAILED_EXT";
        case VK_ERROR_UNKNOWN:
            return "VK_ERROR_UNKNOWN";
        default:
            return "VkResult";
    }
}

// Vulkan tells us the *kind* of failure; the call site tells us the *context*.
// Well-known results map to a code wherever they happen; everything else falls
// back to whatever the caller considers appropriate.
inline constexpr ErrorCode code_from_vk_result(VkResult result, ErrorCode fallback)
{
    switch (result)
    {
        case VK_ERROR_DEVICE_LOST:
            return ErrorCode::DeviceLost;
        case VK_ERROR_OUT_OF_HOST_MEMORY:
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:
        case VK_ERROR_OUT_OF_POOL_MEMORY:
            return ErrorCode::OutOfMemory;
        default:
            return fallback;
    }
}

// Returns nullopt on success, so call sites read:
//
//   if (auto e = check(vkCreateCommandPool(...), "create command pool"))
//       return std::unexpected(*e);
inline std::optional<Error> check(
    VkResult result,
    std::string_view what,
    ErrorCode fallback = ErrorCode::Initialization)
{
    if (result == VK_SUCCESS)
    {
        return std::nullopt;
    }

    Error error;
    error.code = code_from_vk_result(result, fallback);
    error.result = result;
    error.message = std::format("Vulkan: failed to {} ({})", what, vk_result_name(result));
    return error;
}

// Does [offset, offset + length) fit inside a region of `size` bytes?
//
// Every one of these bounds checks used to be written `offset + length > size`, which
// is a bypass rather than a check: the operands are unsigned, so an offset near the type
// maximum wraps the sum to a small number and the check passes. The Python boundary hands
// these straight through, so `buffer.update(data, offset=2**64 - 10)` reached a memcpy.
// Subtracting instead of adding cannot overflow, because the first test proves
// `size - offset` does not wrap.
//
// Three type parameters, not one. The three arguments are naturally a
// VkDeviceSize, a length and a container's size(), and whether those are the SAME
// type is a platform accident: on Windows and Linux VkDeviceSize and size_t are
// spelled by the same underlying type, so one T deduced fine for twenty releases.
// On macOS size_t is `unsigned long` and VkDeviceSize is `unsigned long long` --
// same width, different types -- and every call site failed to deduce.
//
// A signed argument still cannot slip through: common_type_t of a signed and an
// unsigned type is unsigned, so a negative offset converts to a huge one and the
// first test refuses it. Wrong the safe way, which is the only direction a bounds
// check may be wrong in.
template <typename T, typename U, typename V>
constexpr bool fits_within(T offset, U length, V size)
{
    using Widest = std::common_type_t<T, U, V>;
    static_assert(std::is_unsigned_v<Widest>, "fits_within compares sizes, and a size is never negative");

    const auto offset_bytes = static_cast<Widest>(offset);
    const auto length_bytes = static_cast<Widest>(length);
    const auto size_bytes = static_cast<Widest>(size);
    return offset_bytes <= size_bytes && length_bytes <= size_bytes - offset_bytes;
}

// Constructors for non-Vulkan failures.

inline Error err_init(std::string message)
{
    return {ErrorCode::Initialization, std::move(message)};
}

// volkInitialize() fails for exactly one reason: it could not open the Vulkan
// loader. `check()` would report that as "failed to initialize volk
// (VK_ERROR_INITIALIZATION_FAILED)", which names a library the caller did not
// install and no action at all.
//
// Windows and Linux get the loader from the graphics driver. macOS supplies no
// Vulkan at all, so this is the first thing a new Mac user meets, and the fix
// belongs in the message rather than in the documentation they are not reading
// yet.
inline Error err_no_vulkan_loader()
{
    return err_init(
        "Vulkan: no Vulkan loader is installed on this machine. "
#ifdef __APPLE__
        "macOS supplies no Vulkan, so install the Vulkan SDK from LunarG. It gives you the loader and "
        "MoltenVK, which runs Vulkan on Metal."
#else
        "Install a graphics driver that supports Vulkan."
#endif
    );
}

inline Error err_window(std::string message)
{
    return {ErrorCode::Window, std::move(message)};
}

inline Error err_resource(std::string message)
{
    return {ErrorCode::Resource, std::move(message)};
}

inline Error err_shader(std::string message, std::string path = {}, int line = -1)
{
    Error error;
    error.code = ErrorCode::Shader;
    error.message = std::move(message);
    error.path = std::move(path);
    error.line = line;
    return error;
}
