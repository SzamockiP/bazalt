#pragma once
#include <volk.h>

#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

// Unified error type for the whole library.
//
// Every fallible operation returns std::expected<T, Error>. The ErrorCode maps
// 1:1 onto a Python exception class at the pybind boundary (see main.cpp), so
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

// Constructors for non-Vulkan failures.

inline Error err_init(std::string message)
{
    return {ErrorCode::Initialization, std::move(message)};
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
