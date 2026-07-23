#pragma once
#include <volk.h>
#include <functional>
#include <vector>
#include <utility>

// Minimal contract: what Renderer needs from any windowing system.
// No inheritance, no virtual — just callbacks.
struct SurfaceProvider
{
    // Vulkan instance extensions required by this windowing system
    std::vector<const char*> required_instance_extensions;

    // Creates VkSurfaceKHR — called once after VkInstance is created
    std::function<VkSurfaceKHR(VkInstance)> create_surface;

    // Returns current framebuffer size (width, height)
    std::function<std::pair<int, int>()> get_framebuffer_size;

    // Returns true if the window was resized since last check, and resets the flag
    std::function<bool()> consume_resize_flag;
};
