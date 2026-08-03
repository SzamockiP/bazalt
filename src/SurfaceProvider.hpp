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

    // The HMONITOR the window currently sits on, as an opaque pointer, or null
    // where the platform has no such thing (0.25). Exclusive fullscreen is the
    // only caller: VK_EXT_full_screen_exclusive requires it in the swapchain's
    // pNext on Win32, and a swapchain has no other reason to know about
    // monitors. void* keeps <windows.h> out of this header, which every
    // platform includes.
    std::function<void*()> get_win32_monitor;
};
