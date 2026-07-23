#pragma once
#include <volk.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <string>
#include <stdexcept>
#include <memory>
#include <expected>
#include <utility>

#include <atomic>

#include "Error.hpp"
#include "Logger.hpp"
#include "SurfaceProvider.hpp"

struct WindowDeleter
{
    void operator()(GLFWwindow* ptr) const noexcept
    {
        if (ptr)
        {
            glfwDestroyWindow(ptr);
        }
    };
};

struct MouseState
{
    float dx = 0.0f;
    float dy = 0.0f;
    float last_x = 0.0f;
    float last_y = 0.0f;
    bool first_mouse = true;
};

class Window
{
public:
    static inline std::atomic<int> window_count_{0};

    static std::expected<std::unique_ptr<Window>, Error> create(
        int width,
        int height,
        const std::string& title,
        std::shared_ptr<Logger> logger = nullptr)
    {
        // Must be installed before glfwInit — otherwise the most common failure a
        // new user hits (no display, no drivers) reports "Failed to create window"
        // and throws away the one string that says why.
        glfwSetErrorCallback(glfw_error_callback);
        glfw_logger_ = logger;

        if (window_count_.fetch_add(1) == 0)
        {
            if (!glfwInit())
            {
                window_count_.fetch_sub(1);
                return std::unexpected(err_window(describe_glfw_failure("Failed to initialize GLFW")));
            }
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

        GLFWwindow* raw_window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
        if (!raw_window)
        {
            if (window_count_.fetch_sub(1) == 1)
            {
                glfwTerminate();
            }
            return std::unexpected(err_window(describe_glfw_failure("Failed to create window")));
        }

        auto window = std::unique_ptr<Window>(new Window(width, height, title));
        window->window_.reset(raw_window);
        window->logger_ = std::move(logger);

        glfwSetWindowUserPointer(window->window_.get(), window.get());
        glfwSetCursorPosCallback(window->window_.get(), mouse_callback);
        glfwSetFramebufferSizeCallback(window->window_.get(), framebuffer_resize_callback);

        glfwSetInputMode(window->window_.get(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);

        return window;
    };

    ~Window()
    {
        if (window_count_.fetch_sub(1) == 1)
        {
            glfwTerminate();
        }
    };

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    bool should_close() const
    {
        return glfwWindowShouldClose(window_.get());
    }

    bool is_open() const
    {
        return !glfwWindowShouldClose(window_.get());
    }

    void poll_events()
    {
        glfwPollEvents();
    }

    bool is_key_pressed(int key) const
    {
        return glfwGetKey(window_.get(), key) == GLFW_PRESS;
    }

    bool is_mouse_button_pressed(int button) const
    {
        return glfwGetMouseButton(window_.get(), button) == GLFW_PRESS;
    }

    void set_cursor_mode(int mode)
    {
        glfwSetInputMode(window_.get(), GLFW_CURSOR, mode);
    }

    MouseState get_mouse_state() const
    {
        return mouse_;
    }

    GLFWwindow* get_native_handle() const
    {
        return window_.get();
    }

    void set_title(const std::string& title)
    {
        title_ = title;
        glfwSetWindowTitle(window_.get(), title_.c_str());
    }
    bool was_framebuffer_resized() const
    {
        return framebuffer_resized_;
    }
    void reset_framebuffer_resized()
    {
        framebuffer_resized_ = false;
    }

    void get_framebuffer_size(int& width, int& height) const
    {
        glfwGetFramebufferSize(window_.get(), &width, &height);
    }

    int get_width() const
    {
        int w, h;
        glfwGetWindowSize(window_.get(), &w, &h);
        return w;
    }

    int get_height() const
    {
        int w, h;
        glfwGetWindowSize(window_.get(), &w, &h);
        return h;
    }

    // Produce a SurfaceProvider that Renderer can use — decouples GLFW from Vulkan
    SurfaceProvider get_surface_provider()
    {
        SurfaceProvider sp;

        // GLFW knows which Vulkan instance extensions are required for the platform
        uint32_t count = 0;
        const char** exts = glfwGetRequiredInstanceExtensions(&count);
        if (exts)
        {
            sp.required_instance_extensions.assign(exts, exts + count);
        }

        GLFWwindow* raw = window_.get();

        sp.create_surface = [raw](VkInstance instance) -> VkSurfaceKHR
        {
            VkSurfaceKHR surface = VK_NULL_HANDLE;
            if (glfwCreateWindowSurface(instance, raw, nullptr, &surface) != VK_SUCCESS)
            {
                return VK_NULL_HANDLE;
            }
            return surface;
        };

        sp.get_framebuffer_size = [raw]() -> std::pair<int, int>
        {
            int w, h;
            glfwGetFramebufferSize(raw, &w, &h);
            return {w, h};
        };

        // Pointer to this Window's resize flag — consumed (read + reset) each check
        bool* resized_flag = &framebuffer_resized_;
        sp.consume_resize_flag = [resized_flag]() -> bool
        {
            bool was = *resized_flag;
            *resized_flag = false;
            return was;
        };

        return sp;
    }

private:
    Window(int width, int height, const std::string& title)
        : width_(width),
          height_(height),
          title_(title)
    {
    }

    // GLFW's error callback is global rather than per-window, so the routing has
    // to be global too. weak_ptr so this never keeps a Logger alive.
    static inline std::weak_ptr<Logger> glfw_logger_;
    static inline std::string last_glfw_error_;

    static void glfw_error_callback(int error_code, const char* description)
    {
        last_glfw_error_ = description ? std::string(description) : ("GLFW error " + std::to_string(error_code));

        if (auto logger = glfw_logger_.lock())
        {
            logger->log(Severity::Error, Source::Window, last_glfw_error_);
        }
    }

    // Consumes the recorded text, so an unrelated later failure cannot inherit a
    // stale diagnostic.
    static std::string describe_glfw_failure(std::string_view what)
    {
        std::string detail = std::move(last_glfw_error_);
        last_glfw_error_.clear();

        if (detail.empty())
        {
            return std::string(what);
        }
        return std::string(what) + ": " + detail;
    }

    std::unique_ptr<GLFWwindow, WindowDeleter> window_;
    std::shared_ptr<Logger> logger_;
    int width_;
    int height_;
    std::string title_;

    MouseState mouse_;
    bool framebuffer_resized_ = false;

    static void mouse_callback(GLFWwindow* window, double xpos, double ypos)
    {
        Window* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
        if (!win)
            return;

        float fx = static_cast<float>(xpos);
        float fy = static_cast<float>(ypos);

        if (win->mouse_.first_mouse)
        {
            win->mouse_.last_x = fx;
            win->mouse_.last_y = fy;
            win->mouse_.first_mouse = false;
        }

        win->mouse_.dx += fx - win->mouse_.last_x;
        win->mouse_.dy += win->mouse_.last_y - fy;

        win->mouse_.last_x = fx;
        win->mouse_.last_y = fy;
    };

    static void framebuffer_resize_callback(GLFWwindow* window, int width, int height)
    {
        Window* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
        if (win)
        {
            win->framebuffer_resized_ = true;
        }
    };
};