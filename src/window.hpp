#pragma once
#include <volk.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <string>
#include <stdexcept>
#include <memory>
#include <expected>
#include <utility>
#include <vector>
#include <algorithm>
#include <cstdint>

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

// What the pointer did. x/y are the current position; dx/dy and scroll are the
// change during the last poll cycle, so a camera reads them and does no
// bookkeeping. dx/dy used to be running totals, which made every caller
// subtract the previous total to get a delta — eight examples carried the same
// three lines.
struct MouseState
{
    float x = 0.0f;
    float y = 0.0f;
    float dx = 0.0f;
    float dy = 0.0f;
    float scroll_dx = 0.0f;
    float scroll_dy = 0.0f;
};

// How the window presents itself. One enum and one verb rather than a
// set_fullscreen plus a set_decorated plus the rules for combining them: the
// four states are exclusive, and a bool pair would spell two of them twice.
//
// FULLSCREEN is not *exclusive* fullscreen. It takes the monitor and its
// current video mode, but the swapchain stays composited, because exclusive
// fullscreen needs VK_EXT_full_screen_exclusive. That is a Feature, not a
// window mode, and it can arrive later without touching this enum.
enum class WindowMode
{
    // Decorated, at the position and size it had before it left this mode.
    WINDOWED,
    // No title bar and no border, same geometry. No title bar also means no
    // drag handle, so a frameless window is moved by set_position.
    FRAMELESS,
    // Takes the monitor at that monitor's current video mode.
    FULLSCREEN,
    // Borderless, positioned and sized to fill the monitor. No monitor
    // takeover and no video-mode change, so alt-tab and a second display keep
    // behaving normally.
    FULLSCREEN_WINDOWED
};

class Window
{
public:
    static inline std::atomic<int> window_count_{0};

    // Number of completed poll cycles. Every per-cycle input query compares
    // against this to decide whether what it holds is still current, so one
    // counter serves the mouse delta, the scroll and the key edges.
    static inline std::atomic<std::uint64_t> poll_generation_{0};

    static std::expected<std::unique_ptr<Window>, Error> create(
        int width,
        int height,
        const std::string& title,
        std::shared_ptr<Logger> logger = nullptr,
        WindowMode mode = WindowMode::WINDOWED)
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
        glfwSetScrollCallback(window->window_.get(), scroll_callback);
        glfwSetKeyCallback(window->window_.get(), key_callback);
        glfwSetMouseButtonCallback(window->window_.get(), mouse_button_callback);
        glfwSetFramebufferSizeCallback(window->window_.get(), framebuffer_resize_callback);

        // Start level with the world: a window created mid-loop must not rotate
        // an empty pending_ into current_ on its first query.
        window->seen_generation_ = poll_generation_.load(std::memory_order_relaxed);

        glfwSetInputMode(window->window_.get(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);

        // The requested width/height/position become what WINDOWED returns to,
        // whatever mode the window opens in. Then the mode goes on through
        // set_mode: one implementation, so opening fullscreen and switching to
        // fullscreen cannot drift apart.
        window->save_windowed_geometry_();
        if (auto applied = window->set_mode(mode); !applied)
        {
            return std::unexpected(applied.error());
        }

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

    bool is_open() const
    {
        return !glfwWindowShouldClose(window_.get());
    }

    bool is_key_pressed(int key) const
    {
        return glfwGetKey(window_.get(), key) == GLFW_PRESS;
    }

    bool is_mouse_button_pressed(int button) const
    {
        return glfwGetMouseButton(window_.get(), button) == GLFW_PRESS;
    }

    // Went down during the last poll cycle — the edge, where is_key_pressed is
    // the level. A toggle (fullscreen, wireframe, pause) needs the edge, and
    // without one every caller keeps its own "was it down last frame" bool.
    // GLFW_REPEAT is not an edge: auto-repeat would fire the toggle again.
    bool was_key_pressed(int key) const
    {
        rotate_();
        return std::ranges::find(current_.keys, key) != current_.keys.end();
    }

    bool was_mouse_button_pressed(int button) const
    {
        rotate_();
        return std::ranges::find(current_.buttons, button) != current_.buttons.end();
    }

    void set_cursor_mode(int mode)
    {
        glfwSetInputMode(window_.get(), GLFW_CURSOR, mode);
    }

    MouseState get_mouse_state() const
    {
        rotate_();
        return MouseState{
            .x = pos_x_,
            .y = pos_y_,
            .dx = current_.dx,
            .dy = current_.dy,
            .scroll_dx = current_.scroll_dx,
            .scroll_dy = current_.scroll_dy};
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

    WindowMode mode() const
    {
        return mode_;
    }

    // A verb rather than a settable property, to match set_title and
    // set_cursor_mode, and because this one can fail: an assignment that raises
    // WindowError reads worse than a call that does.
    //
    // The swapchain needs nothing from here. Every one of these calls resizes
    // the framebuffer, framebuffer_resize_callback records it, and present()
    // consumes the flag and recreates — the same path a dragged window edge
    // takes.
    std::expected<void, Error> set_mode(WindowMode mode)
    {
        if (mode == mode_)
        {
            return {};
        }

        GLFWwindow* win = window_.get();

        // Only the two windowed modes have a geometry worth coming back to.
        if (mode_ == WindowMode::WINDOWED || mode_ == WindowMode::FRAMELESS)
        {
            save_windowed_geometry_();
        }

        if (mode == WindowMode::WINDOWED || mode == WindowMode::FRAMELESS)
        {
            glfwSetWindowAttrib(win, GLFW_DECORATED, mode == WindowMode::WINDOWED ? GLFW_TRUE : GLFW_FALSE);
            // nullptr releases the monitor if one was held; with no monitor to
            // release this is only the move-and-resize back to the saved rect.
            glfwSetWindowMonitor(win, nullptr, saved_x_, saved_y_, saved_width_, saved_height_, GLFW_DONT_CARE);
        }
        else
        {
            GLFWmonitor* monitor = current_monitor_();
            const GLFWvidmode* video_mode = monitor ? glfwGetVideoMode(monitor) : nullptr;
            if (!monitor || !video_mode)
            {
                return std::unexpected(
                    err_window(describe_glfw_failure("Cannot go fullscreen: no monitor reported a video mode")));
            }

            if (mode == WindowMode::FULLSCREEN)
            {
                glfwSetWindowAttrib(win, GLFW_DECORATED, GLFW_TRUE);
                glfwSetWindowMonitor(
                    win, monitor, 0, 0, video_mode->width, video_mode->height, video_mode->refreshRate);
            }
            else
            {
                int monitor_x = 0;
                int monitor_y = 0;
                glfwGetMonitorPos(monitor, &monitor_x, &monitor_y);
                glfwSetWindowAttrib(win, GLFW_DECORATED, GLFW_FALSE);
                glfwSetWindowMonitor(
                    win, nullptr, monitor_x, monitor_y, video_mode->width, video_mode->height, GLFW_DONT_CARE);
            }
        }

        mode_ = mode;
        return {};
    }

    void set_size(int width, int height)
    {
        glfwSetWindowSize(window_.get(), width, height);
    }

    void set_position(int x, int y)
    {
        glfwSetWindowPos(window_.get(), x, y);
    }

    std::pair<int, int> get_position() const
    {
        int x = 0;
        int y = 0;
        glfwGetWindowPos(window_.get(), &x, &y);
        return {x, y};
    }

    void set_resizable(bool enable)
    {
        glfwSetWindowAttrib(window_.get(), GLFW_RESIZABLE, enable ? GLFW_TRUE : GLFW_FALSE);
    }

    bool is_resizable() const
    {
        return glfwGetWindowAttrib(window_.get(), GLFW_RESIZABLE) == GLFW_TRUE;
    }

    void set_always_on_top(bool enable)
    {
        glfwSetWindowAttrib(window_.get(), GLFW_FLOATING, enable ? GLFW_TRUE : GLFW_FALSE);
    }

    bool is_always_on_top() const
    {
        return glfwGetWindowAttrib(window_.get(), GLFW_FLOATING) == GLFW_TRUE;
    }

    void set_opacity(float opacity)
    {
        glfwSetWindowOpacity(window_.get(), opacity);
    }

    float get_opacity() const
    {
        return glfwGetWindowOpacity(window_.get());
    }

    // Framebuffer pixels per screen coordinate — 2.0 on a HiDPI display. What a
    // UI needs to size text, and the reason width/height (screen coordinates)
    // and the swapchain extent (pixels) can disagree.
    std::pair<float, float> get_content_scale() const
    {
        float x = 1.0f;
        float y = 1.0f;
        glfwGetWindowContentScale(window_.get(), &x, &y);
        return {x, y};
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

    void save_windowed_geometry_()
    {
        glfwGetWindowPos(window_.get(), &saved_x_, &saved_y_);
        glfwGetWindowSize(window_.get(), &saved_width_, &saved_height_);
    }

    // The monitor this window sits on, by largest overlapping area. The primary
    // monitor is the fallback, not the answer: on a two-display setup it sends
    // fullscreen to the other screen, which reads as a bug and is one.
    GLFWmonitor* current_monitor_() const
    {
        int window_x = 0;
        int window_y = 0;
        int window_w = 0;
        int window_h = 0;
        glfwGetWindowPos(window_.get(), &window_x, &window_y);
        glfwGetWindowSize(window_.get(), &window_w, &window_h);

        int count = 0;
        GLFWmonitor** monitors = glfwGetMonitors(&count);
        GLFWmonitor* best = glfwGetPrimaryMonitor();
        int best_area = 0;

        for (int i = 0; i < count; ++i)
        {
            int monitor_x = 0;
            int monitor_y = 0;
            glfwGetMonitorPos(monitors[i], &monitor_x, &monitor_y);
            const GLFWvidmode* video_mode = glfwGetVideoMode(monitors[i]);
            if (!video_mode)
            {
                continue;
            }

            // Parenthesised because <windows.h> arrives through volk with min
            // and max as macros, and they catch qualified calls too.
            const int overlap_w = (std::ranges::min)(window_x + window_w, monitor_x + video_mode->width) -
                                  (std::ranges::max)(window_x, monitor_x);
            const int overlap_h = (std::ranges::min)(window_y + window_h, monitor_y + video_mode->height) -
                                  (std::ranges::max)(window_y, monitor_y);
            if (overlap_w <= 0 || overlap_h <= 0)
            {
                continue;
            }

            const int area = overlap_w * overlap_h;
            if (area > best_area)
            {
                best_area = area;
                best = monitors[i];
            }
        }

        return best;
    }

    std::unique_ptr<GLFWwindow, WindowDeleter> window_;
    std::shared_ptr<Logger> logger_;
    int width_;
    int height_;
    std::string title_;

    WindowMode mode_ = WindowMode::WINDOWED;
    int saved_x_ = 0;
    int saved_y_ = 0;
    int saved_width_ = 0;
    int saved_height_ = 0;

    // Everything the callbacks accumulate that is a *change* rather than a
    // state: it belongs to one poll cycle and must read the same twice within
    // that cycle.
    struct PollState
    {
        float dx = 0.0f;
        float dy = 0.0f;
        float scroll_dx = 0.0f;
        float scroll_dy = 0.0f;
        // A handful of keys per cycle, so a linear scan beats a set.
        std::vector<int> keys;
        std::vector<int> buttons;
    };

    // The callbacks append to pending_; a reader whose generation is stale
    // promotes pending_ to current_. Reader-driven on purpose: poll_events()
    // has no list of live windows, and giving it one means a global mutable
    // vector plus a lock for work the reader can do itself. The callbacks must
    // NOT rotate — one that did would mark the generation seen halfway through
    // the cycle and hide every later event of the same cycle.
    //
    // Rejected: consuming on read. `was_key_pressed(KEY_F11)` twice in one
    // frame would answer True then False, which is a trap rather than an API.
    void rotate_() const
    {
        const std::uint64_t generation = poll_generation_.load(std::memory_order_relaxed);
        if (seen_generation_ == generation)
        {
            return;
        }
        current_ = std::move(pending_);
        pending_ = PollState{};
        seen_generation_ = generation;
    }

    // Mutable because every reader is a const query, and rotation is caching,
    // not a state change the caller can observe.
    mutable PollState pending_;
    mutable PollState current_;
    mutable std::uint64_t seen_generation_ = 0;

    // The position is a state, not a change, so it is readable whatever the
    // poll cycle and does not rotate. first_mouse_ suppresses the meaningless
    // delta of the very first cursor event, which is a jump from (0,0).
    float pos_x_ = 0.0f;
    float pos_y_ = 0.0f;
    bool first_mouse_ = true;

    bool framebuffer_resized_ = false;

    static void mouse_callback(GLFWwindow* window, double xpos, double ypos)
    {
        Window* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
        if (!win)
            return;

        const float fx = static_cast<float>(xpos);
        const float fy = static_cast<float>(ypos);

        if (win->first_mouse_)
        {
            win->pos_x_ = fx;
            win->pos_y_ = fy;
            win->first_mouse_ = false;
        }

        // dy is positive upwards: a camera wants "how far did the look move",
        // not the screen's downward-positive y.
        win->pending_.dx += fx - win->pos_x_;
        win->pending_.dy += win->pos_y_ - fy;

        win->pos_x_ = fx;
        win->pos_y_ = fy;
    };

    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
    {
        Window* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
        if (win)
        {
            win->pending_.scroll_dx += static_cast<float>(xoffset);
            win->pending_.scroll_dy += static_cast<float>(yoffset);
        }
    };

    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
    {
        Window* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
        if (win && action == GLFW_PRESS)
        {
            win->pending_.keys.push_back(key);
        }
    };

    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
    {
        Window* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
        if (win && action == GLFW_PRESS)
        {
            win->pending_.buttons.push_back(button);
        }
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
// Drain the OS event queue and dispatch each event to the window the OS
// addressed it to. Deliberately NOT a method on Window: glfwPollEvents takes no
// window and the OS message queue is per-thread, so there is no such thing as
// "poll only this window's events". As a method it both lied about its scope —
// `window_a.poll_events()` reads as A's events but pumps everyone's — and forced
// a multi-window loop to keep a *closed* window alive just to have a receiver,
// leaving a frozen window on screen.
//
// The per-window distinction lives in the queries instead, where it is real:
// is_key_pressed, get_mouse_state, is_open and the framebuffer size are all read
// off one window's own state, which this dispatch is what updates.
inline std::expected<void, Error> poll_events()
{
    // No window means GLFW is not initialized (Window::create inits on the first
    // window, ~Window terminates on the last), so glfwPollEvents would set
    // GLFW_NOT_INITIALIZED and return — and before any window has ever existed
    // bazalt has not even installed its error callback yet, so the call would
    // vanish without a trace. A loop pumping events over windows that are all
    // gone is a bug; say so rather than silently doing nothing.
    if (Window::window_count_.load() == 0)
    {
        return std::unexpected(err_window(
            "No windows exist, so there is no event queue to drain. poll_events() "
            "dispatches OS events to the open windows; create a Window first, and "
            "stop pumping once the last one is closed."));
    }
    glfwPollEvents();

    // After the dispatch, so the generation counts *completed* cycles and the
    // events this call just delivered are the ones a query now reports.
    Window::poll_generation_.fetch_add(1, std::memory_order_relaxed);
    return {};
}
