#pragma once
#include <volk.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <array>
#include <string>
#include <stdexcept>
#include <memory>
#include <expected>
#include <optional>
#include <utility>
#include <vector>
#include <algorithm>
#include <cmath>
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
                // Framed around what the user hit, not around the C library bazalt
                // happens to use. describe_glfw_failure appends GLFW's own text when
                // it has any, and that detail is where the library name belongs.
                return std::unexpected(err_window(describe_glfw_failure(
                    "Bazalt cannot start the window system. Usually there is no "
                    "display attached, or the display drivers are missing")));
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
            return std::unexpected(err_window(describe_glfw_failure("Bazalt cannot create the window")));
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
        glfwSetDropCallback(window->window_.get(), drop_callback);

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

    // Paths dropped onto the window during the last poll cycle, empty most frames.
    // Reads like was_key_pressed: idempotent within a cycle, and it does not
    // consume, so two readers in one frame both see the drop.
    const std::vector<std::string>& dropped_files() const
    {
        rotate_();
        return current_.dropped;
    }

    void set_cursor_mode(int mode)
    {
        glfwSetInputMode(window_.get(), GLFW_CURSOR, mode);
    }

    // Move the cursor, and make sure the move does not read as the user moving it.
    //
    // A warp is not mouse movement, and mouse_callback cannot tell the difference:
    // it accumulates `new position - pos_x_`, so a warp would inject a delta the
    // size of the jump. Re-arming first_mouse_ makes the next real cursor event
    // adopt its position with no delta, which is the mechanism already in place for
    // the meaningless jump from (0,0) at startup.
    //
    // That suppression is what makes the hidden-cursor recentring pattern work
    // (warp to the centre, then read the delta from the centre). It does NOT
    // combine with CURSOR_DISABLED: that mode already hands out unbounded virtual
    // motion and recentres itself, so warping every frame there cancels every
    // frame's delta and the camera stops turning. Pick one — disabled and no warp,
    // or hidden and warp.
    void set_cursor_position(double x, double y)
    {
        glfwSetCursorPos(window_.get(), x, y);
        pos_x_ = static_cast<float>(x);
        pos_y_ = static_cast<float>(y);
        first_mouse_ = true;
    }

    // The window's icon in the task bar and title bar, as RGBA8 pixels. Empty
    // pixels restore the system default.
    //
    // The platform may refuse: macOS takes its icon from the bundle and Wayland
    // from the desktop file, and GLFW reports that through the error callback
    // rather than failing here. Same contract as set_opacity — a request, not a
    // guarantee.
    void set_icon(const std::vector<std::uint8_t>& rgba, int width, int height)
    {
        if (rgba.empty())
        {
            glfwSetWindowIcon(window_.get(), 0, nullptr);
            return;
        }
        // const_cast because GLFWimage takes a non-const pointer and glfwSetWindowIcon
        // copies the pixels before returning; it never writes through it.
        GLFWimage image{.width = width, .height = height, .pixels = const_cast<unsigned char*>(rgba.data())};
        glfwSetWindowIcon(window_.get(), 1, &image);
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
        // Files dropped onto the window during this cycle, as UTF-8 paths. It
        // belongs here and not in a field of its own for the 0.16 reason: a drop
        // is a change that expires with the cycle, exactly like a key edge or a
        // scroll notch, so it shares the one rotation mechanism rather than
        // inventing a second lifetime to remember.
        std::vector<std::string> dropped;
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

    static void drop_callback(GLFWwindow* window, int count, const char** paths)
    {
        Window* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
        if (!win)
            return;

        // GLFW owns `paths` only for the duration of this call, so the strings are
        // copied, not referenced.
        for (int i = 0; i < count; ++i)
        {
            win->pending_.dropped.emplace_back(paths[i]);
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
            "dispatches OS events to the open windows. Create a Window first, and "
            "stop pumping once the last one is closed."));
    }
    glfwPollEvents();

    // After the dispatch, so the generation counts *completed* cycles and the
    // events this call just delivered are the ones a query now reports.
    Window::poll_generation_.fetch_add(1, std::memory_order_relaxed);
    return {};
}

// The system clipboard. Free functions, not Window methods, for the same reason
// poll_events() is a free function: glfwGetClipboardString and
// glfwSetClipboardString take no window (GLFW accepts NULL and the clipboard is
// per process, not per window), so a method would invent a per-window distinction
// that does not exist and read as "this window's clipboard".
//
// Both need GLFW initialized, which means at least one live Window — the same
// precondition and the same message shape as poll_events().
inline std::expected<std::string, Error> get_clipboard()
{
    if (Window::window_count_.load() == 0)
    {
        return std::unexpected(err_window(
            "The clipboard needs a window: GLFW is initialized with the first Window "
            "and shut down with the last, and the clipboard belongs to the process, "
            "not to any one window."));
    }
    // Null on an empty clipboard, or when it holds something that is not text (an
    // image, a file list). Neither is an error — "no text to paste" is an answer.
    const char* text = glfwGetClipboardString(nullptr);
    return text ? std::string(text) : std::string();
}

inline std::expected<void, Error> set_clipboard(const std::string& text)
{
    if (Window::window_count_.load() == 0)
    {
        return std::unexpected(err_window(
            "The clipboard needs a window: GLFW is initialized with the first Window "
            "and shut down with the last, and the clipboard belongs to the process, "
            "not to any one window."));
    }
    glfwSetClipboardString(nullptr, text.c_str());
    return {};
}

// ── Gamepads ─────────────────────────────────────────────────────────────────
//
// Named after what a hand does, in GLFW's own layout: every pad the mapping
// database knows is presented as this one shape, whatever the hardware reports
// underneath. That is the same argument Features.hpp makes — the caller says
// which control they mean, and the library resolves what this device calls it.
//
// The values are GLFW's, so the enums are a rename rather than a translation
// table that could drift.
enum class GamepadButton
{
    A = GLFW_GAMEPAD_BUTTON_A,
    B = GLFW_GAMEPAD_BUTTON_B,
    X = GLFW_GAMEPAD_BUTTON_X,
    Y = GLFW_GAMEPAD_BUTTON_Y,
    LEFT_BUMPER = GLFW_GAMEPAD_BUTTON_LEFT_BUMPER,
    RIGHT_BUMPER = GLFW_GAMEPAD_BUTTON_RIGHT_BUMPER,
    BACK = GLFW_GAMEPAD_BUTTON_BACK,
    START = GLFW_GAMEPAD_BUTTON_START,
    GUIDE = GLFW_GAMEPAD_BUTTON_GUIDE,
    LEFT_THUMB = GLFW_GAMEPAD_BUTTON_LEFT_THUMB,
    RIGHT_THUMB = GLFW_GAMEPAD_BUTTON_RIGHT_THUMB,
    DPAD_UP = GLFW_GAMEPAD_BUTTON_DPAD_UP,
    DPAD_RIGHT = GLFW_GAMEPAD_BUTTON_DPAD_RIGHT,
    DPAD_DOWN = GLFW_GAMEPAD_BUTTON_DPAD_DOWN,
    DPAD_LEFT = GLFW_GAMEPAD_BUTTON_DPAD_LEFT
};

enum class GamepadAxis
{
    LEFT_X = GLFW_GAMEPAD_AXIS_LEFT_X,
    LEFT_Y = GLFW_GAMEPAD_AXIS_LEFT_Y,
    RIGHT_X = GLFW_GAMEPAD_AXIS_RIGHT_X,
    RIGHT_Y = GLFW_GAMEPAD_AXIS_RIGHT_Y,
    LEFT_TRIGGER = GLFW_GAMEPAD_AXIS_LEFT_TRIGGER,
    RIGHT_TRIGGER = GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER
};

// One reading of one pad. A snapshot by value, not a live handle: the state is
// what the last poll_events() left behind, and holding it means a value that
// cannot change halfway through the frame that is reading it.
struct Gamepad
{
    int index = 0;
    std::string name;
    std::array<float, 6> axes{};
    std::array<unsigned char, 15> buttons{};

    float axis(GamepadAxis a) const
    {
        const auto i = static_cast<std::size_t>(a);
        return i < axes.size() ? axes[i] : 0.0f;
    }

    // Not std::unreachable on an out-of-range value: a pybind enum accepts any
    // int, so a forged one has to land somewhere safe.
    bool button(GamepadButton b) const
    {
        const auto i = static_cast<std::size_t>(b);
        return i < buttons.size() && buttons[i] != 0;
    }
};

// A stick that reads 0.03 when nobody is touching it is what real hardware does,
// so the knob exists. Scaled rather than clipped, so the value stays continuous
// as the stick leaves the dead zone instead of jumping to `deadzone`.
//
// Per axis, which makes the dead area a square rather than a circle. A radial
// dead zone needs the two axes of a stick paired up, and nothing has asked for
// the difference.
inline constexpr float apply_deadzone(float value, float deadzone)
{
    const float magnitude = value < 0.0f ? -value : value;
    if (magnitude <= deadzone)
    {
        return 0.0f;
    }
    const float scaled = (magnitude - deadzone) / (1.0f - deadzone);
    return value < 0.0f ? -scaled : scaled;
}

// The one piece of arithmetic here, and no test can reach it without a stick in
// somebody's hand — so it is checked at compile time instead. Full deflection
// stays full deflection (a dead zone that shortened the range would be felt), the
// dead area is zero on both sides, and the value is continuous across the edge.
static_assert(apply_deadzone(1.0f, 0.25f) == 1.0f);
static_assert(apply_deadzone(-1.0f, 0.25f) == -1.0f);
static_assert(apply_deadzone(0.2f, 0.25f) == 0.0f);
static_assert(apply_deadzone(-0.2f, 0.25f) == 0.0f);
static_assert(apply_deadzone(0.25f, 0.25f) == 0.0f);
static_assert(apply_deadzone(0.5f, 0.0f) == 0.5f);
static_assert(apply_deadzone(0.625f, 0.25f) == 0.5f);

// Read one gamepad, or nothing when that slot is empty.
//
// A free function for the reason poll_events() is one: a pad belongs to the
// process, not to a window, and glfwGetGamepadState takes a joystick id and no
// window. The state itself is refreshed by poll_events(), so this is a read of
// what the last pump saw.
//
// Level state only — which buttons are down now. The edge queries
// (was_key_pressed and friends) rotate on a per-window generation counter, and a
// pad has no window to hang that on; 0.16 rejected a process-wide registry of
// live windows for exactly this, and an edge here would need the global that
// decision refused.
// The index and the deadzone are checked in the binding layer and raise
// ValueError there: both are values outside a fixed range in the signature, which
// is the half of the 0.20 rule that does not belong to the BazaltError hierarchy.
// An out-of-range index reaching here is answered by GLFW with "no such pad".
inline std::expected<std::optional<Gamepad>, Error> get_gamepad(int index, float deadzone)
{
    if (Window::window_count_.load() == 0)
    {
        return std::unexpected(err_window(
            "Reading a gamepad needs a window: GLFW is initialized with the first Window "
            "and shut down with the last, and the pads belong to the process, not to any "
            "one window."));
    }

    GLFWgamepadstate state{};
    // False for an empty slot AND for a stick GLFW has no mapping for, and the
    // answer is the same either way: bazalt cannot present it as A/B/X/Y, so
    // there is no gamepad here as far as this API is concerned.
    if (!glfwGetGamepadState(index, &state))
    {
        return std::nullopt;
    }

    Gamepad pad;
    pad.index = index;
    if (const char* name = glfwGetGamepadName(index))
    {
        pad.name = name;
    }
    for (std::size_t i = 0; i < pad.axes.size(); ++i)
    {
        pad.axes[i] = state.axes[i];
    }
    for (std::size_t i = 0; i < pad.buttons.size(); ++i)
    {
        pad.buttons[i] = state.buttons[i];
    }

    // GLFW reports a trigger as -1 released to +1 pressed, which is the hardware
    // talking rather than the hand: "how far in is the trigger" is a 0..1
    // question, and every caller would write the same conversion. The sticks keep
    // their -1..1, because that IS the question there.
    for (auto trigger : {GamepadAxis::LEFT_TRIGGER, GamepadAxis::RIGHT_TRIGGER})
    {
        float& value = pad.axes[static_cast<std::size_t>(trigger)];
        value = (value + 1.0f) * 0.5f;
    }
    // Deadzone on the sticks only. A trigger rests at one end of its range, so a
    // dead zone around zero would eat the first part of the pull.
    for (auto stick : {GamepadAxis::LEFT_X, GamepadAxis::LEFT_Y, GamepadAxis::RIGHT_X, GamepadAxis::RIGHT_Y})
    {
        float& value = pad.axes[static_cast<std::size_t>(stick)];
        value = apply_deadzone(value, deadzone);
    }
    return pad;
}
