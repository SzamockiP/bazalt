#pragma once
#include <volk.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#ifdef _WIN32
// For glfwGetWin32Window, which exclusive fullscreen needs to find the HMONITOR.
// volk.h above already pulled in <windows.h> through VK_USE_PLATFORM_WIN32_KHR.
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif
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

// One video mode a monitor can be set to. Width, height and refresh rate: the
// three a caller chooses between. GLFWvidmode also carries the per-channel bit
// depths, which no fullscreen decision has ever needed, so they stay out until
// something asks.
struct VideoMode
{
    int width = 0;
    int height = 0;
    int refresh_rate = 0;
};

// One monitor, as inert data — the Device shape, and for the same reason:
// choosing between monitors means seeing them, and what a caller sees must not
// be a live handle it can outlive.
//
// The GLFWmonitor* is kept, unlike Device's VkPhysicalDevice, because the two
// dangle differently. A VkPhysicalDevice dies with the instance list_devices
// destroys a moment later, so keeping it would be a guaranteed dangle. A
// GLFWmonitor* dies only when somebody unplugs that monitor, and every use
// re-scans the live list first — so the pointer is a key, never a dereference.
struct Monitor
{
    std::string name;
    // Position in the virtual desktop, which is what makes "the left one" a
    // question a caller can answer.
    int x = 0;
    int y = 0;
    // The mode the monitor is in right now, not the largest it can take.
    VideoMode current_mode;
    // Millimetres, as the OS reports them. Some drivers report zero, and that is
    // the OS talking rather than an error.
    int physical_width_mm = 0;
    int physical_height_mm = 0;
    float scale_x = 1.0f;
    float scale_y = 1.0f;
    bool primary = false;
    std::vector<VideoMode> video_modes;

    GLFWmonitor* handle = nullptr;
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
        WindowMode mode = WindowMode::WINDOWED,
        const Monitor* monitor = nullptr)
    {
        // Must be installed before glfwInit — otherwise the most common failure a
        // new user hits (no display, no drivers) reports "Failed to create window"
        // and throws away the one string that says why.
        glfwSetErrorCallback(glfw_error_callback);
        glfw_logger_ = logger;

        if (window_count_.fetch_add(1) == 0)
        {
            // glfwInit is idempotent, so this is a no-op when list_monitors()
            // already brought GLFW up to answer a question about displays.
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
                terminate_glfw_();
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
        glfwSetCharCallback(window->window_.get(), char_callback);

        // Start level with the world: a window created mid-loop must not rotate
        // an empty pending_ into current_ on its first query.
        window->seen_generation_ = poll_generation_.load(std::memory_order_relaxed);

        glfwSetInputMode(window->window_.get(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);

        // The requested width/height/position become what WINDOWED returns to,
        // whatever mode the window opens in. Then the mode goes on through
        // set_mode: one implementation, so opening fullscreen and switching to
        // fullscreen cannot drift apart — and monitor= inherits every rule
        // set_mode has about which modes it means anything for.
        window->save_windowed_geometry_();
        if (auto applied = window->set_mode(mode, monitor); !applied)
        {
            return std::unexpected(applied.error());
        }

        return window;
    };

    ~Window()
    {
        // The handle goes first, explicitly. A destructor body runs BEFORE the
        // members are destroyed, so leaving this to the unique_ptr put
        // glfwDestroyWindow after the glfwTerminate below — and on the last
        // window that is a call into a library that no longer exists. GLFW
        // reports it rather than crashing ("The GLFW library is not
        // initialized"), which is why it survived until somebody read the log
        // after closing a window (0.26).
        window_.reset();
        if (window_count_.fetch_sub(1) == 1)
        {
            terminate_glfw_();
        }
    };

    // Bring GLFW up without owning a window.
    //
    // Every other process-wide query — the clipboard, the gamepads — refuses
    // when no Window is alive, and monitor enumeration cannot: choosing which
    // display to open on happens BEFORE the first window exists. So it
    // initializes GLFW itself, the way list_devices() builds its own instance
    // rather than borrowing a Context's.
    //
    // Nothing shuts it down again if no window is ever created. glfwTerminate is
    // process-wide cleanup the OS does at exit anyway, and the alternative is a
    // second reference count beside window_count_ for a case nobody has.
    static std::expected<void, Error> ensure_glfw_ready()
    {
        glfwSetErrorCallback(glfw_error_callback);
        if (!glfwInit())
        {
            return std::unexpected(err_window(describe_glfw_failure(
                "Bazalt cannot start the window system. Usually there is no "
                "display attached, or the display drivers are missing")));
        }
        return {};
    }

    // The live GLFWmonitor* behind a Monitor value, or a refusal naming it.
    //
    // A Monitor is a copy the caller may have kept for minutes, and a monitor
    // can be unplugged in that time. So the pointer is treated as a key looked
    // up in the current list, never as something to dereference on trust: the
    // failure is a message about that display, not a crash inside GLFW.
    static std::expected<GLFWmonitor*, Error> live_monitor(const Monitor& monitor)
    {
        int count = 0;
        GLFWmonitor** monitors = glfwGetMonitors(&count);
        for (int i = 0; i < count; ++i)
        {
            if (monitors[i] == monitor.handle)
            {
                return monitors[i];
            }
        }
        return std::unexpected(err_window(
            std::format(
                "the monitor '{}' is no longer connected. Call bz.list_monitors() again to see "
                "what is.",
                monitor.name)));
    }

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

    // The characters typed during the last poll cycle, as UTF-8. Empty on almost
    // every frame.
    //
    // This is NOT the key edges seen from another angle, and the difference is
    // the whole reason the entry point exists. glfwSetCharCallback reports a
    // Unicode codepoint after the keyboard layout, the shift state, AltGr, the
    // dead keys and the IME have each had their say; is_key_pressed reports a
    // physical key. Nothing reconstructs the first from the second, which is why
    // every text field in every toolkit reads this stream instead.
    //
    // Same lifetime as the key edges: it belongs to one poll cycle, it reads the
    // same twice inside that cycle, and it does not consume.
    const std::string& text_input() const
    {
        rotate_();
        return current_.text;
    }

    void set_cursor_mode(int mode)
    {
        glfwSetInputMode(window_.get(), GLFW_CURSOR, mode);
    }

    // The pointer's shape, from the ten standard ones GLFW draws. Orthogonal to
    // set_cursor_mode, which decides whether the pointer is visible at all.
    //
    // The shape takes an int for the reason every other input query does: a
    // pybind enum converts through its value, so bz.Cursor.IBEAM and a bare GLFW
    // constant both arrive here.
    //
    // A platform may not have a given shape — Wayland lacks several — and
    // glfwCreateStandardCursor then returns null, which glfwSetCursor reads as
    // "the default arrow". That is the set_icon contract: a request, not a
    // guarantee, and a missing shape is not worth an error channel that every
    // caller would then have to handle for a cosmetic detail.
    void set_cursor(int shape)
    {
        glfwSetCursor(window_.get(), standard_cursor_(shape));
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
    std::expected<void, Error> set_mode(
        WindowMode mode,
        const Monitor* monitor_choice = nullptr,
        const VideoMode* video_mode_choice = nullptr)
    {
        // A repeat of the same mode is a no-op only when nothing else changed.
        // Moving a fullscreen window to another monitor asks for the same mode
        // and has real work to do.
        if (mode == mode_ && !monitor_choice && !video_mode_choice)
        {
            return {};
        }
        // Both extras are about taking over a monitor, so neither means anything
        // to a windowed mode. Refused rather than ignored: a call that quietly
        // does half of what it says is worse than one that says no.
        const bool fullscreen = mode == WindowMode::FULLSCREEN || mode == WindowMode::FULLSCREEN_WINDOWED;
        if (monitor_choice && !fullscreen)
        {
            return std::unexpected(err_window(
                "monitor= applies to FULLSCREEN and FULLSCREEN_WINDOWED only. A windowed "
                "window is moved with set_position(x, y)."));
        }
        // FULLSCREEN_WINDOWED is defined by NOT changing the video mode — that is
        // what keeps alt-tab and a second display behaving normally — so a video
        // mode there would be a request the mode cannot honour.
        if (video_mode_choice && mode != WindowMode::FULLSCREEN)
        {
            return std::unexpected(err_window(
                "video_mode= applies to FULLSCREEN only. FULLSCREEN_WINDOWED deliberately "
                "keeps the monitor's current mode, and a windowed window is resized with "
                "set_size(width, height)."));
        }

        GLFWmonitor* chosen = nullptr;
        if (monitor_choice)
        {
            auto live = live_monitor(*monitor_choice);
            if (!live)
            {
                return std::unexpected(live.error());
            }
            chosen = *live;
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
            // Without monitor= the window takes the monitor it overlaps most,
            // which is what it has always done and what a single-display machine
            // needs to know nothing about.
            GLFWmonitor* monitor = chosen ? chosen : current_monitor_();
            const GLFWvidmode* video_mode = monitor ? glfwGetVideoMode(monitor) : nullptr;
            if (!monitor || !video_mode)
            {
                return std::unexpected(
                    err_window(describe_glfw_failure("Cannot go fullscreen: no monitor reported a video mode")));
            }
            // A chosen video mode replaces the monitor's current one. Not checked
            // against the monitor's list: GLFW picks the closest match it has, so
            // a mode from another monitor lands on something sane instead of
            // failing, and refusing it would mean re-enumerating to say what the
            // driver is about to work out anyway.
            const VideoMode wanted = video_mode_choice
                                         ? *video_mode_choice
                                         : VideoMode{video_mode->width, video_mode->height, video_mode->refreshRate};

            if (mode == WindowMode::FULLSCREEN)
            {
                glfwSetWindowAttrib(win, GLFW_DECORATED, GLFW_TRUE);
                glfwSetWindowMonitor(win, monitor, 0, 0, wanted.width, wanted.height, wanted.refresh_rate);
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

        // Which display the window is on right now, for exclusive fullscreen
        // (0.25). Read per call rather than captured, because dragging a window
        // to the other display changes the answer, and the swapchain asks again
        // every time it is recreated.
#ifdef _WIN32
        sp.get_win32_monitor = [raw]() -> void*
        { return MonitorFromWindow(glfwGetWin32Window(raw), MONITOR_DEFAULTTONEAREST); };
#endif

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
        // Characters typed during this cycle, already UTF-8. Encoded in the
        // callback rather than kept as codepoints, because every reader of this
        // wants text: a vector<unsigned int> would make the binding layer do the
        // encoding and make C++ callers do it themselves.
        std::string text;
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

        // Both deltas point the way Vulkan's y does: +x right, +y DOWN. dy used
        // to be flipped here, on the argument that a camera wants "how far did
        // the look move" — but that made the mouse the one thing in bazalt whose
        // y disagreed with the clip space, the framebuffer rows and the cursor
        // position beside it. A first-person camera subtracts it, which is one
        // sign in the caller against a convention that had to be remembered
        // everywhere else (0.26).
        win->pending_.dx += fx - win->pos_x_;
        win->pending_.dy += fy - win->pos_y_;

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

    // A GLFWcursor belongs to the process rather than to a window — the same fact
    // that makes poll_events() and the clipboard free functions — so the ten
    // standard shapes are created once, on first use, and shared by every window.
    //
    // GLFW defines the ten constants contiguously from GLFW_ARROW_CURSOR, which is
    // what lets an array stand in for a map. A shape outside that range is not a
    // standard cursor, and null is what glfwSetCursor takes for "the default".
    static GLFWcursor* standard_cursor_(int shape)
    {
        const int index = shape - GLFW_ARROW_CURSOR;
        if (index < 0 || index >= static_cast<int>(cursors_.size()))
        {
            return nullptr;
        }
        if (!cursors_[index])
        {
            cursors_[index] = glfwCreateStandardCursor(shape);
        }
        return cursors_[index];
    }

    // glfwTerminate destroys every remaining cursor, so the cache holds dangling
    // pointers the moment the last window closes. Cleared beside the call rather
    // than in a destructor of its own, because the cache has no owner: it belongs
    // to the same process-wide lifetime glfwInit and glfwTerminate bracket.
    static void terminate_glfw_()
    {
        cursors_.fill(nullptr);
        glfwTerminate();
    }

    static inline std::array<GLFWcursor*, 10> cursors_{};

    // GLFW hands out one codepoint; the rest of the library speaks UTF-8, so the
    // conversion happens here, once, at the only place a codepoint exists.
    static void append_utf8(std::string& out, unsigned int codepoint)
    {
        if (codepoint < 0x80)
        {
            out.push_back(static_cast<char>(codepoint));
        }
        else if (codepoint < 0x800)
        {
            out.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        }
        else if (codepoint < 0x10000)
        {
            out.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        }
        else
        {
            out.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        }
    }

    static void char_callback(GLFWwindow* window, unsigned int codepoint)
    {
        Window* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
        if (win)
        {
            append_utf8(win->pending_.text, codepoint);
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

// The same dispatch, but it SLEEPS until there is something to dispatch.
//
// poll_events() was the only pump, so every program that does not animate — a
// viewer, a parameter editor, examples/08_pyqt_integration — burned a core
// spinning through frames identical to the last one. This is the one-line fix
// GLFW has had all along, and a free function for exactly the reasons above:
// glfwWaitEvents takes no window either.
//
// timeout is in seconds, and None waits indefinitely. Give it a number when
// something other than input has to wake the loop up — an animation that runs
// only sometimes, or a watchdog. Note that a hot-reload edit does NOT wake it:
// the watcher runs on its own thread and its result is applied from
// begin_frame(), so a program that only ever waits for input would not see the
// reload until the next click. Pass a timeout there.
inline std::expected<void, Error> wait_events(std::optional<double> timeout)
{
    if (Window::window_count_.load() == 0)
    {
        return std::unexpected(err_window(
            "No windows exist, so there is no event queue to wait on. wait_events() "
            "sleeps until an OS event arrives for an open window. Create a Window "
            "first, and stop pumping once the last one is closed."));
    }
    if (timeout.has_value())
    {
        glfwWaitEventsTimeout(*timeout);
    }
    else
    {
        glfwWaitEvents();
    }

    // Same order and the same meaning as poll_events(): the generation counts
    // completed cycles, so a per-cycle query reports what this call delivered.
    Window::poll_generation_.fetch_add(1, std::memory_order_relaxed);
    return {};
}

// Every monitor the OS reports, as inert data.
//
// A free function for the reason poll_events() is one: a monitor belongs to the
// process, and glfwGetMonitors takes no window. Unlike the clipboard and the
// gamepads it does NOT need a live Window, because its whole use is choosing
// where to open one — see ensure_glfw_ready.
//
// The primary monitor comes first, which is GLFW's own order and the answer a
// caller who does not care wants.
inline std::expected<std::vector<Monitor>, Error> list_monitors()
{
    if (auto ready = Window::ensure_glfw_ready(); !ready)
    {
        return std::unexpected(ready.error());
    }

    int count = 0;
    GLFWmonitor** monitors = glfwGetMonitors(&count);
    // An error rather than an empty list: with nothing to choose between there is
    // nothing this call can be used for, and every caller would have to write the
    // same check. GLFW reports no error of its own here, so there is none to add.
    if (!monitors || count == 0)
    {
        return std::unexpected(err_window("No monitor is connected"));
    }
    GLFWmonitor* primary = glfwGetPrimaryMonitor();

    std::vector<Monitor> out;
    out.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i)
    {
        Monitor monitor;
        monitor.handle = monitors[i];
        monitor.primary = monitors[i] == primary;
        if (const char* name = glfwGetMonitorName(monitors[i]))
        {
            monitor.name = name;
        }
        glfwGetMonitorPos(monitors[i], &monitor.x, &monitor.y);
        glfwGetMonitorPhysicalSize(monitors[i], &monitor.physical_width_mm, &monitor.physical_height_mm);
        glfwGetMonitorContentScale(monitors[i], &monitor.scale_x, &monitor.scale_y);
        if (const GLFWvidmode* current = glfwGetVideoMode(monitors[i]))
        {
            monitor.current_mode = {current->width, current->height, current->refreshRate};
        }

        int mode_count = 0;
        const GLFWvidmode* modes = glfwGetVideoModes(monitors[i], &mode_count);
        monitor.video_modes.reserve(static_cast<std::size_t>(mode_count));
        for (int m = 0; m < mode_count; ++m)
        {
            monitor.video_modes.push_back({modes[m].width, modes[m].height, modes[m].refreshRate});
        }
        out.push_back(std::move(monitor));
    }
    return out;
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

// The keyboard, the same way (0.23): the values ARE the GLFW ones, so the
// query methods keep their int signatures and the bare KEY_* module ints stay
// valid — an enum member converts through its value. D0..D9 are the top-row
// digits (an identifier cannot start with one); the keypad is KP_*. There is
// no LAST: it is GLFW's array-size sentinel, not a key, and the KEY_LAST int
// remains for anyone sizing an array.
enum class Key
{
    SPACE = GLFW_KEY_SPACE,
    APOSTROPHE = GLFW_KEY_APOSTROPHE,
    COMMA = GLFW_KEY_COMMA,
    MINUS = GLFW_KEY_MINUS,
    PERIOD = GLFW_KEY_PERIOD,
    SLASH = GLFW_KEY_SLASH,
    D0 = GLFW_KEY_0,
    D1 = GLFW_KEY_1,
    D2 = GLFW_KEY_2,
    D3 = GLFW_KEY_3,
    D4 = GLFW_KEY_4,
    D5 = GLFW_KEY_5,
    D6 = GLFW_KEY_6,
    D7 = GLFW_KEY_7,
    D8 = GLFW_KEY_8,
    D9 = GLFW_KEY_9,
    SEMICOLON = GLFW_KEY_SEMICOLON,
    EQUAL = GLFW_KEY_EQUAL,
    A = GLFW_KEY_A,
    B = GLFW_KEY_B,
    C = GLFW_KEY_C,
    D = GLFW_KEY_D,
    E = GLFW_KEY_E,
    F = GLFW_KEY_F,
    G = GLFW_KEY_G,
    H = GLFW_KEY_H,
    I = GLFW_KEY_I,
    J = GLFW_KEY_J,
    K = GLFW_KEY_K,
    L = GLFW_KEY_L,
    M = GLFW_KEY_M,
    N = GLFW_KEY_N,
    O = GLFW_KEY_O,
    P = GLFW_KEY_P,
    Q = GLFW_KEY_Q,
    R = GLFW_KEY_R,
    S = GLFW_KEY_S,
    T = GLFW_KEY_T,
    U = GLFW_KEY_U,
    V = GLFW_KEY_V,
    W = GLFW_KEY_W,
    X = GLFW_KEY_X,
    Y = GLFW_KEY_Y,
    Z = GLFW_KEY_Z,
    LEFT_BRACKET = GLFW_KEY_LEFT_BRACKET,
    BACKSLASH = GLFW_KEY_BACKSLASH,
    RIGHT_BRACKET = GLFW_KEY_RIGHT_BRACKET,
    GRAVE_ACCENT = GLFW_KEY_GRAVE_ACCENT,
    WORLD_1 = GLFW_KEY_WORLD_1,
    WORLD_2 = GLFW_KEY_WORLD_2,
    ESCAPE = GLFW_KEY_ESCAPE,
    ENTER = GLFW_KEY_ENTER,
    TAB = GLFW_KEY_TAB,
    BACKSPACE = GLFW_KEY_BACKSPACE,
    INSERT = GLFW_KEY_INSERT,
    // DEL, not DELETE: <windows.h> arrives through volk and defines DELETE as
    // a macro. The Python name is still DELETE — the binding names it.
    DEL = GLFW_KEY_DELETE,
    RIGHT = GLFW_KEY_RIGHT,
    LEFT = GLFW_KEY_LEFT,
    DOWN = GLFW_KEY_DOWN,
    UP = GLFW_KEY_UP,
    PAGE_UP = GLFW_KEY_PAGE_UP,
    PAGE_DOWN = GLFW_KEY_PAGE_DOWN,
    HOME = GLFW_KEY_HOME,
    END = GLFW_KEY_END,
    CAPS_LOCK = GLFW_KEY_CAPS_LOCK,
    SCROLL_LOCK = GLFW_KEY_SCROLL_LOCK,
    NUM_LOCK = GLFW_KEY_NUM_LOCK,
    PRINT_SCREEN = GLFW_KEY_PRINT_SCREEN,
    PAUSE = GLFW_KEY_PAUSE,
    F1 = GLFW_KEY_F1,
    F2 = GLFW_KEY_F2,
    F3 = GLFW_KEY_F3,
    F4 = GLFW_KEY_F4,
    F5 = GLFW_KEY_F5,
    F6 = GLFW_KEY_F6,
    F7 = GLFW_KEY_F7,
    F8 = GLFW_KEY_F8,
    F9 = GLFW_KEY_F9,
    F10 = GLFW_KEY_F10,
    F11 = GLFW_KEY_F11,
    F12 = GLFW_KEY_F12,
    F13 = GLFW_KEY_F13,
    F14 = GLFW_KEY_F14,
    F15 = GLFW_KEY_F15,
    F16 = GLFW_KEY_F16,
    F17 = GLFW_KEY_F17,
    F18 = GLFW_KEY_F18,
    F19 = GLFW_KEY_F19,
    F20 = GLFW_KEY_F20,
    F21 = GLFW_KEY_F21,
    F22 = GLFW_KEY_F22,
    F23 = GLFW_KEY_F23,
    F24 = GLFW_KEY_F24,
    F25 = GLFW_KEY_F25,
    KP_0 = GLFW_KEY_KP_0,
    KP_1 = GLFW_KEY_KP_1,
    KP_2 = GLFW_KEY_KP_2,
    KP_3 = GLFW_KEY_KP_3,
    KP_4 = GLFW_KEY_KP_4,
    KP_5 = GLFW_KEY_KP_5,
    KP_6 = GLFW_KEY_KP_6,
    KP_7 = GLFW_KEY_KP_7,
    KP_8 = GLFW_KEY_KP_8,
    KP_9 = GLFW_KEY_KP_9,
    KP_DECIMAL = GLFW_KEY_KP_DECIMAL,
    KP_DIVIDE = GLFW_KEY_KP_DIVIDE,
    KP_MULTIPLY = GLFW_KEY_KP_MULTIPLY,
    KP_SUBTRACT = GLFW_KEY_KP_SUBTRACT,
    KP_ADD = GLFW_KEY_KP_ADD,
    KP_ENTER = GLFW_KEY_KP_ENTER,
    KP_EQUAL = GLFW_KEY_KP_EQUAL,
    LEFT_SHIFT = GLFW_KEY_LEFT_SHIFT,
    LEFT_CONTROL = GLFW_KEY_LEFT_CONTROL,
    LEFT_ALT = GLFW_KEY_LEFT_ALT,
    LEFT_SUPER = GLFW_KEY_LEFT_SUPER,
    RIGHT_SHIFT = GLFW_KEY_RIGHT_SHIFT,
    RIGHT_CONTROL = GLFW_KEY_RIGHT_CONTROL,
    RIGHT_ALT = GLFW_KEY_RIGHT_ALT,
    RIGHT_SUPER = GLFW_KEY_RIGHT_SUPER,
    MENU = GLFW_KEY_MENU
};

enum class MouseButton
{
    LEFT = GLFW_MOUSE_BUTTON_LEFT,
    RIGHT = GLFW_MOUSE_BUTTON_RIGHT,
    MIDDLE = GLFW_MOUSE_BUTTON_MIDDLE,
    // GLFW reports eight buttons; the queries pass any of them through.
    BUTTON_4 = GLFW_MOUSE_BUTTON_4,
    BUTTON_5 = GLFW_MOUSE_BUTTON_5,
    BUTTON_6 = GLFW_MOUSE_BUTTON_6,
    BUTTON_7 = GLFW_MOUSE_BUTTON_7,
    BUTTON_8 = GLFW_MOUSE_BUTTON_8
};

enum class CursorMode
{
    NORMAL = GLFW_CURSOR_NORMAL,
    HIDDEN = GLFW_CURSOR_HIDDEN,
    DISABLED = GLFW_CURSOR_DISABLED
};

// What the pointer looks like, which is a different question from CursorMode:
// that one says whether the pointer is visible or captured, this one says what
// it draws while it is visible. Two questions, two verbs, and no rules for
// combining them.
//
// The values ARE the GLFW ints, by the rule Key and GamepadButton already
// follow: the two cannot drift, and a bare GLFW constant stays valid.
enum class Cursor
{
    ARROW = GLFW_ARROW_CURSOR,
    // A text caret. What a text field shows, so it arrives with text_input().
    IBEAM = GLFW_IBEAM_CURSOR,
    CROSSHAIR = GLFW_CROSSHAIR_CURSOR,
    POINTING_HAND = GLFW_POINTING_HAND_CURSOR,
    RESIZE_EW = GLFW_RESIZE_EW_CURSOR,
    RESIZE_NS = GLFW_RESIZE_NS_CURSOR,
    RESIZE_NWSE = GLFW_RESIZE_NWSE_CURSOR,
    RESIZE_NESW = GLFW_RESIZE_NESW_CURSOR,
    RESIZE_ALL = GLFW_RESIZE_ALL_CURSOR,
    NOT_ALLOWED = GLFW_NOT_ALLOWED_CURSOR
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

    // Went down since the last poll cycle — the edge, where button() is the
    // level. Same pair as was_key_pressed beside is_key_pressed, and the same
    // reason: without it every caller keeps its own "was it down last frame"
    // array.
    bool was_button_pressed(GamepadButton b) const
    {
        const auto i = static_cast<std::size_t>(b);
        return i < buttons.size() && buttons[i] != 0 && previous[i] == 0;
    }

    // The state this pad was in one poll cycle ago. Carried in the snapshot
    // rather than looked up by the query, so a Gamepad stays a value that
    // answers every question about one moment out of its own fields.
    std::array<unsigned char, 15> previous{};
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

// What each pad's buttons read when this cycle started, so an edge has something
// to compare against.
//
// Process-wide, because the pad is: 0.16 refused a registry of live *windows* to
// hang a global on, and this needs no such thing — the pads are a fixed set of
// slots, and the per-cycle counter it rotates on has existed since 0.16. Not
// synchronized, like the per-window PollState it mirrors, because GLFW's input
// belongs to the main thread.
struct GamepadHistory
{
    std::array<unsigned char, 15> previous{};
    std::array<unsigned char, 15> current{};
    // 0 is a real generation, so "never read" needs its own flag rather than a
    // sentinel: without it the first read of the very first cycle would roll a
    // zeroed `current` into `previous` and report every held button as an edge.
    bool seen = false;
    std::uint64_t generation = 0;
};

inline std::array<GamepadHistory, GLFW_JOYSTICK_LAST + 1> gamepad_history_{};

// Read one gamepad, or nothing when that slot is empty.
//
// A free function for the reason poll_events() is one: a pad belongs to the
// process, not to a window, and glfwGetGamepadState takes a joystick id and no
// window. The state itself is refreshed by poll_events(), so this is a read of
// what the last pump saw.
//
// The edges rotate the same way the window's do: the first read in a new poll
// cycle rolls what it read last time into `previous`, and every later read in
// that cycle answers identically. Reader-driven, so it does not consume — the
// 0.16 rule, and the reason was_button_pressed twice in one frame is not a trap.
//
// What that costs, and it is the honest half: GLFW has no gamepad callback, so a
// press and a release inside a cycle nobody read are both invisible. A key edge
// cannot be missed that way, because the key callback records it whether or not
// anybody asks. Reading each pad every frame is what closes the gap.
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
        // An empty slot forgets its history, so a pad that is unplugged and
        // plugged back in starts level instead of reporting whatever it held
        // before as an edge.
        if (index >= 0 && index < static_cast<int>(gamepad_history_.size()))
        {
            gamepad_history_[static_cast<std::size_t>(index)] = GamepadHistory{};
        }
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

    // Two conversions, and they answer the same question: what did the HAND do?
    //
    // A trigger becomes 0..1. GLFW reports -1 released to +1 pressed, which is the
    // hardware talking: "how far in is the trigger" is a 0..1 question and every
    // caller would write the same line. A stick keeps -1..1, because that IS the
    // question there.
    for (auto trigger : {GamepadAxis::LEFT_TRIGGER, GamepadAxis::RIGHT_TRIGGER})
    {
        float& value = pad.axes[static_cast<std::size_t>(trigger)];
        value = (value + 1.0f) * 0.5f;
    }
    // A stick pushed UP reads +1. GLFW reports the opposite, and it is right to:
    // its Y is screen space, where down is positive, and that is what a cursor
    // position means. A stick has no screen to agree with — it is a thing in a
    // hand, and every caller would negate it.
    //
    // The cost is that window.get_mouse_state().dy and GamepadAxis.LEFT_Y now
    // disagree about which way is positive, and that is the honest answer rather
    // than an oversight: the mouse delta IS a screen measurement and the stick is
    // not. A camera driven by both negates one of them, and it would have had to
    // negate one of them either way.
    for (auto stick : {GamepadAxis::LEFT_Y, GamepadAxis::RIGHT_Y})
    {
        float& value = pad.axes[static_cast<std::size_t>(stick)];
        value = -value;
    }
    // Deadzone on the sticks only. A trigger rests at one end of its range, so a
    // dead zone around zero would eat the first part of the pull.
    for (auto stick : {GamepadAxis::LEFT_X, GamepadAxis::LEFT_Y, GamepadAxis::RIGHT_X, GamepadAxis::RIGHT_Y})
    {
        float& value = pad.axes[static_cast<std::size_t>(stick)];
        value = apply_deadzone(value, deadzone);
    }

    // The buttons, not the axes: an axis edge is a threshold the caller picks,
    // and bazalt has no business picking it.
    if (index >= 0 && index < static_cast<int>(gamepad_history_.size()))
    {
        GamepadHistory& history = gamepad_history_[static_cast<std::size_t>(index)];
        const std::uint64_t generation = Window::poll_generation_.load(std::memory_order_relaxed);
        if (!history.seen || history.generation != generation)
        {
            history.previous = history.seen ? history.current : pad.buttons;
            history.current = pad.buttons;
            history.generation = generation;
            history.seen = true;
        }
        pad.previous = history.previous;
    }
    return pad;
}
