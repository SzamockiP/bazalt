#include "window.hpp"

#ifdef _WIN32
// For glfwGetWin32Window, which exclusive fullscreen needs to find the HMONITOR.
// volk.h — pulled in by window.hpp above — already brought in <windows.h>
// through VK_USE_PLATFORM_WIN32_KHR.
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif

#include <format>
#include <string_view>

std::expected<std::unique_ptr<Window>, Error> Window::create(
    int width,
    int height,
    const std::string& title,
    std::shared_ptr<Logger> logger,
    WindowMode mode,
    const Monitor* monitor)
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
}

Window::~Window()
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
}

std::expected<void, Error> Window::ensure_glfw_ready()
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

std::expected<GLFWmonitor*, Error> Window::live_monitor(const Monitor& monitor)
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

void Window::set_cursor_mode(int mode)
{
    glfwSetInputMode(window_.get(), GLFW_CURSOR, mode);
}

void Window::set_cursor(int shape)
{
    glfwSetCursor(window_.get(), standard_cursor_(shape));
}

void Window::set_cursor_position(double x, double y)
{
    glfwSetCursorPos(window_.get(), x, y);
    pos_x_ = static_cast<float>(x);
    pos_y_ = static_cast<float>(y);
    first_mouse_ = true;
}

void Window::set_icon(const std::vector<std::uint8_t>& rgba, int width, int height)
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

MouseState Window::get_mouse_state() const
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

std::expected<void, Error> Window::set_mode(
    WindowMode mode,
    const Monitor* monitor_choice,
    const VideoMode* video_mode_choice)
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

std::pair<int, int> Window::get_position() const
{
    int x = 0;
    int y = 0;
    glfwGetWindowPos(window_.get(), &x, &y);
    return {x, y};
}

std::pair<float, float> Window::get_content_scale() const
{
    float x = 1.0f;
    float y = 1.0f;
    glfwGetWindowContentScale(window_.get(), &x, &y);
    return {x, y};
}

SurfaceProvider Window::get_surface_provider()
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

Window::Window(int width, int height, const std::string& title)
    : width_(width),
      height_(height),
      title_(title)
{
}

void Window::glfw_error_callback(int error_code, const char* description)
{
    last_glfw_error_ = description ? std::string(description) : ("GLFW error " + std::to_string(error_code));

    if (auto logger = glfw_logger_.lock())
    {
        logger->log(Severity::Error, Source::Window, last_glfw_error_);
    }
}

std::string Window::describe_glfw_failure(std::string_view what)
{
    std::string detail = std::move(last_glfw_error_);
    last_glfw_error_.clear();

    if (detail.empty())
    {
        return std::string(what);
    }
    return std::string(what) + ": " + detail;
}

void Window::save_windowed_geometry_()
{
    glfwGetWindowPos(window_.get(), &saved_x_, &saved_y_);
    glfwGetWindowSize(window_.get(), &saved_width_, &saved_height_);
}

GLFWmonitor* Window::current_monitor_() const
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

void Window::rotate_() const
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

void Window::mouse_callback(GLFWwindow* window, double xpos, double ypos)
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
}

void Window::scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    Window* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (win)
    {
        win->pending_.scroll_dx += static_cast<float>(xoffset);
        win->pending_.scroll_dy += static_cast<float>(yoffset);
    }
}

void Window::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    Window* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (win && action == GLFW_PRESS)
    {
        win->pending_.keys.push_back(key);
    }
}

void Window::mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    Window* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (win && action == GLFW_PRESS)
    {
        win->pending_.buttons.push_back(button);
    }
}

void Window::framebuffer_resize_callback(GLFWwindow* window, int width, int height)
{
    Window* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (win)
    {
        win->framebuffer_resized_ = true;
    }
}

GLFWcursor* Window::standard_cursor_(int shape)
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

void Window::terminate_glfw_()
{
    cursors_.fill(nullptr);
    glfwTerminate();
}

void Window::append_utf8(std::string& out, unsigned int codepoint)
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

void Window::char_callback(GLFWwindow* window, unsigned int codepoint)
{
    Window* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (win)
    {
        append_utf8(win->pending_.text, codepoint);
    }
}

void Window::drop_callback(GLFWwindow* window, int count, const char** paths)
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
}

std::expected<void, Error> poll_events()
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

std::expected<void, Error> wait_events(std::optional<double> timeout)
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

std::expected<std::vector<Monitor>, Error> list_monitors()
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

std::expected<std::string, Error> get_clipboard()
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

std::expected<void, Error> set_clipboard(const std::string& text)
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

std::expected<std::optional<Gamepad>, Error> get_gamepad(int index, float deadzone)
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
