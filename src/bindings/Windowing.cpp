#include "Bindings.hpp"

void bind_windowing(py::module_& m)
{
    py::class_<LogMessage>(m, "LogMessage")
        .def_readonly("severity", &LogMessage::severity)
        .def_readonly("source", &LogMessage::source)
        .def_readonly("text", &LogMessage::text)
        .def(
            "__str__",
            [](const LogMessage& msg) { return std::format("{}: {}", severity_name(msg.severity), msg.text); })
        .def(
            "__repr__",
            [](const LogMessage& msg)
            { return std::format("<LogMessage {} '{}'>", severity_name(msg.severity), msg.text); });

    py::class_<MouseState>(m, "MouseState")
        .def_readonly("x", &MouseState::x)
        .def_readonly("y", &MouseState::y)
        .def_readonly("dx", &MouseState::dx)
        .def_readonly("dy", &MouseState::dy)
        .def_readonly("scroll_dx", &MouseState::scroll_dx)
        .def_readonly("scroll_dy", &MouseState::scroll_dy);

    py::class_<Window>(m, "Window")
        .def(
            py::init(
                [](int width, int height, const std::string& title, std::shared_ptr<Logger> logger, WindowMode mode)
                {
                    // Window used to have no way to reach a Logger at all, so GLFW's own
                    // diagnostics went nowhere.
                    return unwrap(Window::create(width, height, title, logger, mode), logger.get());
                }),
            py::arg("width"),
            py::arg("height"),
            py::arg("title"),
            py::arg("logger") = py::none(),
            py::arg("mode") = WindowMode::WINDOWED)
        .def("is_open", &Window::is_open)
        .def("is_key_pressed", &Window::is_key_pressed, py::arg("key"))
        .def("is_mouse_button_pressed", &Window::is_mouse_button_pressed, py::arg("button"))
        .def("was_key_pressed", &Window::was_key_pressed, py::arg("key"))
        .def("was_mouse_button_pressed", &Window::was_mouse_button_pressed, py::arg("button"))
        .def("set_cursor_mode", &Window::set_cursor_mode, py::arg("mode"))
        .def("get_mouse_state", &Window::get_mouse_state)
        .def("set_title", &Window::set_title, py::arg("title"))
        .def(
            "set_mode",
            // nullptr logger: GLFW's own error callback has already logged the
            // reason through the Window's Logger, so passing it here would say
            // the same thing twice.
            [](Window& self, WindowMode mode) { unwrap(self.set_mode(mode), nullptr); },
            py::arg("mode"))
        .def("set_size", &Window::set_size, py::arg("width"), py::arg("height"))
        .def("set_position", &Window::set_position, py::arg("x"), py::arg("y"))
        .def("set_cursor_position", &Window::set_cursor_position, py::arg("x"), py::arg("y"))
        .def(
            "dropped_files",
            // Copied into a Python list rather than returned by reference: the
            // vector is rotated out from under the caller on the next poll cycle.
            [](const Window& self) { return py::cast(self.dropped_files()); })
        .def(
            "set_icon",
            [](Window& self, py::object icon)
            {
                if (icon.is_none())
                {
                    self.set_icon({}, 0, 0);
                    return;
                }
                // Validated here, in the binding, for the same reason create_image
                // validates here: this is a user error about the shape of a Python
                // object, the GIL is held, and raise_error is legal.
                auto array = icon.cast<py::array>();
                // Compared against the dtype object, like image.update does, rather
                // than against a kind character: numpy spells uint8's kind 'u' and
                // its char code 'B', and a hand-written check picks the wrong one.
                if (!array.dtype().is(py::dtype("uint8")))
                {
                    raise_error(err_window(
                        std::format(
                            "set_icon needs an RGBA8 array of dtype uint8, not {}. Convert it with "
                            "arr.astype(np.uint8)",
                            py::str(array.dtype()).cast<std::string>())));
                }
                // Two conditions, two messages: "wrong number of dimensions" and
                // "no alpha channel" are different mistakes, and one message
                // covering both names neither.
                if (array.ndim() != 3)
                {
                    raise_error(err_window(
                        std::format(
                            "set_icon needs a (height, width, 4) RGBA array, got {} dimensions", array.ndim())));
                }
                if (array.shape(2) != 4)
                {
                    raise_error(err_window(
                        std::format(
                            "set_icon needs 4 channels (RGBA), got {}. An icon has an alpha channel", array.shape(2))));
                }
                // memcpy ignores strides, so a view like arr[::2] or arr.T would
                // copy other bytes. The 0.4 rule, applied again.
                if (!(array.flags() & py::array::c_style))
                {
                    raise_error(err_window(
                        "set_icon: the array must be C-contiguous (a strided view like arr.T "
                        "or arr[::2] would copy other bytes). Use numpy.ascontiguousarray(a)."));
                }
                const auto height = static_cast<int>(array.shape(0));
                const auto width = static_cast<int>(array.shape(1));
                const auto* bytes = static_cast<const std::uint8_t*>(array.data());
                self.set_icon(std::vector<std::uint8_t>(bytes, bytes + array.nbytes()), width, height);
            },
            py::arg("icon"))
        .def("set_resizable", &Window::set_resizable, py::arg("enable"))
        .def("set_always_on_top", &Window::set_always_on_top, py::arg("enable"))
        .def("set_opacity", &Window::set_opacity, py::arg("opacity"))
        .def_property_readonly("mode", &Window::mode)
        .def_property_readonly("position", &Window::get_position)
        .def_property_readonly("resizable", &Window::is_resizable)
        .def_property_readonly("always_on_top", &Window::is_always_on_top)
        .def_property_readonly("opacity", &Window::get_opacity)
        .def_property_readonly("content_scale", &Window::get_content_scale)
        .def_property_readonly("width", &Window::get_width)
        .def_property_readonly("height", &Window::get_height);

    // ── Logger ──
    py::class_<Logger, std::shared_ptr<Logger>>(m, "Logger")
        .def(py::init<Severity>(), py::arg("min_severity") = Severity::Warning)
        // One callback receiving a structured LogMessage, not on_error/on_warning/
        // on_info. Three callbacks would be three ways to do one thing, and the old
        // on_error was a lie anyway — it received INFO and WARNING alike.
        .def(
            "on_message",
            [](Logger& self, py::function callback)
            {
                self.register_callback(callback);
                return callback; // returned so it works as a decorator
            },
            py::arg("callback"))
        .def(
            "log",
            [](Logger& self, const std::string& text, Severity severity, Source source)
            { self.log(severity, source, text); },
            py::arg("text"),
            py::arg("severity") = Severity::Info,
            py::arg("source") = Source::General)
        // Delivery is async; without flush(), asserting "no errors happened" only
        // asserts "none had arrived yet".
        .def("flush", &Logger::flush)
        .def_property("min_severity", &Logger::min_severity, &Logger::set_min_severity);

    // ── Context ──
    // ── Device ──
    // Inert data, not a live handle: see Device.hpp on why a VkPhysicalDevice
    // could not survive the enumeration that produced it.
    py::class_<Device>(m, "Device")
        .def_readonly("name", &Device::name)
        .def_readonly("type", &Device::type)
        .def_property_readonly("api_version", [](const Device& d) { return api_version_string(d.api_version); })
        // Megabytes rather than bytes: the number is read by a human choosing a
        // card, and "8188" beats "8584495104".
        .def_property_readonly(
            "memory_mb", [](const Device& d) { return static_cast<std::uint64_t>(d.memory_bytes / (1024 * 1024)); })
        .def("supports", &Device::supports, py::arg("feature"))
        .def(
            "__repr__",
            [](const Device& d)
            {
                return std::format("<bazalt.Device '{}' ({}, {} MB)>", d.name, d.type, d.memory_bytes / (1024 * 1024));
            });

    // Free function, not a Window method: GLFW's event queue is process-wide.
    m.def(
        "poll_events",
        []() { unwrap(poll_events(), nullptr); },
        "Drain the OS event queue and dispatch each event to the window it was\n"
        "addressed to. One call services every window. The per-window distinction\n"
        "lives in the queries (is_key_pressed, is_open, renderer.acquire).\n"
        "Raises WindowError when no window exists.");

    // The sleeping half of the same pump. The GIL goes, because the whole point
    // is that this call blocks — holding it would freeze every other Python
    // thread for as long as the user does not move the mouse.
    m.def(
        "wait_events",
        [](std::optional<double> timeout)
        {
            // ValueError, not WindowError: nothing has to be asked of a window to
            // know that a negative number of seconds is not a duration. See
            // "Which exception a user error gets" in DESIGN.md.
            if (timeout.has_value() && !(*timeout >= 0.0))
            {
                throw py::value_error(
                    std::format("wait_events(timeout={}): the timeout is in seconds and cannot be negative", *timeout));
            }
            std::expected<void, Error> r;
            {
                py::gil_scoped_release release;
                r = wait_events(timeout);
            }
            unwrap(std::move(r), nullptr);
        },
        py::arg("timeout") = py::none(),
        "Sleep until an OS event arrives, then dispatch it like poll_events().\n"
        "timeout is in seconds; None waits indefinitely. Use it for a program that\n"
        "only redraws on input — with poll_events as the only pump, such a program\n"
        "spins a CPU core. Raises WindowError when no window exists.");

    // Free functions for the same reason poll_events is one: the clipboard belongs
    // to the process and the GLFW calls take no window.
    m.def(
        "get_clipboard",
        []() { return unwrap(get_clipboard(), nullptr); },
        "The system clipboard as text, or an empty string when it holds nothing or\n"
        "holds something that is not text. Needs at least one live Window, because\n"
        "GLFW is initialized with the first one.");

    m.def(
        "set_clipboard",
        [](const std::string& text) { unwrap(set_clipboard(text), nullptr); },
        py::arg("text"),
        "Put text on the system clipboard. Needs at least one live Window.");

    m.def(
        "list_devices",
        []()
        {
            auto devices = list_devices();
            return unwrap(std::move(devices), nullptr);
        },
        "Every GPU on this machine, without creating a Context. Pass one to\n"
        "Context(device=...) to run on it. The default picks automatically.");

    // ── Gamepad ──
    // A snapshot by value: read it, use it for the frame, drop it. The state
    // itself is refreshed by poll_events().
    py::class_<Gamepad>(m, "Gamepad")
        .def_readonly("index", &Gamepad::index)
        .def_readonly("name", &Gamepad::name)
        .def("axis", &Gamepad::axis, py::arg("axis"))
        .def("button", &Gamepad::button, py::arg("button"))
        .def("__repr__", [](const Gamepad& g) { return std::format("<bazalt.Gamepad {} '{}'>", g.index, g.name); });

    // Free function, not a Window method, for the reason poll_events() is one:
    // glfwGetGamepadState takes a joystick id and no window.
    m.def(
        "get_gamepad",
        [](int index, float deadzone) -> py::object
        {
            // ValueError, not a BazaltError: both are values outside a fixed range
            // in the signature, so nothing had to be consulted to know they are
            // wrong. See DESIGN.md on which exception a user error gets.
            if (index < 0 || index > GLFW_JOYSTICK_LAST)
            {
                throw py::value_error(
                    std::format("gamepad index must be between 0 and {}, got {}", GLFW_JOYSTICK_LAST, index));
            }
            if (deadzone < 0.0f || deadzone >= 1.0f)
            {
                throw py::value_error(std::format("deadzone must be at least 0.0 and below 1.0, got {}", deadzone));
            }
            auto pad = unwrap(get_gamepad(index, deadzone), nullptr);
            return pad ? py::cast(*pad) : py::none();
        },
        py::arg("index") = 0,
        py::kw_only(),
        py::arg("deadzone") = 0.0f,
        "The gamepad in slot `index`, or None when that slot is empty.\n\n"
        "Level state only: which buttons are down and where the sticks are, as of\n"
        "the last poll_events(). Needs at least one live Window, because GLFW is\n"
        "initialized with the first one.");

    // ── Key Constants ──
    m.attr("KEY_SPACE") = GLFW_KEY_SPACE;
    m.attr("KEY_APOSTROPHE") = GLFW_KEY_APOSTROPHE;
    m.attr("KEY_COMMA") = GLFW_KEY_COMMA;
    m.attr("KEY_MINUS") = GLFW_KEY_MINUS;
    m.attr("KEY_PERIOD") = GLFW_KEY_PERIOD;
    m.attr("KEY_SLASH") = GLFW_KEY_SLASH;
    m.attr("KEY_0") = GLFW_KEY_0;
    m.attr("KEY_1") = GLFW_KEY_1;
    m.attr("KEY_2") = GLFW_KEY_2;
    m.attr("KEY_3") = GLFW_KEY_3;
    m.attr("KEY_4") = GLFW_KEY_4;
    m.attr("KEY_5") = GLFW_KEY_5;
    m.attr("KEY_6") = GLFW_KEY_6;
    m.attr("KEY_7") = GLFW_KEY_7;
    m.attr("KEY_8") = GLFW_KEY_8;
    m.attr("KEY_9") = GLFW_KEY_9;
    m.attr("KEY_SEMICOLON") = GLFW_KEY_SEMICOLON;
    m.attr("KEY_EQUAL") = GLFW_KEY_EQUAL;
    m.attr("KEY_A") = GLFW_KEY_A;
    m.attr("KEY_B") = GLFW_KEY_B;
    m.attr("KEY_C") = GLFW_KEY_C;
    m.attr("KEY_D") = GLFW_KEY_D;
    m.attr("KEY_E") = GLFW_KEY_E;
    m.attr("KEY_F") = GLFW_KEY_F;
    m.attr("KEY_G") = GLFW_KEY_G;
    m.attr("KEY_H") = GLFW_KEY_H;
    m.attr("KEY_I") = GLFW_KEY_I;
    m.attr("KEY_J") = GLFW_KEY_J;
    m.attr("KEY_K") = GLFW_KEY_K;
    m.attr("KEY_L") = GLFW_KEY_L;
    m.attr("KEY_M") = GLFW_KEY_M;
    m.attr("KEY_N") = GLFW_KEY_N;
    m.attr("KEY_O") = GLFW_KEY_O;
    m.attr("KEY_P") = GLFW_KEY_P;
    m.attr("KEY_Q") = GLFW_KEY_Q;
    m.attr("KEY_R") = GLFW_KEY_R;
    m.attr("KEY_S") = GLFW_KEY_S;
    m.attr("KEY_T") = GLFW_KEY_T;
    m.attr("KEY_U") = GLFW_KEY_U;
    m.attr("KEY_V") = GLFW_KEY_V;
    m.attr("KEY_W") = GLFW_KEY_W;
    m.attr("KEY_X") = GLFW_KEY_X;
    m.attr("KEY_Y") = GLFW_KEY_Y;
    m.attr("KEY_Z") = GLFW_KEY_Z;
    m.attr("KEY_LEFT_BRACKET") = GLFW_KEY_LEFT_BRACKET;
    m.attr("KEY_BACKSLASH") = GLFW_KEY_BACKSLASH;
    m.attr("KEY_RIGHT_BRACKET") = GLFW_KEY_RIGHT_BRACKET;
    m.attr("KEY_GRAVE_ACCENT") = GLFW_KEY_GRAVE_ACCENT;
    m.attr("KEY_WORLD_1") = GLFW_KEY_WORLD_1;
    m.attr("KEY_WORLD_2") = GLFW_KEY_WORLD_2;
    m.attr("KEY_ESCAPE") = GLFW_KEY_ESCAPE;
    m.attr("KEY_ENTER") = GLFW_KEY_ENTER;
    m.attr("KEY_TAB") = GLFW_KEY_TAB;
    m.attr("KEY_BACKSPACE") = GLFW_KEY_BACKSPACE;
    m.attr("KEY_INSERT") = GLFW_KEY_INSERT;
    m.attr("KEY_DELETE") = GLFW_KEY_DELETE;
    m.attr("KEY_RIGHT") = GLFW_KEY_RIGHT;
    m.attr("KEY_LEFT") = GLFW_KEY_LEFT;
    m.attr("KEY_DOWN") = GLFW_KEY_DOWN;
    m.attr("KEY_UP") = GLFW_KEY_UP;
    m.attr("KEY_PAGE_UP") = GLFW_KEY_PAGE_UP;
    m.attr("KEY_PAGE_DOWN") = GLFW_KEY_PAGE_DOWN;
    m.attr("KEY_HOME") = GLFW_KEY_HOME;
    m.attr("KEY_END") = GLFW_KEY_END;
    m.attr("KEY_CAPS_LOCK") = GLFW_KEY_CAPS_LOCK;
    m.attr("KEY_SCROLL_LOCK") = GLFW_KEY_SCROLL_LOCK;
    m.attr("KEY_NUM_LOCK") = GLFW_KEY_NUM_LOCK;
    m.attr("KEY_PRINT_SCREEN") = GLFW_KEY_PRINT_SCREEN;
    m.attr("KEY_PAUSE") = GLFW_KEY_PAUSE;
    m.attr("KEY_F1") = GLFW_KEY_F1;
    m.attr("KEY_F2") = GLFW_KEY_F2;
    m.attr("KEY_F3") = GLFW_KEY_F3;
    m.attr("KEY_F4") = GLFW_KEY_F4;
    m.attr("KEY_F5") = GLFW_KEY_F5;
    m.attr("KEY_F6") = GLFW_KEY_F6;
    m.attr("KEY_F7") = GLFW_KEY_F7;
    m.attr("KEY_F8") = GLFW_KEY_F8;
    m.attr("KEY_F9") = GLFW_KEY_F9;
    m.attr("KEY_F10") = GLFW_KEY_F10;
    m.attr("KEY_F11") = GLFW_KEY_F11;
    m.attr("KEY_F12") = GLFW_KEY_F12;
    m.attr("KEY_F13") = GLFW_KEY_F13;
    m.attr("KEY_F14") = GLFW_KEY_F14;
    m.attr("KEY_F15") = GLFW_KEY_F15;
    m.attr("KEY_F16") = GLFW_KEY_F16;
    m.attr("KEY_F17") = GLFW_KEY_F17;
    m.attr("KEY_F18") = GLFW_KEY_F18;
    m.attr("KEY_F19") = GLFW_KEY_F19;
    m.attr("KEY_F20") = GLFW_KEY_F20;
    m.attr("KEY_F21") = GLFW_KEY_F21;
    m.attr("KEY_F22") = GLFW_KEY_F22;
    m.attr("KEY_F23") = GLFW_KEY_F23;
    m.attr("KEY_F24") = GLFW_KEY_F24;
    m.attr("KEY_F25") = GLFW_KEY_F25;
    m.attr("KEY_KP_0") = GLFW_KEY_KP_0;
    m.attr("KEY_KP_1") = GLFW_KEY_KP_1;
    m.attr("KEY_KP_2") = GLFW_KEY_KP_2;
    m.attr("KEY_KP_3") = GLFW_KEY_KP_3;
    m.attr("KEY_KP_4") = GLFW_KEY_KP_4;
    m.attr("KEY_KP_5") = GLFW_KEY_KP_5;
    m.attr("KEY_KP_6") = GLFW_KEY_KP_6;
    m.attr("KEY_KP_7") = GLFW_KEY_KP_7;
    m.attr("KEY_KP_8") = GLFW_KEY_KP_8;
    m.attr("KEY_KP_9") = GLFW_KEY_KP_9;
    m.attr("KEY_KP_DECIMAL") = GLFW_KEY_KP_DECIMAL;
    m.attr("KEY_KP_DIVIDE") = GLFW_KEY_KP_DIVIDE;
    m.attr("KEY_KP_MULTIPLY") = GLFW_KEY_KP_MULTIPLY;
    m.attr("KEY_KP_SUBTRACT") = GLFW_KEY_KP_SUBTRACT;
    m.attr("KEY_KP_ADD") = GLFW_KEY_KP_ADD;
    m.attr("KEY_KP_ENTER") = GLFW_KEY_KP_ENTER;
    m.attr("KEY_KP_EQUAL") = GLFW_KEY_KP_EQUAL;
    m.attr("KEY_LEFT_SHIFT") = GLFW_KEY_LEFT_SHIFT;
    m.attr("KEY_LEFT_CONTROL") = GLFW_KEY_LEFT_CONTROL;
    m.attr("KEY_LEFT_ALT") = GLFW_KEY_LEFT_ALT;
    m.attr("KEY_LEFT_SUPER") = GLFW_KEY_LEFT_SUPER;
    m.attr("KEY_RIGHT_SHIFT") = GLFW_KEY_RIGHT_SHIFT;
    m.attr("KEY_RIGHT_CONTROL") = GLFW_KEY_RIGHT_CONTROL;
    m.attr("KEY_RIGHT_ALT") = GLFW_KEY_RIGHT_ALT;
    m.attr("KEY_RIGHT_SUPER") = GLFW_KEY_RIGHT_SUPER;
    m.attr("KEY_MENU") = GLFW_KEY_MENU;
    m.attr("KEY_LAST") = GLFW_KEY_LAST;

    m.attr("MOUSE_BUTTON_LEFT") = GLFW_MOUSE_BUTTON_LEFT;
    m.attr("MOUSE_BUTTON_RIGHT") = GLFW_MOUSE_BUTTON_RIGHT;
    m.attr("MOUSE_BUTTON_MIDDLE") = GLFW_MOUSE_BUTTON_MIDDLE;

    m.attr("CURSOR_NORMAL") = GLFW_CURSOR_NORMAL;
    m.attr("CURSOR_DISABLED") = GLFW_CURSOR_DISABLED;
    m.attr("CURSOR_HIDDEN") = GLFW_CURSOR_HIDDEN;
}
