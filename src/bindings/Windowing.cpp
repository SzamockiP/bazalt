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

    // Inert data, like Device: what list_monitors() reports so a caller can
    // choose where to open. No methods — every question it answers is a field.
    py::class_<VideoMode>(m, "VideoMode")
        .def_readonly("width", &VideoMode::width)
        .def_readonly("height", &VideoMode::height)
        .def_readonly("refresh_rate", &VideoMode::refresh_rate)
        .def(
            "__repr__",
            [](const VideoMode& v)
            { return std::format("<bazalt.VideoMode {}x{} @{}Hz>", v.width, v.height, v.refresh_rate); });

    py::class_<Monitor>(m, "Monitor")
        .def_readonly("name", &Monitor::name)
        .def_readonly("primary", &Monitor::primary)
        .def_property_readonly("position", [](const Monitor& mon) { return py::make_tuple(mon.x, mon.y); })
        .def_readonly("current_mode", &Monitor::current_mode)
        .def_property_readonly(
            "physical_size_mm",
            [](const Monitor& mon) { return py::make_tuple(mon.physical_width_mm, mon.physical_height_mm); })
        .def_property_readonly(
            "content_scale", [](const Monitor& mon) { return py::make_tuple(mon.scale_x, mon.scale_y); })
        .def_readonly("video_modes", &Monitor::video_modes)
        .def(
            "__repr__",
            [](const Monitor& mon)
            {
                return std::format(
                    "<bazalt.Monitor '{}' {}x{}{}>",
                    mon.name,
                    mon.current_mode.width,
                    mon.current_mode.height,
                    mon.primary ? " primary" : "");
            });

    m.def(
        "list_monitors",
        []() { return unwrap(list_monitors(), nullptr); },
        "Every connected monitor, primary first (0.25).\n\n"
        "Unlike the clipboard and the gamepads this needs no live Window, because\n"
        "choosing where to open one happens first. Pass a result to\n"
        "bz.Window(monitor=...) or window.set_mode(mode, monitor=...).");

    py::class_<Window>(m, "Window")
        .def(
            py::init(
                [](int width,
                   int height,
                   const std::string& title,
                   const std::shared_ptr<Logger>& logger,
                   WindowMode mode,
                   std::optional<Monitor> monitor)
                {
                    // Window used to have no way to reach a Logger at all, so GLFW's own
                    // diagnostics went nowhere.
                    return unwrap(
                        Window::create(width, height, title, logger, mode, monitor ? &*monitor : nullptr),
                        logger.get());
                }),
            py::arg("width"),
            py::arg("height"),
            py::arg("title"),
            py::arg("logger") = py::none(),
            py::arg("mode") = WindowMode::WINDOWED,
            py::arg("monitor") = py::none())
        .def("is_open", &Window::is_open)
        .def("is_key_pressed", &Window::is_key_pressed, py::arg("key"))
        .def("is_mouse_button_pressed", &Window::is_mouse_button_pressed, py::arg("button"))
        .def("was_key_pressed", &Window::was_key_pressed, py::arg("key"))
        .def("was_mouse_button_pressed", &Window::was_mouse_button_pressed, py::arg("button"))
        .def("set_cursor_mode", &Window::set_cursor_mode, py::arg("mode"))
        .def("set_cursor", &Window::set_cursor, py::arg("shape"))
        .def("get_mouse_state", &Window::get_mouse_state)
        .def("set_title", &Window::set_title, py::arg("title"))
        .def(
            "set_mode",
            // nullptr logger: GLFW's own error callback has already logged the
            // reason through the Window's Logger, so passing it here would say
            // the same thing twice.
            [](Window& self, WindowMode mode, std::optional<Monitor> monitor, std::optional<VideoMode> video_mode)
            {
                unwrap(
                    self.set_mode(mode, monitor ? &*monitor : nullptr, video_mode ? &*video_mode : nullptr), nullptr);
            },
            py::arg("mode"),
            py::kw_only(),
            py::arg("monitor") = py::none(),
            py::arg("video_mode") = py::none())
        .def("set_size", &Window::set_size, py::arg("width"), py::arg("height"))
        .def("set_position", &Window::set_position, py::arg("x"), py::arg("y"))
        .def("set_cursor_position", &Window::set_cursor_position, py::arg("x"), py::arg("y"))
        .def(
            "dropped_files",
            // Copied into a Python list rather than returned by reference: the
            // vector is rotated out from under the caller on the next poll cycle.
            [](const Window& self) { return py::cast(self.dropped_files()); })
        .def(
            "text_input",
            // Copied for the same reason dropped_files is, and decoded as UTF-8
            // rather than handed over as bytes: the C++ side already encoded it,
            // and a str is what a text field appends.
            [](const Window& self) { return py::str(self.text_input()); })
        .def(
            "set_icon",
            [](Window& self, const py::object& icon)
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
        .def_property_readonly("limits", [](const Device& d) { return d.limits; })
        .def("supports", &Device::supports, py::arg("feature"))
        .def(
            "__repr__",
            [](const Device& d)
            {
                return std::format(
                    "<bazalt.Device '{}' ({}, {} MB)>",
                    d.name,
                    d.type,
                    d.limits.device_memory / (VkDeviceSize{1024} * 1024));
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
        .def("was_button_pressed", &Gamepad::was_button_pressed, py::arg("button"))
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
        "What the pad reads as of the last poll_events(): which buttons are down,\n"
        "where the sticks are, and which buttons went down since the previous poll\n"
        "cycle (pad.was_button_pressed). Needs at least one live Window, because\n"
        "GLFW is initialized with the first one.\n\n"
        "GLFW gives no gamepad callback, so an edge is measured between two reads.\n"
        "Read each pad every frame, or a press and a release inside a frame you\n"
        "skipped are both invisible.");
}
