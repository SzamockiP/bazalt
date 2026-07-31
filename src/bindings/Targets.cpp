#include "Bindings.hpp"

void bind_targets(py::module_& m)
{
    // ── RenderTarget ──
    py::class_<RenderTarget, std::shared_ptr<RenderTarget>>(m, "RenderTargetBase");

    py::class_<OffscreenTarget, RenderTarget, std::shared_ptr<OffscreenTarget>>(m, "RenderTarget")
        // color: None | Format | [Format, ...]; depth: None | Format. A bool
        // depth is refused with a migration hint (it was one in 0.4).
        .def(
            py::init(
                [](Context& context,
                   std::uint32_t width,
                   std::uint32_t height,
                   py::object color,
                   py::object depth,
                   std::uint32_t samples,
                   std::uint32_t layers,
                   bool cube,
                   std::uint32_t mip_levels,
                   const std::string& name)
                {
                    return make_offscreen_target(
                        context, width, height, color, depth, samples, layers, cube, mip_levels, name);
                }),
            py::arg("context"),
            py::arg("width"),
            py::arg("height"),
            py::arg("color") = Format::RGBA8,
            py::arg("depth") = py::none(),
            py::arg("samples") = 1,
            py::kw_only(),
            py::arg("layers") = 1,
            py::arg("cube") = false,
            py::arg("mip_levels") = 1,
            py::arg("name") = "")
        // A target on images from create_image, instead of attachments the target
        // allocates. A second __init__ rather than optional width/height on the one
        // above: this signature has no width, height, layers, cube or mip_levels,
        // because every one of those is a property of the images now. samples= stays,
        // because it is not: it says how many samples to RENDER with, and the images
        // handed in are the resolve targets.
        // pybind picks between the two on arity — width and height are required
        // positionals up there and absent here.
        .def(
            py::init(
                [](Context& context, py::object color, py::object depth, std::uint32_t samples, const std::string& name)
                { return make_offscreen_target_from_images(context, color, depth, samples, name); }),
            py::arg("context"),
            py::kw_only(),
            py::arg("color") = py::none(),
            py::arg("depth") = py::none(),
            py::arg("samples") = 1,
            py::arg("name") = "")
        .def_property_readonly("width", [](const OffscreenTarget& t) { return t.extent().width; })
        .def_property_readonly("height", [](const OffscreenTarget& t) { return t.extent().height; })
        // The attachments are ordinary Images — this is the whole
        // render-to-texture API: target.color[0] / target.depth into set_image.
        .def_property_readonly(
            "color",
            [](const OffscreenTarget& t)
            {
                py::tuple out(t.colors().size());
                for (size_t i = 0; i < t.colors().size(); ++i)
                {
                    out[i] = py::cast(t.colors()[i]);
                }
                return out;
            })
        .def_property_readonly(
            "depth",
            [](const OffscreenTarget& t) -> py::object { return t.depth() ? py::cast(t.depth()) : py::none(); })
        // Render-to-layer / render-to-mip: a lightweight view of one subresource.
        // Pass it straight to cmd.rendering(...). Cube face i == layer i, Vulkan
        // order +X, -X, +Y, -Y, +Z, -Z.
        .def(
            "layer",
            [](std::shared_ptr<OffscreenTarget> self, std::uint32_t index, std::uint32_t mip)
            { return unwrap(self->layer(index, mip), nullptr); },
            py::arg("index"),
            py::arg("mip") = 0)
        // Multiview: one pass into every layer (the shader uses gl_ViewIndex).
        .def("all_layers", [](std::shared_ptr<OffscreenTarget> self) { return unwrap(self->all_layers(), nullptr); });

    // Registered so pybind's automatic downcasting (the base is polymorphic)
    // returns a NAMED type from layer() / all_layers() above — until 0.23 both
    // came back as opaque RenderTargetBase, so nothing could be said about them
    // in the stub. No methods: each is a view that exists to be handed to
    // cmd.rendering(...), and the parent keeps every knob.
    py::class_<SubresourceTarget, RenderTarget, std::shared_ptr<SubresourceTarget>>(m, "SubresourceTarget");
    py::class_<MultiviewTarget, RenderTarget, std::shared_ptr<MultiviewTarget>>(m, "MultiviewTarget");

    py::class_<SwapchainRenderer, RenderTarget, std::shared_ptr<SwapchainRenderer>>(m, "SwapchainRenderer")
        .def(
            py::init(
                [](Window& window,
                   std::shared_ptr<Context> context,
                   PresentMode present_mode,
                   std::uint32_t samples,
                   bool stencil)
                {
                    auto sp = window.get_surface_provider();
                    return std::shared_ptr<SwapchainRenderer>(unwrap(
                        SwapchainRenderer::create(context, std::move(sp), present_mode, samples, stencil),
                        context->logger().get()));
                }),
            py::arg("window"),
            py::arg("context"),
            py::arg("present_mode") = PresentMode::MAILBOX,
            py::arg("samples") = 1,
            py::arg("stencil") = false,
            // The SurfaceProvider that get_surface_provider() returns captures the raw
            // GLFWwindow* and a pointer to the Window's own resize flag, and the renderer
            // keeps it for its whole life. So the Window has to outlive the renderer, and
            // nothing else says so: `del window` alone would leave both pointers dangling.
            // The hwnd overload below captures an integer and needs no such tie.
            py::keep_alive<1, 2>())
        .def(
            py::init(
                [](uint64_t hwnd,
                   std::shared_ptr<Context> context,
                   PresentMode present_mode,
                   std::uint32_t samples,
                   bool stencil) -> std::shared_ptr<SwapchainRenderer>
                {
#ifdef _WIN32
                    SurfaceProvider sp;
                    sp.required_instance_extensions = {"VK_KHR_surface", "VK_KHR_win32_surface"};

                    sp.create_surface = [hwnd](VkInstance instance) -> VkSurfaceKHR
                    {
                        auto pfnCreateWin32Surface =
                            (PFN_vkCreateWin32SurfaceKHR)vkGetInstanceProcAddr(instance, "vkCreateWin32SurfaceKHR");
                        if (!pfnCreateWin32Surface)
                        {
                            return VK_NULL_HANDLE;
                        }
                        VkWin32SurfaceCreateInfoKHR createInfo{
                            .sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
                            .pNext = nullptr,
                            .flags = 0,
                            .hinstance = GetModuleHandle(nullptr),
                            .hwnd = (HWND)hwnd};
                        VkSurfaceKHR surface = VK_NULL_HANDLE;
                        if (pfnCreateWin32Surface(instance, &createInfo, nullptr, &surface) != VK_SUCCESS)
                        {
                            return VK_NULL_HANDLE;
                        }
                        return surface;
                    };

                    sp.get_framebuffer_size = [hwnd]() -> std::pair<int, int>
                    {
                        RECT rect;
                        if (GetClientRect((HWND)hwnd, &rect))
                        {
                            return {rect.right - rect.left, rect.bottom - rect.top};
                        }
                        return {0, 0};
                    };

                    auto last_width = std::make_shared<int>(0);
                    auto last_height = std::make_shared<int>(0);

                    sp.consume_resize_flag = [hwnd, last_width, last_height]() -> bool
                    {
                        RECT rect;
                        if (GetClientRect((HWND)hwnd, &rect))
                        {
                            int w = rect.right - rect.left;
                            int h = rect.bottom - rect.top;
                            if (w != *last_width || h != *last_height)
                            {
                                *last_width = w;
                                *last_height = h;
                                return true;
                            }
                        }
                        return false;
                    };

                    return std::shared_ptr<SwapchainRenderer>(unwrap(
                        SwapchainRenderer::create(context, std::move(sp), present_mode, samples, stencil),
                        context->logger().get()));
#else
            raise_error(err_window("win32_hwnd constructor is only supported on Windows"));
#endif
                }),
            py::arg("win32_hwnd"),
            py::arg("context"),
            py::arg("present_mode") = PresentMode::MAILBOX,
            py::arg("samples") = 1,
            py::arg("stencil") = false)
        .def_property_readonly("present_mode", &SwapchainRenderer::present_mode)
        // A verb, not a settable property, because the request is a preference:
        // read present_mode back to see what the driver actually gave you.
        .def(
            "set_present_mode",
            [](SwapchainRenderer& self, PresentMode mode)
            {
                std::expected<void, Error> result;
                {
                    // Recreation takes the device idle — release the GIL for the
                    // wait, and unwrap only once it is back (raising needs it).
                    py::gil_scoped_release release;
                    result = self.set_present_mode(mode);
                }
                unwrap(std::move(result), nullptr);
            },
            py::arg("mode"))
        // bool, not a Frame: the frame belongs to the Context (ctx.begin_frame),
        // and what a window contributes to it is one acquired image. False means
        // "this window sits this frame out" — the others carry on.
        .def(
            "acquire",
            [](SwapchainRenderer& self) -> bool
            {
                std::expected<bool, Error> acquired;
                {
                    // Waits on the in-flight fence — release the GIL meanwhile.
                    py::gil_scoped_release release;
                    acquired = self.acquire();
                }
                return unwrap(std::move(acquired), self.context()->logger().get());
            })
        .def(
            "present",
            [](SwapchainRenderer& self, std::shared_ptr<CommandBuffer> cmd, bool capture)
            {
                require_same_context(self.owner(), cmd->owner(), "present");
                std::expected<void, Error> r;
                {
                    // May CPU-wait for an upload still decoding — release the GIL.
                    py::gil_scoped_release release;
                    r = present_command_buffer(self, std::move(cmd), capture);
                }
                unwrap(std::move(r), nullptr);
            },
            py::arg("cmd"),
            py::kw_only(),
            py::arg("capture") = false)
        // float milliseconds, or None until the ring has cycled once / on
        // devices without timestamp support. Per renderer, because the timestamp
        // pool is: two windows have two GPU frame times.
        .def_property_readonly(
            "gpu_time_ms",
            [](const SwapchainRenderer& r) -> py::object
            {
                auto ms = r.gpu_time_ms();
                return ms ? py::cast(*ms) : py::none();
            })
        // The frame asked for with present(capture=True). RGBA8 whatever channel
        // order the compositor picked, so out[y, x, 0] is red on every machine.
        // Shaped from the CAPTURE extent, not the current one: the window may
        // have been resized since.
        .def(
            "read_pixels",
            [](SwapchainRenderer& self) -> py::array
            {
                std::expected<std::vector<std::byte>, Error> bytes;
                {
                    // Waits on the capturing frame's fence.
                    py::gil_scoped_release release;
                    bytes = self.read_pixels();
                }
                auto data = unwrap(std::move(bytes), nullptr);
                const std::vector<py::ssize_t> shape{
                    static_cast<py::ssize_t>(self.capture_extent().height),
                    static_cast<py::ssize_t>(self.capture_extent().width),
                    4};
                py::array out(py::dtype("uint8"), shape);
                std::memcpy(out.mutable_data(), data.data(), data.size());
                return out;
            })
        .def_property_readonly("width", [](const SwapchainRenderer& r) { return r.extent().width; })
        .def_property_readonly("height", [](const SwapchainRenderer& r) { return r.extent().height; });
}
