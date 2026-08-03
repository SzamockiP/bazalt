#include "Bindings.hpp"

void bind_targets(py::module_& m)
{
    // ── RenderTarget ──
    py::class_<RenderTarget, std::shared_ptr<RenderTarget>>(m, "RenderTargetBase");

    // No py::init since 0.23: a target comes from ctx.create_render_target(),
    // like every other resource the Context owns. The class stays a type — it
    // is what isinstance and the annotations name — and the two construction
    // shapes live on the Context (see bind_context).
    py::class_<OffscreenTarget, RenderTarget, std::shared_ptr<OffscreenTarget>>(m, "RenderTarget")
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
        // The multisampled attachments, for a custom resolve: bind one to a
        // sampler2DMS and read the samples with texelFetch. Empty tuple / None
        // unless samples > 1, and `color` / `depth` above stay the resolve — the
        // images almost everything wants.
        .def_property_readonly(
            "multisampled_color",
            [](const OffscreenTarget& t)
            {
                py::tuple out(t.multisampled_colors().size());
                for (size_t i = 0; i < t.multisampled_colors().size(); ++i)
                {
                    out[i] = py::cast(t.multisampled_colors()[i]);
                }
                return out;
            })
        .def_property_readonly(
            "multisampled_depth",
            [](const OffscreenTarget& t) -> py::object
            { return t.multisampled_depth() ? py::cast(t.multisampled_depth()) : py::none(); })
        // Render-to-layer / render-to-mip: a lightweight view of one subresource.
        // Pass it straight to cmd.rendering(...). Cube face i == layer i, Vulkan
        // order +X, -X, +Y, -Y, +Z, -Z.
        //
        // mip is keyword-only since 0.23, by the rule the set_image break wrote:
        // two adjacent ints of which the second selects a different axis is the
        // trap that made set_image(0, img, 3) read as "index 3". layer(0, 2) is
        // worse, because on a 3D target the FIRST int is a Z slice, so both
        // spellings look like a coordinate.
        .def(
            "layer",
            [](std::shared_ptr<OffscreenTarget> self, std::uint32_t index, std::uint32_t mip)
            { return unwrap(self->layer(index, mip), nullptr); },
            py::arg("index"),
            py::kw_only(),
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

    // Also constructor-free since 0.23: ctx.create_renderer(window) makes one.
    py::class_<SwapchainRenderer, RenderTarget, std::shared_ptr<SwapchainRenderer>>(m, "SwapchainRenderer")
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
        // float milliseconds, and None means one thing since 0.24: the ring has
        // not cycled once yet. Timing that was never switched on raises
        // StateError and a device that cannot measure raises UnsupportedError,
        // because a caller polling this in a frame loop needs to know which of
        // the three it is looking at. Per renderer, because the timestamp pool
        // is: two windows have two GPU frame times.
        .def_property_readonly(
            "gpu_time_ms",
            [](const SwapchainRenderer& r) -> py::object
            {
                raise_for_query_status(r.timing_status(), "gpu_time_ms");
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
