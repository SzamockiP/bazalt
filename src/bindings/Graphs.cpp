#include "Bindings.hpp"

void bind_graphs(py::module_& m)
{
    // A submit's identity: one timeline value per queue, because a submit of a
    // two-queue graph signals both. Opaque on purpose — no attributes, no
    // ordering, no arithmetic. Handing out the integers would let callers build
    // on a total order across submits, and with two timelines there is none.
    // The repr shows both values because a debugger deserves it; code cannot
    // reach them.
    py::class_<Serial>(m, "Serial")
        .def(
            "__repr__",
            [](const Serial& self)
            {
                return std::format(
                    "<bazalt.Serial graphics={} compute={} transfer={}>",
                    self.values[queue_index(QueueKind::Graphics)],
                    self.values[queue_index(QueueKind::Compute)],
                    self.values[queue_index(QueueKind::Transfer)]);
            });

    py::class_<Graph, std::shared_ptr<Graph>>(m, "Graph")
        // Render pass: the target is required and never None — a pass without
        // one is the second overload, and the clear arguments only exist here
        // (they answer "what happens to the attachments", a question a
        // compute pass does not have).
        .def(
            "add_pass",
            [](const std::shared_ptr<Graph>& self,
               std::shared_ptr<RenderTarget> target,
               const py::object& clear_color,
               float clear_depth,
               std::uint32_t clear_stencil,
               std::string name,
               QueueKind queue,
               std::optional<bool> auto_barriers)
            {
                require_same_context(self->owner(), target->owner(), "add_pass");
                if (queue != QueueKind::Graphics)
                {
                    raise_error(err_state(
                        std::format(
                            "add_pass: a pass on Queue.{} cannot draw, because only the graphics "
                            "queue has a rasterizer. Use queue=Queue.GRAPHICS for a pass with a "
                            "render target.",
                            queue_name(queue))));
                }
                require_sliced_when_3d(*target, "add_pass");
                auto clears = parse_clear_colors(clear_color);
                require_preservable(*target, !clears.has_value(), "add_pass");
                return self->add_pass(
                    std::move(target),
                    std::move(clears),
                    clear_depth,
                    clear_stencil,
                    std::move(name),
                    queue,
                    auto_barriers);
            },
            py::arg("target").none(false),
            py::arg("clear_color") = py::make_tuple(0.0f, 0.0f, 0.0f, 1.0f),
            py::arg("clear_depth") = 1.0f,
            py::arg("clear_stencil") = 0,
            py::kw_only(),
            py::arg("name") = std::string{},
            py::arg("queue") = QueueKind::Graphics,
            py::arg("auto_barriers") = py::none())
        // Compute/transfer pass: no target, and therefore no clear arguments —
        // passing one is a TypeError, which is the refusal the plan asks for.
        .def(
            "add_pass",
            [](const std::shared_ptr<Graph>& self, std::string name, QueueKind queue, std::optional<bool> auto_barriers)
            { return self->add_pass(nullptr, std::nullopt, 1.0f, 0, std::move(name), queue, auto_barriers); },
            py::kw_only(),
            py::arg("name") = std::string{},
            py::arg("queue") = QueueKind::Graphics,
            py::arg("auto_barriers") = py::none())
        .def(
            "remove",
            [](Graph& self, const std::shared_ptr<Pass>& pass) { unwrap(self.remove(pass), nullptr); },
            py::arg("pass_"),
            py::pos_only())
        .def("reset", &Graph::reset);
}
