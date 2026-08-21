#include "Bindings.hpp"

void bind_graphs(py::module_& m)
{
    // A submit's identity: which queue signalled, and the timeline value it
    // signalled. Opaque on purpose — no attributes, no ordering, no
    // arithmetic. Handing out the integer would let callers build on a total
    // order across submits, which stops existing the day a second queue
    // arrives; the handle keeps that door open at zero cost. The repr shows
    // the value because a debugger deserves it; code cannot reach it.
    py::class_<Serial>(m, "Serial")
        .def(
            "__repr__",
            [](const Serial& self)
            { return std::format("<bazalt.Serial queue={} value={}>", self.queue_id, self.value); });

    py::class_<Graph, std::shared_ptr<Graph>>(m, "Graph")
        // Render pass: the target is required and never None — a pass without
        // one is the second overload, and the clear arguments only exist here
        // (they answer "what happens to the attachments", a question a
        // compute pass does not have).
        .def(
            "add_pass",
            [](std::shared_ptr<Graph> self,
               std::shared_ptr<RenderTarget> target,
               const py::object& clear_color,
               float clear_depth,
               std::uint32_t clear_stencil,
               std::string name,
               QueueKind queue,
               std::optional<bool> auto_barriers)
            {
                require_same_context(self->owner(), target->owner(), "add_pass");
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
            [](std::shared_ptr<Graph> self, std::string name, QueueKind queue, std::optional<bool> auto_barriers)
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
