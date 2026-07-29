#include "bindings/Bindings.hpp"

PYBIND11_MODULE(_core, m)
{
    m.doc() = "Bazalt native core module";

    // Logger drain threads call into Python; joining them after the
    // interpreter starts finalizing crashes ("could not acquire lock for
    // stderr at interpreter shutdown"). atexit runs while Python is intact,
    // so every script that ends with a live Context exits cleanly.
    py::module_::import("atexit").attr("register")(py::cpp_function([]() { Logger::shutdown_all(); }));

    register_exceptions(m);

    // Enums first: `py::arg(x) = SomeEnum::VALUE` casts the default at .def()
    // time, not at call time, so every enum used as a default must already be
    // registered.
    bind_enums(m);

    bind_resources(m);
    bind_pipelines(m);
    bind_commands(m);
    bind_windowing(m);
    bind_context(m);

    // Last, and this one is a hard requirement rather than a preference:
    // RenderTargetBase must be registered before OffscreenTarget and
    // SwapchainRenderer, the module's only two derived registrations. Both live
    // in Targets.cpp in that order, so the constraint is contained in one file.
    //
    // RenderTarget as a PARAMETER type (GraphicsPipelineBuilder::build,
    // CommandBuffer::begin_rendering) does not constrain anything: pybind
    // resolves a caster at the first Python call, not at .def() time. It looks
    // like it should, which is why this says so.
    bind_targets(m);
}
