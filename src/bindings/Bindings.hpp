#pragma once
#include "Common.hpp"

// One function per subject, called in this order by PYBIND11_MODULE in main.cpp.
// The order is not cosmetic -- see the comments on the call site.

void bind_enums(py::module_& m);
void bind_resources(py::module_& m);
void bind_pipelines(py::module_& m);
void bind_commands(py::module_& m);
void bind_windowing(py::module_& m);
void bind_context(py::module_& m);
void bind_targets(py::module_& m);
