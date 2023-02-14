#pragma once

#include <pybind11/pybind11.h>

namespace nope {
void registerTensorBindings(pybind11::module_& module);
} // namespace nope
