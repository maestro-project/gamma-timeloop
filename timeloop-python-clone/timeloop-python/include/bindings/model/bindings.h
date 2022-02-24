#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace model_bindings {

void BindEngine(py::module& m);
void BindLevel(py::module& m);
void BindSparseOptimizationInfo(py::module &m);
void BindSparseOptimizationParser(py::module& m);
void BindTopology(py::module& m);

}  // namespace model_bindings
