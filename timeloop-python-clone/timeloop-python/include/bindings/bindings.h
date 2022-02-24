#pragma once

// PyBind11 headers
#include "pybind11/iostream.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#define USE_ACCELERGY

namespace py = pybind11;

void BindAccelergyInterface(py::module &m);
void BindConfigClasses(py::module &m);
void BindMappingClasses(py::module &m);
void BindModelClasses(py::module &m);
void BindProblemClasses(py::module &m);
