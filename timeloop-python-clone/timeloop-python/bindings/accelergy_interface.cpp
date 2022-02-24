#include "bindings/bindings.h"

// Timeloop headers
#include "util/accelergy_interface.hpp"

void BindAccelergyInterface(py::module &m) {
  m.def("native_invoke_accelergy", &accelergy::invokeAccelergy,
        py::call_guard<py::scoped_ostream_redirect,
                       py::scoped_estream_redirect>(),
        "Invokes Accelergy");
}
