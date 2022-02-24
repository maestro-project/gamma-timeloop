#include "bindings/bindings.h"

#include <variant>
#include <vector>

#include "bindings/model/bindings.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

PYBIND11_MODULE(bindings, m) {
  m.doc() = "PyTimeloop bindings to C++ timeloop code ";

  BindAccelergyInterface(m);
  BindConfigClasses(m);
  BindMappingClasses(m);

  model_bindings::BindEngine(m);
  model_bindings::BindLevel(m);
  model_bindings::BindSparseOptimizationInfo(m);
  model_bindings::BindTopology(m);

  BindProblemClasses(m);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
