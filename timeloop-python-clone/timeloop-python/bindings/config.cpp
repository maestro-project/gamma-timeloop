#include <optional>
#include <variant>

#include "bindings/bindings.h"

// PyBind11 headers
#include "pybind11/stl.h"

// Timeloop headers
#include "compound-config/compound-config.hpp"

typedef std::variant<bool, long long, unsigned long long, double, std::string,
                     config::CompoundConfigNode>
    CompoundConfigLookupReturn;

void BindConfigClasses(py::module& m) {
  py::class_<config::CompoundConfig>(m, "NativeConfig")
      .def(py::init<>())
      .def(py::init<char*>())
      .def(py::init<std::vector<std::string>>())
      .def(py::init<std::string, std::string>())
      .def_readonly("in_files", &config::CompoundConfig::inFiles)
      .def("get_root", &config::CompoundConfig::getRoot)
      .def("get_variable_root", &config::CompoundConfig::getVariableRoot);

  py::class_<config::CompoundConfigNode>(m, "NativeConfigNode")
      .def(py::init<>())
      .def("__getitem__",
           [](config::CompoundConfigNode& n,
              std::string key) -> CompoundConfigLookupReturn {
             if (!n.exists(key)) {
               throw py::key_error(key);
             }

             bool bool_res;
             if (n.lookupValue(key, bool_res)) return bool_res;
             long long int_res;
             if (n.lookupValue(key, int_res)) return int_res;
             unsigned long long uint_res;
             if (n.lookupValue(key, uint_res)) return uint_res;
             double float_res;
             if (n.lookupValue(key, float_res)) return float_res;
             std::string string_res;
             if (n.lookupValue(key, string_res)) return string_res;

             return n.lookup(key);
           })
      .def("__getitem__",
           [](config::CompoundConfigNode& n,
              int idx) -> CompoundConfigLookupReturn {
             if (n.isArray()) {
               std::vector<std::string> resultVector;
               if (n.getArrayValue(resultVector)) {
                 return resultVector[idx];
               }
               throw py::index_error("Failed to access array.");
             }
             if (n.isList()) {
               return n[idx];
             }
             throw py::index_error("Not a list nor an array.");
           })
      .def("lookup", [](config::CompoundConfigNode& n,
                        std::string key) { return n.lookup(key); })
      .def("__contains__", [](config::CompoundConfigNode& n,
                              std::string key) { return n.exists(key); })
      .def("keys", [](config::CompoundConfigNode& n) {
        std::vector<std::string> all_keys;
        n.getMapKeys(all_keys);
        return all_keys;
      });
}
