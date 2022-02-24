#include <sstream>
#include <string>

#include "bindings/model/bindings.h"
#include "bindings/type_casters.h"

// PyBind11 headers
#include "pybind11/iostream.h"

// Timeloop headers
#include "model/engine.hpp"
#include "model/level.hpp"

namespace model_bindings {

void BindEngine(py::module& m) {
  py::class_<model::Engine::Specs>(m, "NativeArchSpecs")
      .def(py::init(&model::Engine::ParseSpecs))
      .def_static("parse_specs", &model::Engine::ParseSpecs,
                  "Parse architecture specifications.")
      .def("parse_accelergy_art",
           [](model::Engine::Specs& s, config::CompoundConfigNode& art) {
             s.topology.ParseAccelergyART(art);
           })
      .def(
          "parse_accelergy_ert",
          [](model::Engine::Specs& s, config::CompoundConfigNode& ert) {
            s.topology.ParseAccelergyERT(ert);
          },
          py::call_guard<py::scoped_ostream_redirect,
                         py::scoped_estream_redirect>())
      .def("level_names",
           [](model::Engine::Specs& s) { return s.topology.LevelNames(); })
      .def("storage_level_names", [](model::Engine::Specs& s) {
        return s.topology.StorageLevelNames();
      });

  py::class_<model::Engine>(m, "NativeEngine")
      .def(py::init<>(),
           "Construct wrapper over Timeloop's native Engine class. Consider "
           "using `pytimeloop.Accelerator` instead. \n"
           "Engine.spec has to be called later with ArchSpecs.")
      .def(py::init([](model::Engine::Specs specs) {
             auto e = std::make_unique<model::Engine>();
             e->Spec(specs);
             return e;
           }),
           "Construct and spec Engine.")
      .def("spec", &model::Engine::Spec)
      .def("pre_evaluation_check", &model::Engine::PreEvaluationCheck)
      .def("evaluate", &model::Engine::Evaluate, py::arg("mapping"),
           py::arg("workload"), py::arg("sparse_optimizations"),
           py::arg("break_on_failure") = false)
      .def("is_evaluated", &model::Engine::IsEvaluated)
      .def("utilization", &model::Engine::Utilization)
      .def("energy", &model::Engine::Energy)
      .def("area", &model::Engine::Area)
      .def("cycles", &model::Engine::Cycles)
      .def("get_topology", &model::Engine::GetTopology)
      .def("pretty_print_stats", [](model::Engine& e) -> std::string {
        std::stringstream ss;
        ss << e << std::endl;
        return ss.str();
      });
}

void BindLevel(py::module& m) {
  py::class_<model::EvalStatus>(m, "EvalStatus")
      .def_readonly("success", &model::EvalStatus::success)
      .def_readonly("fail_reason", &model::EvalStatus::fail_reason)
      .def("__repr__", [](model::EvalStatus& e) -> std::string {
        if (e.success) {
          return "EvalStatus(success=1)";
        } else {
          return "EvalStatus(success=0, fail_reason=" + e.fail_reason + ")";
        }
      });
}

void BindTopology(py::module& m) {
  py::class_<model::Topology>(m, "Topology")
      .def("algorithmic_computes", &model::Topology::AlgorithmicComputes)
      .def("actual_computes", &model::Topology::ActualComputes)
      .def("tile_sizes", &model::Topology::TileSizes)
      .def("utilized_capacities", &model::Topology::UtilizedCapacities);
}

}  // namespace model_bindings
