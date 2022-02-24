#include "bindings/bindings.h"
#include "bindings/type_casters.h"

// Timeloop headers
#include "mapping/arch-properties.hpp"
#include "mapping/constraints.hpp"
#include "mapping/mapping.hpp"
#include "mapping/parser.hpp"


using PerDataSpaceInt = problem::PerDataSpace<std::uint64_t>;

void BindMappingClasses(py::module& m) {
  py::class_<ArchProperties>(m, "ArchProperties")
      .def(py::init<>())
      .def(py::init<const model::Engine::Specs&>());

  py::class_<mapping::Constraints>(m, "NativeArchConstraints")
      .def(py::init<const ArchProperties&, const problem::Workload&>(),
           "Construct ArchConstraints. ArchConstraints.parse has to be called "
           "later with the configs.")
      .def(py::init([](const ArchProperties& props,
                       const problem::Workload& workload,
                       config::CompoundConfigNode config) {
             auto c = std::make_unique<mapping::Constraints>(props, workload);
             c->Parse(config);
             return c;
           }),
           "Construct and parse ArchConstraints.")
      .def("parse", [](mapping::Constraints& c,
                       config::CompoundConfigNode config) { c.Parse(config); })
      .def("satisfied_by", &mapping::Constraints::SatisfiedBy,
           "Checks if the given mapping satisfies this constraint.");

  py::class_<Mapping>(m, "NativeMapping")
      .def(py::init(&mapping::ParseAndConstruct))
      .def_static("parse_and_construct", &mapping::ParseAndConstruct)
      .def("datatype_bypass_nest",
           [](Mapping& m) { return &Mapping::datatype_bypass_nest; })
      .def(
          "pretty_print",
          [](Mapping& m, const std::vector<std::string>& storage_level_names,
             const std::vector<PerDataSpaceInt>& utilized_capacities,
             const std::vector<PerDataSpaceInt>& tile_sizes,
             const std::string indent) {
            std::ostringstream out;
            m.PrettyPrint(out, storage_level_names, utilized_capacities,
                          tile_sizes, indent);
            return out.str();
          },
          py::arg() = {}, py::arg() = {}, py::arg() = {},
          py::arg("indent") = "");
}
