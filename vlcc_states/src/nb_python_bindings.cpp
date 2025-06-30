#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/ndarray.h>
#include "state_provider.hpp"

#include "rawptr_state_provider.hpp"
#include "object_state_provider.hpp"
#include "composite_state_provider.hpp"

#include "vlcc_states/defs.hpp"
#include "vlcc_states/state_region.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace vlcc_states;

NB_MODULE(vlcc_states, m) {
      m.doc() = "VLCC State Management Engine";

      // Base class
      nb::class_<state_provider_t>(m, "StateProvider")
            .def("get_name", &state_provider_t::get_name)
            .def("get_tier", &state_provider_t::get_tier)
            .def("get_next_chunk", &state_provider_t::get_next_chunk, "tier_type"_a)
            .def("release", &state_provider_t::release);

      // Raw pointer provider
      nb::class_<rawptr_state_provider_t, state_provider_t>(m, "RawPtrProvider")
            .def(nb::init<std::string, TIER_TYPES>(), "name"_a, "tier"_a)
            .def("register_state", &rawptr_state_provider_t::register_state,
                  "ptr"_a, "size"_a, "is_serialized"_a = true);

      // Object provider
      nb::class_<object_state_provider_t, state_provider_t>(m, "ObjectProvider")
            .def(nb::init<std::string, TIER_TYPES>(), "name"_a, "tier"_a)
            .def("register_state", &object_state_provider_t::register_state,
                  "obj"_a);

      // Composite provider
      nb::class_<composite_state_provider_t, state_provider_t>(m, "CompositeProvider")
            .def(nb::init<std::string, TIER_TYPES>(), "name"_a, "tier"_a)
            .def("register_provider", &composite_state_provider_t::register_provider,
                  "provider"_a);

      // -------- BELOW ARE HELPERS JUST FOR TESTING PURPOSES -------- 
      // We might need to change how these are exposed in the future.
      nb::enum_<vlcc_states::TIER_TYPES>(m, "TIER_TYPES")
            .value("HOST_UNPINNED_TIER", vlcc_states::TIER_TYPES::HOST_UNPINNED_TIER)
            .value("HOST_PINNED_TIER",   vlcc_states::TIER_TYPES::HOST_PINNED_TIER)
            .value("GPU_TIER",           vlcc_states::TIER_TYPES::GPU_TIER)
            .value("UNIFIED_MEM_TIER",   vlcc_states::TIER_TYPES::UNIFIED_MEM_TIER)
            .value("COMPOSITE_TIER",     vlcc_states::TIER_TYPES::COMPOSITE_TIER)
            .value("FILE_TIER",          vlcc_states::TIER_TYPES::FILE_TIER)
            .export_values();

      nb::class_<vlcc_states::state_region_t>(m, "StateRegion")
            .def_rw("region_id", &vlcc_states::state_region_t::region_id)
            .def_rw("size", &vlcc_states::state_region_t::size)
            .def_rw("tier", &vlcc_states::state_region_t::tier)
            .def_rw("status", &vlcc_states::state_region_t::status)
            .def_prop_ro("ptr", [](const vlcc_states::state_region_t &r) {
                  return reinterpret_cast<uintptr_t>(r.ptr);  // exposes pointer as int
            })
            .def("__repr__", [](const vlcc_states::state_region_t &r) {
                  return "<StateRegion id=" + std::to_string(r.region_id) +
                        " size=" + std::to_string(r.size) +
                        " ptr=0x" + std::string("%x", reinterpret_cast<uintptr_t>(r.ptr)) + ">";
            });


      m.def("create_object_provider", [](const std::string& name, TIER_TYPES tier, nb::object py_object) {
                  auto provider = std::make_shared<object_state_provider_t>(name, tier);
                  provider->register_state(py_object);
                  return provider;
            }, "name"_a, "tier"_a, "py_object"_a, nb::rv_policy::take_ownership, "Create an object state provider.");


    // Factory functions (optional — class ctors already exposed)
//     m.def("create_rawptr_provider", &create_rawptr_provider,
//           "name"_a, "tier"_a, "ptr"_a, "size"_a,
//           nb::rv_policy::take_ownership);

//     m.def("create_object_provider", &create_object_provider,
//           "name"_a, "tier"_a, "py_object"_a,
//           nb::rv_policy::take_ownership);

//     m.def("create_composite_provider", &create_composite_provider,
//           "name"_a, "tier"_a,
//           nb::rv_policy::take_ownership);
}
