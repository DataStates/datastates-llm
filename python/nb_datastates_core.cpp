#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/ndarray.h>
#include "datastates.hpp"
#include "state_manager.hpp"
#include "state_io_engine.hpp"

namespace nb = nanobind;
using namespace nb::literals;

std::pair<const char*, size_t> extract_ptr_and_size(const nb::handle& obj) {
    // Torch tensor
    if (nb::isinstance<nb::ndarray<>>(obj)) {
        nb::ndarray<> arr = nb::cast<nb::ndarray<>>(obj);
        return {reinterpret_cast<char*>(arr.data()), arr.size() * arr.itemsize()};
    }
    
    // memoryview / bytes / bytearray / str
    if (nb::isinstance<nb::bytes>(obj)) {
        nb::bytes buf = nb::cast<nb::bytes>(obj);
        return {buf.c_str(), buf.size()};
    }

    std::string nb_type_name = nb::cast<std::string>(nb::type_name(obj));
    throw std::runtime_error("Unsupported object of type '" + nb_type_name +
                         "': expected Tensor, ndarray, bytearray, bytes, or str");
}

void ckpt_wrapper(datastates::core_t& self, int version, int region_uid, nb::handle obj, uint64_t size, uint64_t offset, std::string path) {
    auto [ptr, actual_size] = extract_ptr_and_size(obj);
    if (size != actual_size)
        throw std::runtime_error("Requested size exceeds buffer size");
    self.ckpt(version, region_uid, ptr, size, offset, path);
}

void restore_wrapper(datastates::core_t& self, int version, int region_uid, nb::handle obj, uint64_t size, uint64_t offset, std::string path) {
    auto [ptr, actual_size] = extract_ptr_and_size(obj);
    if (size != actual_size)
        throw std::runtime_error("Requested size exceeds buffer size");
    self.restore(version, region_uid, ptr, size, offset, path);
}

NB_MODULE(datastates_core, m) {
    nb::set_leak_warnings(false);
    m.doc() = "DataStates-LLM Checkpoint Engine";

    nb::class_<datastates::core_t>(m, "core_engine")
        .def("ckpt", &ckpt_wrapper,
             "version"_a, "region_uid"_a, "obj"_a, "size"_a, "offset"_a, "path"_a)
        .def("restore", &restore_wrapper,
             "version"_a, "region_uid"_a, "obj"_a, "size"_a, "offset"_a, "path"_a)
        .def("wait", &datastates::core_t::wait, "persist"_a = false)
        .def("get_queue_stats", &datastates::core_t::get_queue_stats, "for_flush_queue"_a = true)
        .def("shutdown", &datastates::core_t::shutdown);
    
    // Factory function to create core engine instance (PIMPL)
    m.def("create_core_engine", &datastates::create_core_engine,
          nb::rv_policy::reference,
          "host_cache_size"_a, "gpu_id"_a, "rank"_a = -1,
          "use_io_uring"_a = false, "fs_block_alignment"_a = datastates::FS_BLOCK_SIZE_ALIGNMENT,
          "Create a new core engine instance.");

    
    // The following snippets pertain to the VLCC state-management system
    nb::class_<datastates::state_manager_t>(m, "state_manager")
        .def(nb::init<>())
        .def("add_var", &datastates::state_manager_t::add_var,
            "data"_a, "key"_a, "Add a variable to the state manager.")
        .def("print_state", &datastates::state_manager_t::print_state,
             "Print the state of all registered providers.")
        .def("release", &datastates::state_manager_t::release,
             "Release all resources held by the state manager.")
        .def("has_next_chunk", &datastates::state_manager_t::has_next_chunk,
             "Check if there is a next chunk available.");

    nb::class_<datastates::state_io_engine_t>(m, "state_io_engine")
        .def("ckpt", &datastates::state_io_engine_t::ckpt,
             "version"_a, "state"_a, "path"_a,
             "Checkpoint a state provider's data.")
        .def("restore", &datastates::state_io_engine_t::restore,
             "version"_a, "path"_a,
             "Restore a state provider's data from a checkpoint.")
        .def("wait", &datastates::state_io_engine_t::wait,
             "state"_a, "persist"_a = false, "Wait for all operations to complete.")
        .def("get_queue_stats", &datastates::state_io_engine_t::get_queue_stats,
             "for_flush_queue"_a = true, "Get the current queue statistics.")
        .def("shutdown", &datastates::state_io_engine_t::shutdown,
             "Shutdown the state I/O engine.");

    // We need to provide a factory function to create the singleton instance
    // because state_io_engine_t is an abstract class, and implemented through PIMPL.
    m.def("create_state_io_engine", &datastates::create_state_io_engine,
          nb::rv_policy::reference,
          "host_cache_size"_a, "gpu_id"_a, "rank"_a = -1,
          "use_io_uring"_a = false, "fs_block_alignment"_a = datastates::FS_BLOCK_SIZE_ALIGNMENT,
          "Create a new state I/O engine instance.");

    m.def("get_fs_block_alignment", &get_fs_block_alignment,
          "Get the current filesystem block size alignment.");
}
