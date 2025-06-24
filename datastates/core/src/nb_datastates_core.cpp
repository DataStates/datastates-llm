#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/ndarray.h>
#include <torch/extension.h>  
#include "engine.hpp"

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

void ckpt_wrapper(datastates_core_t& self, int version, nb::handle obj, uint64_t size, uint64_t offset, std::string path) {
    auto [ptr, actual_size] = extract_ptr_and_size(obj);
    if (size > actual_size)
        throw std::runtime_error("Requested size exceeds buffer size");
    self.ckpt(version, ptr, size, offset, path);
}

void restore_wrapper(datastates_core_t& self, int version, nb::handle obj, uint64_t size, uint64_t offset, std::string path) {
    auto [ptr, actual_size] = extract_ptr_and_size(obj);
    if (size > actual_size)
        throw std::runtime_error("Requested size exceeds buffer size");
    self.restore(version, ptr, size, offset, path);
}

NB_MODULE(datastates_core, m) {
    m.doc() = "DataStates-LLM Checkpoint Engine";

    nb::class_<datastates_core_t>(m, "handle")
        .def(nb::init<const size_t, int, int>())
        .def("ckpt", &ckpt_wrapper, 
             "version"_a, "obj"_a, "size"_a, "offset"_a, "path"_a,
             "Checkpoint a tensor or buffer to a file.")
        .def("restore", &restore_wrapper, 
             "version"_a, "obj"_a, "size"_a, "offset"_a, "path"_a,
             "Restore a tensor or buffer from a file.")
        .def("wait", &datastates_core_t::wait)
        .def("shutdown", &datastates_core_t::shutdown);
}
