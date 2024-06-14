#include <torch/extension.h>
#include "engine.hpp"
#include <pybind11/iostream.h>

namespace py = pybind11;
// PYBIND11_MODULE(_datastates, m) {
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = R"pbdoc(
        DataStates-LLM
        -----------------------
        .. currentmodule:: datastates
        .. autosummary::
           :toctree: _generate
           ckpt_tensor
           restore_tensor
           wait
           shutdown
    )pbdoc";

    py::class_<datastates_llm_t>(m, "handle")
        .def(py::init<const size_t, int, int>())
        .def("ckpt_tensor", &datastates_llm_t::ckpt_tensor, py::call_guard<py::gil_scoped_release>())
        .def("restore_tensor", &datastates_llm_t::restore_tensor, py::call_guard<py::gil_scoped_release>())
        .def("wait", &datastates_llm_t::wait, py::call_guard<py::gil_scoped_release>())
        .def("shutdown", &datastates_llm_t::shutdown);
}