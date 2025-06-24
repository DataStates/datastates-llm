#ifndef __DATASTATES_BASE_SERIALIZER_HPP
#define __DATASTATES_BASE_SERIALIZER_HPP

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>  
#include <pybind11/stl.h>

namespace py = pybind11;

class base_serializer_t {
public:
    base_serializer_t() = default;
    virtual ~base_serializer_t() = default;

    // Serialize an object to a byte string
    virtual py::bytes serialize(const py::object& obj) = 0;

    // Deserialize a byte string back to an object
    virtual py::object deserialize(const py::bytes& data) = 0;

    // Get the name of the serializer
    virtual std::string get_name() const = 0;
};

#endif // __DATASTATES_BASE_SERIALIZER_HPP