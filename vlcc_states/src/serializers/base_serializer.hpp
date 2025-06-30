#ifndef __VLCC_STATES_BASE_SERIALIZER_HPP
#define __VLCC_STATES_BASE_SERIALIZER_HPP

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
namespace nb = nanobind;

namespace vlcc_states {

class base_serializer_t {
public:
    base_serializer_t() = default;
    virtual ~base_serializer_t() = default;

    // Serialize an object to a byte string
    virtual nb::bytes serialize(const nb::object& obj) = 0;

    // Deserialize a byte string back to an object
    virtual nb::object deserialize(const nb::bytes& data) = 0;

    // Get the name of the serializer
    virtual std::string get_name() const = 0;
};
}  // namespace vlcc_states

#endif // __VLCC_STATES_BASE_SERIALIZER_HPP