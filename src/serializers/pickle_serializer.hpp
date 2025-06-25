#ifndef __DATASTATES_PICKLE_SERIALIZER_HPP
#define __DATASTATES_PICKLE_SERIALIZER_HPP

#include "base_serializer.hpp"


class pickle_serializer_t : public base_serializer_t {
    std::string name = "pickle";
    py::module_ pickle = py::module_::import("pickle");
    py::object serializer = pickle.attr("dumps");
    py::object deserializer = pickle.attr("loads");
public:
    pickle_serializer_t() = default;
    ~pickle_serializer_t() override = default;

    py::bytes serialize(const py::object& obj) override {
        return serializer(obj);
    }

    py::object deserialize(const py::bytes& data) override {
        return deserializer(data);
    }

    std::string get_name() const override {
        return name;
    }
};

#endif // __DATASTATES_PICKLE_SERIALIZER_HPP