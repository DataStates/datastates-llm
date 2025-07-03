#ifndef __STATE_PICKLE_SERIALIZER_HPP
#define __STATE_PICKLE_SERIALIZER_HPP

#include "base_serializer.hpp"

namespace datastates {
class pickle_serializer_t : public base_serializer_t {
    const std::string name = "pickle";
    const nb::module_ pickle_module = nb::module_::import_("pickle");
    const nb::object serializer = pickle_module.attr("dumps");
    const nb::object deserializer = pickle_module.attr("loads");
public:
    pickle_serializer_t() = default;
    ~pickle_serializer_t() override = default;

    nb::bytes serialize(const nb::object& obj) override {
        return nb::cast<nb::bytes>(serializer(obj));
    }

    nb::object deserialize(const nb::bytes& data) override {
        return deserializer(data);
    }

    std::string get_name() const override {
        return name;
    }
};

} // namespace datastates

#endif // __STATE_PICKLE_SERIALIZER_HPP