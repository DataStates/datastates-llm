#ifndef __VLCC_STATES_OBJECT_PROVIDER_HPP
#define __VLCC_STATES_OBJECT_PROVIDER_HPP

#include "state_provider.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "serializers/pickle_serializer.hpp"

namespace vlcc_states {
static nb::object torch = nb::module_::import_("torch");
class object_state_provider_t: public state_provider_t {
        pickle_serializer_t* serializer = nullptr;
        nb::object data_object = nb::none(); // The Python object to be serialized
        bool is_serialized = false; // Flag to check if the object is serialized
        bool is_tensor = false; // Flag to check if the object is a tensor
    public:
    object_state_provider_t(std::string name, TIER_TYPES device_type) 
        : state_provider_t(name, device_type) {}

    object_state_provider_t(std::string name, TIER_TYPES device_type, nb::object d_object)
        : state_provider_t(name, device_type){
        register_state(d_object);
    }

    ~object_state_provider_t() override {
        delete serializer;
        if (data_region != nullptr) {
            data_region.reset();
        }
    }

    void register_state(nb::object d_object) {
        assert((!d_object.is_none() && d_object != nb::none()) && "Object to register cannot be null");
        assert(data_object.is_none() && "State provider has already registered a data_object state");
        is_tensor = nb::isinstance<nb::ndarray<>>(d_object) || nb::cast<bool>(torch.attr("is_tensor")(d_object));
        is_serialized = nb::isinstance<nb::bytes>(d_object);
        if ((!is_serialized && !is_tensor) && serializer == nullptr) {
            serializer = new pickle_serializer_t();
        }
        data_object = d_object;
    }

    std::shared_ptr<state_region_t> get_next_chunk(TIER_TYPES tier) {
        assert(!data_object.is_none() && "Data object is not registered");
        if (tier == provider_device && data_status == STATE_PROVIDER_UNREAD_CHUNK) {
            data_status = STATE_PROVIDER_CONSUMING_CHUNK;
            void *data_ptr = nullptr;
            size_t size = 0;
            if (is_tensor) {
                nb::ndarray<> arr = nb::cast<nb::ndarray<>>(data_object);
                data_ptr = arr.data();
                size = arr.size() * arr.itemsize();
            } else if (is_serialized) {
                nb::bytes serialized_data = nb::cast<nb::bytes>(data_object);
                data_ptr = const_cast<void*>(serialized_data.data());
                size = serialized_data.size();
            } else {
                assert(serializer != nullptr && "Serializer must be initialized for non-serialized objects");
                nb::bytes serialized_data = serializer->serialize(data_object);
                data_ptr = const_cast<void*>(serialized_data.data());
                size = serialized_data.size();
            }
            data_region = std::make_shared<state_region_t>(0, data_ptr, size, provider_device);
            return data_region;
        }
        return nullptr; 
    }

    void release() override {
        assert(data_region != nullptr && "Region to release cannot be null");
        assert(data_status == STATE_PROVIDER_CONSUMING_CHUNK && "Region must be in consuming state before releasing");
        data_status = STATE_PROVIDER_UNREAD_CHUNK;
        data_region.reset();
    }

};

} // namespace vlcc_states

#endif //__VLCC_STATES_OBJECT_PROVIDER_HPP