#ifndef __DATASTATES_GENERIC_PROVIDER_HPP
#define __DATASTATES_GENERIC_PROVIDER_HPP

#include "common/d_object.hpp"
#include "common/defs.hpp"
#define NDEBUG
#include <cassert>
#include <pybind/pybind11.h>
#include "serializers/pickle_serializer.hpp"
#include "base_state_provider.hpp"


class generic_state_provider_t: public base_state_provider_t {
    static py::object torch = py::module_::import("torch");
    base_serializer_t* serializer = new pickle_serializer_t();
    uint64_t region_id = 0; // Unique identifier for the memory region

    public:
    generic_state_provider_t(std::string name, TIER_TYPES device_type, size_t c_size = STATE_PROVIDER_DEFAULT_CHUNK_SIZE) 
        : base_state_provider_t(name, device_type, c_size) {}
    
    ~generic_state_provider_t() override {
        delete serializer;
        if (data_region != nullptr) {
            free(data_region->ptr);
            delete data_region;
        }
    }

    void register_data_object(py::object d_object, uint64_t region_id, std::string serializer_name = "pickle") {
        assert((!d_object.is_none() && d_object != NULL) && "Object to register cannot be null");
        assert(data_object.is_none() && "State provider has already registered a data_object state");
        assert((!py::isinstance(d_object, torch.attr("Tensor")) && 
               !py::isinstance(d_object, py::module_::import("torch").attr("nn").attr("Module"))) && 
               "Use tensor_state_provider_t for PyTorch tensors or modules");
        assert(registered_region_ids.count(region_id) == 0 && "The specified region ID already registered for another data object");
        assert(serializer_name == "pickle" && "Only pickle serializer is supported for generic state provider");
        data_object = d_object;
    }

    void add_child_provider(base_state_provider_t* child) {
        throw std::runtime_error("Generic state provider does not support adding child providers. Use composite state provider instead.");
    }

    mem_region_t* get_next_chunk(TIER_TYPES tier_type) {
        assert(!data_object.is_none() && provider_name + " state provider has no data_object state registered");
        if (tier_type == provider_device && data_status == STATE_PROVIDER_UNREAD_CHUNK) {
            data_status = STATE_PROVIDER_CONSUMING_CHUNK;
            py::bytes serialized_data = serializer.serialize(data_object);
            size_t size = serialized_data.size();
            data_region = new mem_region_t(version, region_id, const_cast<char*>(serialized_data.data()), size, 0, "", provider_device);
            return data_region;
        }
        return nullptr; 
    }

    void release_chunk(mem_region_t* chunk) {
        assert(chunk != nullptr && "Chunk to release cannot be null");
        assert(data_status == STATE_PROVIDER_CONSUMING_CHUNK && "Chunk must be in consuming state before releasing");
        data_status = STATE_PROVIDER_CONSUMED_CHUNK;
        if (data_region != nullptr) {
            free(data_region->ptr);
            delete data_region;
            data_region = nullptr;
        }
    }

};


#endif //__DATASTATES_GENERIC_PROVIDER_HPP