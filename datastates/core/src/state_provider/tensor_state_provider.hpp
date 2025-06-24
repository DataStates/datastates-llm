#ifndef __DATASTATES_TENSOR_PROVIDER_HPP
#define __DATASTATES_TENSOR_PROVIDER_HPP

#include "common/d_object.hpp"
#include "common/defs.hpp"
#define NDEBUG
#include <cassert>
#include <pybind/pybind11.h>
#include "base_state_provider.hpp"

class tensor_state_provider_t: public base_state_provider_t {
    static py::object torch = py::module_::import("torch");
    uint64_t region_id = 0; // Unique identifier for the memory region

    public:
    tensor_state_provider_t(std::string name, TIER_TYPES device_type, size_t c_size = STATE_PROVIDER_DEFAULT_CHUNK_SIZE)
        : base_state_provider_t(name, device_type, c_size), region_id(0) {}

    void register_data_object(py::object d_object, uint64_t r_id, std::string serializer_name = "") override {
        assert(!d_object.is_none());
        assert(data_object.is_none());
        assert(py::isinstance(d_object, torch.attr("Tensor")));
        assert(registered_region_ids.count(r_id) == 0 && "The specified region ID already registered for another data object");

        std::string dev_type = d_object.attr("device").attr("type").cast<std::string>();
        assert(dev_type == "cpu" || dev_type == "cuda");

        void* ptr = d_object.attr("data_ptr")().cast<void*>();
        cudaPointerAttributes attributes;
        cudaPointerGetAttributes(&attributes, ptr);
        assert(attributes.type == provider_device);

        region_id = r_id;
        data_object = d_object;
    }

    void add_child_provider(tensor_state_provider_t* child) {
        throw std::runtime_error("Tensor state provider does not support adding child providers. Use composite state provider instead.");
    }

    mem_region_t* get_next_chunk(TIER_TYPES tier_type) {
        assert(!data_object.is_none() && provider_name + " state provider has no data_object state registered");
        if (tier_type == provider_device && data_status == STATE_PROVIDER_UNREAD_CHUNK) {
            data_status = STATE_PROVIDER_CONSUMING_CHUNK;
            void* ptr = data_object.attr("data_ptr")().cast<void*>();
            cudaPointerAttributes attributes;
            cudaPointerGetAttributes(&attributes, ptr);
            size_t tensor_size = data_object.attr("numel")().cast<size_t>() * data_object.attr("element_size")().cast<size_t>();
            data_region = new mem_region_t(version, region_id, static_cast<char*>(ptr), tensor_size, 0, "", provider_device);
            return data_region;
        }
        return nullptr; 
    }

    void release_chunk(mem_region_t* chunk) {
        assert(chunk != nullptr && "Chunk to release cannot be null");
        assert(data_status == STATE_PROVIDER_CONSUMING_CHUNK && "Chunk must be in consuming state before releasing");
        data_status = STATE_PROVIDER_CONSUMED_CHUNK;
        data_region.reset();
    }


};


#endif //__DATASTATES_TENSOR_PROVIDER_HPP