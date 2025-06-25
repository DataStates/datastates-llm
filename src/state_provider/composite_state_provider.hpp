#ifndef __DATASTATES_COMPOSITE_PROVIDER_HPP
#define __DATASTATES_COMPOSITE_PROVIDER_HPP

#include "common/d_object.hpp"
#include "common/defs.hpp"
#define NDEBUG
#include <cassert>
#include <pybind/pybind11.h>
#include "serializers/pickle_serializer.hpp"
#include "base_state_provider.hpp"


class composite_state_provider_t: public base_state_provider_t {
    public:
    composite_state_provider_t(std::string name, TIER_TYPES device_type, size_t c_size = STATE_PROVIDER_DEFAULT_CHUNK_SIZE) 
        : base_state_provider_t(name, device_type, c_size), provider_device(COMPOSED_STATE_TIER) {}
    
    ~composite_state_provider_t() override {
        delete serializer;
        if (data_region != nullptr) {
            delete data_region;
        }
        for (auto& child : child_providers) {
            delete child;
        }
    }

    void register_data_object(py::object d_object, uint64_t region_id, std::string serializer_name = "") override {
        throw std::runtime_error("Composite state provider does not support registering data objects directly. Use child providers instead.");
    }

    void add_child_provider(base_state_provider_t* child) {
        assert(child != nullptr && "Child provider cannot be null");
        assert(data_object == nullptr && provider_name + " state provider can either register a data_object state or a child provider, not both");
        child_providers.push_back(child);
    }

    mem_region_t* get_next_chunk(TIER_TYPES tier_type) {
        assert(!child_providers.empty() && provider_name + " state provider must have either a data_object state or child providers registered");
        for (auto& child : child_providers) {
            if (child->get_tier_type() == tier_type) {
                auto next_chunk = child->get_next_chunk(tier_type);
                if (next_chunk != nullptr) {
                    return next_chunk;
                }
            }
        }
        return nullptr; 
    }
};


#endif //__DATASTATES_COMPOSITE_PROVIDER_HPP