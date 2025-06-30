#ifndef __VLCC_STATES_RAWPTR_PROVIDER_HPP
#define __VLCC_STATES_RAWPTR_PROVIDER_HPP

#include <state_provider.hpp>
#define NDEBUG
#include <cassert>

namespace vlcc_states {
class rawptr_state_provider_t: public state_provider_t {
    private:
        void* data_ptr = nullptr;
        size_t data_size = 0;
    public:
    rawptr_state_provider_t(std::string name, TIER_TYPES device_type)
        : state_provider_t(name, device_type) {}
    rawptr_state_provider_t(std::string name, TIER_TYPES device_type, void* ptr, size_t size, bool is_serialized=true)
        : state_provider_t(name, device_type) {
        register_state(ptr, size, is_serialized);
    }

    ~rawptr_state_provider_t() override {
        if (data_ptr) {
            data_ptr = nullptr;
            data_size = 0;
        }
    }

    void register_state(void* ptr, size_t size, bool is_serialized=true) {
        assert(ptr != nullptr && "Pointer to register cannot be null");
        assert(size > 0 && "Size of the data to register must be greater than zero");
        assert(data_ptr == nullptr && "State provider has already registered a data pointer");
        assert(is_serialized == true && "Raw pointer provider does not yet support unserialized datastructures");
        data_ptr = ptr;
        data_size = size;
    }

    std::shared_ptr<state_region_t> get_next_chunk(TIER_TYPES tier) override {
        assert(data_ptr && "Data pointer is not registered");
        assert(data_size > 0 && "Data size must be greater than zero");
        
        if (tier == provider_device && data_status == STATE_PROVIDER_UNREAD_CHUNK) {
            data_status = STATE_PROVIDER_CONSUMING_CHUNK;
            data_region = std::make_shared<state_region_t>(0, data_ptr, data_size, provider_device);
            return data_region;
        }
        return nullptr; 
    }

    void release() override {
        assert(data_ptr != nullptr && "Data pointer to release cannot be null");
        assert(data_size > 0 && "Data size must be greater than zero before releasing");
        assert(data_region != nullptr && "Region to release cannot be null");
        assert(data_status == STATE_PROVIDER_CONSUMING_CHUNK && "Region must be in consuming state before releasing");
        data_status = STATE_PROVIDER_UNREAD_CHUNK;
        data_region.reset();
    }
};

} // namespace vlcc_states


#endif //__VLCC_STATES_RAWPTR_PROVIDER_HPP