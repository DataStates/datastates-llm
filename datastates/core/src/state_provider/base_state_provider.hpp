#ifndef __DATASTATES_BASE_PROVIDER_HPP
#define __DATASTATES_BASE_PROVIDER_HPP

#include "common/d_object.hpp"
#include "common/defs.hpp"
#define NDEBUG
#include <cassert>
#include <pybind/pybind11.h>


const size_t STATE_PROVIDER_DEFAULT_CHUNK_SIZE = 64 * (1<<20);

enum STATE_PROVIDER_CHUNK_STATUS: int {
    STATE_PROVIDER_UNREAD_CHUNK = 0,
    STATE_PROVIDER_CONSUMING_CHUNK = 1,
    STATE_PROVIDER_CONSUMED_CHUNK = 2
};

enum class StateDataType { TENSOR, GENERIC, COMPOSITE };

class base_state_provider_t {
protected:
    std::string provider_name;
    TIER_TYPES provider_device;
    size_t chunk_size;
    py::object data_object;
    std::unique_ptr<mem_region_t> data_region;
    STATE_PROVIDER_CHUNK_STATUS data_status;
    int version;
    static std::set<uint64_t> registered_region_ids;

public:
    base_state_provider_t(std::string name, TIER_TYPES device_type, size_t c_size)
        : provider_name(name), provider_device(device_type), chunk_size(c_size),
          data_object(py::none()), data_status(STATE_PROVIDER_UNREAD_CHUNK), version(0) {}

    virtual ~base_state_provider_t() = default;

    virtual void begin_state_capture(int version_) {
        assert(version_ >= 0);
        assert(data_status == STATE_PROVIDER_UNREAD_CHUNK);
        version = version_;
    }

    virtual void register_data_object(py::object d_object, uint64_t region_id, std::string serializer_name = "") = 0;
    virtual mem_region_t* get_next_chunk(TIER_TYPES tier_type) = 0;
    virtual void release_chunk(mem_region_t* chunk) = 0;

    virtual TIER_TYPES get_tier_type() const {
        return provider_device;
    }

    virtual void end_state_capture() {
        assert(version >= 0);
        data_status = STATE_PROVIDER_UNREAD_CHUNK;
    }
};


#endif //__DATASTATES_BASE_PROVIDER_HPP