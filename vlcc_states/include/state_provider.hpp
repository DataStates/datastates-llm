#ifndef __VLCC_STATES_PROVIDER_HPP
#define __VLCC_STATES_PROVIDER_HPP

#include "vlcc_states/defs.hpp"
#include "vlcc_states/state_region.hpp"
#include <iostream>
#include <string>
#include <memory>
#define NDEBUG
#include <cassert>
#include <nanobind/nanobind.h>


namespace vlcc_states {
class state_provider_t {
protected:
    std::string provider_name;
    TIER_TYPES provider_device; 
    STATE_PROVIDER_CHUNK_STATUS data_status = STATE_PROVIDER_UNREAD_CHUNK;
    std::shared_ptr<state_region_t> data_region = nullptr;
    // typedef std::vector<state_chunk_t> state_chunks_t; // Maybe we don't need this- revisit later.
public:
    state_provider_t(std::string name, TIER_TYPES device_type)
        : provider_name(std::move(name)), provider_device(device_type) {}
    virtual ~state_provider_t() = default;
    std::string get_name();
    TIER_TYPES get_tier(); /* E.g. GPU, CPU, UVM, etc. */
    virtual std::shared_ptr<state_region_t> get_next_chunk(TIER_TYPES tier) = 0;
    virtual void release() = 0;
};

std::shared_ptr<state_provider_t> create_rawptr_provider(const std::string& name, TIER_TYPES tier, void* ptr, size_t size);

std::shared_ptr<state_provider_t> create_object_provider(const std::string& name, TIER_TYPES tier, nanobind::object py_object);

std::shared_ptr<state_provider_t> create_composite_provider(const std::string& name, TIER_TYPES tier);
} // namespace vlcc_states

#endif //__VLCC_STATES_PROVIDER_HPP

// class base_state_provider_t {
// protected:
//     std::string provider_name;
//     TIER_TYPES provider_device;
//     size_t chunk_size;
//     nanobind::object data_object;
//     std::unique_ptr<mem_region_t> data_region;
//     STATE_PROVIDER_CHUNK_STATUS data_status;
//     int version;
//     static std::set<uint64_t> registered_region_ids;

// public:
//     base_state_provider_t(std::string name, TIER_TYPES device_type, size_t c_size)
//         : provider_name(name), provider_device(device_type), chunk_size(c_size),
//           data_object(py::none()), data_status(STATE_PROVIDER_UNREAD_CHUNK), version(0) {}

//     virtual ~base_state_provider_t() = default;

//     virtual void begin_state_capture(int version_) {
//         assert(version_ >= 0);
//         assert(data_status == STATE_PROVIDER_UNREAD_CHUNK);
//         version = version_;
//     }

//     virtual void register_data_object(py::object d_object, uint64_t region_id, std::string serializer_name = "") = 0;
//     virtual mem_region_t* get_next_chunk(TIER_TYPES tier_type) = 0;
//     virtual void release_chunk(mem_region_t* chunk) = 0;

//     virtual TIER_TYPES get_tier_type() const {
//         return provider_device;
//     }

//     virtual void end_state_capture() {
//         assert(version >= 0);
//         data_status = STATE_PROVIDER_UNREAD_CHUNK;
//     }
// };


