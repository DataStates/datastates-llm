#ifndef __VLCC_STATES_REGION_HPP
#define __VLCC_STATES_REGION_HPP  

#include <cstddef>
#include <cstdint>
#include <vector>
#include "defs.hpp"

namespace vlcc_states{

struct state_region_t {
    uint64_t region_id; // Unique identifier for the memory region
    void* ptr;          // Pointer to the memory region
    size_t size;       // Size of the memory region in bytes
    TIER_TYPES tier;   // Tier type (e.g., CPU, GPU, UVM, etc.)
    STATE_PROVIDER_CHUNK_STATUS status = STATE_PROVIDER_UNREAD_CHUNK; // Status of the region

    state_region_t(uint64_t id, void* p, size_t s, TIER_TYPES t)
        : region_id(id), ptr(p), size(s), tier(t) {}
};

// struct state_chunk_t {
//     int chunk_id;
//     std::shared_ptr<void*> data_ptr;
//     size_t data_start_offset = 0; // Offset in the data region where this chunk starts
//     size_t chunk_size;
//     STATE_PROVIDER_CHUNK_STATUS chunk_status = STATE_PROVIDER_UNREAD_CHUNK;
    
//     state_chunk_t(std::shared_ptr<void*> ptr, size_t start_offset, size_t c_size)
//         : chunk_id(0), data_ptr(ptr), data_start_offset(start_offset), chunk_size(c_size) {}

//     state_chunk_t(int id, std::shared_ptr<void*> ptr, size_t start_offset, size_t c_size)
//         : chunk_id(id), data_ptr(ptr), data_start_offset(start_offset), chunk_size(c_size) {};

//     void set_chunk_status(STATE_PROVIDER_CHUNK_STATUS new_status) {
//         chunk_status = new_status;
//     }
// };

} // namespace vlcc_states

#endif // __VLCC_STATES_REGION_HPP