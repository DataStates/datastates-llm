// src/datastates_core.cpp (or similar)
#include "datastates.hpp"
#include "state_io_engine_impl.hpp"


namespace datastates {
state_io_engine_t* create_io_engine(size_t host_cache_size, int gpu_id, int rank, bool use_io_uring, size_t fs_block_alignment) {
    if (!state_io_engine_instance) {
        state_io_engine_instance = new state_io_engine_impl_t(host_cache_size, gpu_id, rank, use_io_uring, fs_block_alignment);
    }
    return state_io_engine_instance;
}

// state_io_engine_t::~state_io_engine_t() {
//     if (state_io_engine_instance) {
//         delete state_io_engine_instance;
//         state_io_engine_instance = nullptr;
//     }
// }

} // namespace datastates