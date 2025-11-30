// src/datastates_core.cpp (or similar)
#include "datastates.hpp"
#include "state_io_engine_impl.hpp"


namespace datastates {
state_io_engine_t* create_state_io_engine(size_t host_cache_size, int gpu_id, int rank, bool use_io_uring, size_t fs_block_alignment) {
    if (!is_state_io_engine_active)
        return nullptr;
    if (!state_io_engine_instance) {
        state_io_engine_instance = std::make_shared<state_io_engine_impl_t>(host_cache_size, gpu_id, rank, use_io_uring, fs_block_alignment);
    }
    return state_io_engine_instance.get();
}

} // namespace datastates