// src/datastates_core.cpp (or similar)
#include "datastates.hpp"
#include "state_io_engine_impl.hpp"

static datastates::state_io_engine_t* instance = nullptr;
namespace datastates {
state_io_engine_t* create_io_engine(size_t host_cache_size, int gpu_id, int rank) {
    if (!instance) {
        instance = new state_io_engine_impl_t(host_cache_size, gpu_id, rank);
    }
    return instance;
}
}
