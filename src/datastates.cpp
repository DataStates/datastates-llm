// src/datastates_core.cpp (or similar)
#include "datastates.hpp"
#include "core_impl.hpp"

static datastates::core_impl_t* instance = nullptr;
namespace datastates {
core_t* dstates_engine(size_t host_cache_size, int gpu_id, int rank) {
    if (!instance) {
        instance = new core_impl_t(host_cache_size, gpu_id, rank);
    }
    return instance;
}
}
