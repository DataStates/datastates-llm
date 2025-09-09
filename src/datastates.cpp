// src/datastates_core.cpp (or similar)
#include "datastates.hpp"
#include "common/utils.hpp"
#include "core_impl.hpp"
#include "common/perf_profiler.hpp"

size_t FS_BLOCK_SIZE_ALIGNMENT = 4096; // Set default filesystem block size alignment
bool USE_URING = false; // Default to not using io_uring
namespace datastates {
static core_t* instance {nullptr};
core_t* dstates_engine(size_t host_cache_size, int gpu_id, int rank) {
    if (!instance) {
        instance = new core_impl_t(host_cache_size, gpu_id, rank);
    }
    return instance;
}
}