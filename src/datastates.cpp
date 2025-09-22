// src/datastates_core.cpp (or similar)
#include "datastates.hpp"
#include "common/utils.hpp"
#include "core_impl.hpp"
#include "common/perf_profiler.hpp"

namespace datastates {
static core_t* instance {nullptr};
core_t* dstates_engine(size_t host_cache_size, int gpu_id, int rank, bool use_io_uring, size_t fs_block_alignment) {
    if (FS_BLOCK_SIZE_ALIGNMENT != fs_block_alignment) {
        FATAL("Cannot change FS Block size other than the default value of " + std::to_string(FS_BLOCK_SIZE_ALIGNMENT) + " bytes.");
    }
    if (!instance) {
        instance = new core_impl_t(host_cache_size, gpu_id, rank, use_io_uring, fs_block_alignment);
    }
    return instance;
}
}