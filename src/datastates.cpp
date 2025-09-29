// src/datastates_core.cpp (or similar)
#include "datastates.hpp"
#include "common/utils.hpp"
#include "core_impl.hpp"
#include "common/perf_profiler.hpp"

namespace datastates {
core_t* create_core_engine(size_t host_cache_size, int gpu_id, int rank, bool use_io_uring, size_t fs_block_alignment) {
    if (!is_core_engine_active)
        return nullptr;
    if (FS_BLOCK_SIZE_ALIGNMENT != fs_block_alignment) {
        FATAL("Cannot change FS Block size other than the default value of " + std::to_string(FS_BLOCK_SIZE_ALIGNMENT) + " bytes.");
    }
    if (!core_engine_instance) {
        core_engine_instance = std::make_shared<core_impl_t>(host_cache_size, gpu_id, rank, use_io_uring, fs_block_alignment);
    }
    return core_engine_instance.get();
}
}