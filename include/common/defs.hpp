#ifndef __DATASTATES_DEFS_HPP
#define __DATASTATES_DEFS_HPP
#include <cstddef>
#include <cuda_runtime.h>

namespace datastates {

enum TIER_TYPES: int {
    HOST_UNPINNED_TIER=cudaMemoryTypeUnregistered,          // cudaMemoryTypeUnregistered = 0
    HOST_PINNED_TIER=cudaMemoryTypeHost,                    // cudaMemoryTypeHost = 1
    GPU_TIER=cudaMemoryTypeDevice,                          // cudaMemoryTypeDevice = 2
    UNIFIED_MEM_TIER=cudaMemoryTypeManaged,                 // cudaMemoryTypeManaged = 3
    COMPOSITE_TIER=4,                                       // Tier for composite providers
    FILE_TIER=5
};

static const char* TIER_TYPE_NAMES[] = {
    "HOST_UNPINNED_TIER",
    "HOST_PINNED_TIER",
    "GPU_TIER",
    "UNIFIED_MEM_TIER",
    "COMPOSITE_TIER",
    "FILE_TIER"
};

enum STATE_PROVIDER_CHUNK_STATUS: int {
    STATE_PROVIDER_UNREAD_CHUNK = 0,
    STATE_PROVIDER_CONSUMING_CHUNK = 1,
    STATE_PROVIDER_CONSUMED_CHUNK = 2
};

const size_t STATE_PROVIDER_DEFAULT_CHUNK_SIZE = 64 * (1<<20);
const size_t MAX_FILE_WRITE_SIZE = (1ULL << 26); // 64 MB
static constexpr size_t MAX_OPEN_FDS = 2048;
const size_t MAX_PENDING_FLUSHES_SIZE = (1ULL << 22); // 512 MB
const size_t WAIT_TIMEOUT_US = 1000*1000;
const size_t FS_BLOCK_SIZE_ALIGNMENT = 4096; // Set default filesystem block size alignment
}

#endif // __DATASTATES_DEFS_HPP