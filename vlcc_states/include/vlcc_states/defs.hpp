#ifndef __VLCC_STATES_DEFS_HPP
#define __VLCC_STATES_DEFS_HPP
#include <cuda_runtime.h>

namespace vlcc_states {
const size_t STATE_PROVIDER_DEFAULT_CHUNK_SIZE = 64 * (1<<20);

enum STATE_PROVIDER_CHUNK_STATUS: int {
    STATE_PROVIDER_UNREAD_CHUNK = 0,
    STATE_PROVIDER_CONSUMING_CHUNK = 1,
    STATE_PROVIDER_CONSUMED_CHUNK = 2
};

enum TIER_TYPES: int {
    HOST_UNPINNED_TIER=0,               // cudaMemoryTypeUnregistered = 0
    HOST_PINNED_TIER=1,                 // cudaMemoryTypeHost = 1
    GPU_TIER=2,                         // cudaMemoryTypeDevice = 2
    UNIFIED_MEM_TIER=3,                 // cudaMemoryTypeManaged = 3
    COMPOSITE_TIER=4,                   // Tier for composite providers
    FILE_TIER=5
};
} // namespace vlcc_states

#endif // __VLCC_STATES_DEFS_HPP