#ifndef __DATASTATES_DEFS_HPP
#define __DATASTATES_DEFS_HPP
typedef unsigned int uint;
enum TIER_TYPES: int {
    HOST_UNPINNED_TIER=0,               // cudaMemoryTypeUnregistered = 0
    HOST_PINNED_TIER=1,                 // cudaMemoryTypeHost = 1
    GPU_TIER=2,                         // cudaMemoryTypeDevice = 2
    UNIFIED_MEM_TIER=3,                 // cudaMemoryTypeManaged = 3
    COMPOSITE_TIER=4,                   // Tier for composite providers
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



#endif // __DATASTATES_DEFS_HPP