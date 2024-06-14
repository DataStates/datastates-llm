#ifndef __DATASTATES_DEFS_HPP
#define __DATASTATES_DEFS_HPP

enum TIER_TYPES: int {
    HOST_UNPINNED_TIER=0,               // cudaMemoryTypeUnregistered = 0
    HOST_PINNED_TIER=1,                 // cudaMemoryTypeHost = 1
    GPU_TIER=2,                         // cudaMemoryTypeDevice = 2
    UNIFIED_MEM_TIER=3,                 // cudaMemoryTypeManaged = 3
    FILE_TIER=4
};

#endif // __DATASTATES_DEFS_HPP