#ifndef __DATASTATES_MEM_REGION_HPP
#define __DATASTATES_MEM_REGION_HPP
#include <iostream>
#include <limits.h>
#include <deque>
#include "defs.hpp"


struct mem_region_t {
    const int           version;            // Checkpoint version.
    const uint64_t      uid;                // Unique memory region identifier.
    char*               ptr;                // Pointer of this memory region (can be either on GPU/CPU)
    const size_t        size;               // Size of the memory region
    const size_t        file_start_offset;  // Start offset in file
    const std::string   path;               // Pathname of the checkpoint file.
    TIER_TYPES          curr_tier_type;     // Memory/cache tier on which it currently resides
    mem_region_t(const int version_, const uint64_t uid_, char* const ptr_, 
        const size_t size_, const size_t file_start_offset_, const std::string path_, TIER_TYPES tier): 
        version(version_), uid(uid_), ptr(ptr_), size(size_), file_start_offset(file_start_offset_), path(path_), curr_tier_type(tier) {};
    mem_region_t(const mem_region_t* other, TIER_TYPES next_tier): version(other->version), uid(other->uid), ptr(nullptr), size(other->size), file_start_offset(other->file_start_offset), path(other->path), curr_tier_type(next_tier) {};
    mem_region_t& operator=(const mem_region_t&) = delete;
};

#endif //__DATASTATES_MEM_REGION_HPP