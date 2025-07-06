#ifndef __DATASTATES_MEM_REGION_HPP
#define __DATASTATES_MEM_REGION_HPP
#include <iostream>
#include <limits.h>
#include <deque>
#include <string>
#include "defs.hpp"

static uint internal_uid_counter = 1;
struct mem_region_t {
    uint            version;            // Checkpoint version.
    uint            uid;                // Unique memory region identifier.
    uint            internal_uid;       // Internal unique identifier for the memory region, used for tracking in the pool.
    char*           ptr;                // Pointer of this memory region (can be either on GPU/CPU)
    size_t          size;               // Size of the memory region
    size_t          file_start_offset;  // Start offset in file
    std::string     path;               // Pathname of the checkpoint file.
    TIER_TYPES      curr_tier_type;     // Memory/cache tier on which it currently resides
    mem_region_t(uint version_, uint uid_, char* ptr_, 
        size_t size_, size_t file_start_offset_, std::string path_, TIER_TYPES tier): 
        version(version_), uid(uid_), ptr(ptr_), size(size_), file_start_offset(file_start_offset_), path(path_), curr_tier_type(tier) { internal_uid = internal_uid_counter++; };
    mem_region_t(const mem_region_t* other, TIER_TYPES next_tier): version(other->version), uid(other->uid), ptr(nullptr), size(other->size), file_start_offset(other->file_start_offset), path(other->path), curr_tier_type(next_tier), internal_uid(other->internal_uid) {};
    mem_region_t& operator=(const mem_region_t&) = delete;
};

#endif //__DATASTATES_MEM_REGION_HPP