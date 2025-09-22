#ifndef __DATASTATES_MEM_REGION_HPP
#define __DATASTATES_MEM_REGION_HPP
#include <iostream>
#include <limits.h>
#include <deque>
#include <string>
#include <memory>
#include "defs.hpp"
#include "utils.hpp"

namespace datastates {

volatile static std::uint64_t internal_uid_counter = 1;
struct mem_region_t {
    std::uint64_t   version;            // Checkpoint version.
    std::uint64_t   uid;                // Unique memory region identifier.
    std::uint64_t   internal_uid;       // Internal unique identifier for the memory region, used for tracking in the pool.
    char*           ptr;                // Pointer of this memory region (can be either on GPU/CPU)
    size_t          size;               // Size of the memory region
    size_t          aligned_size;       // Aligned size of the memory region, used for file I/O
    size_t          file_start_offset;  // Start offset in file
    std::string     path;               // Pathname of the checkpoint file.
    TIER_TYPES      curr_tier_type;     // Memory/cache tier on which it currently resides
    mem_region_t(std::uint64_t version_, std::uint64_t uid_, char* ptr_, 
        size_t size_, size_t file_start_offset_, std::string path_, TIER_TYPES tier): 
        version(version_), uid(uid_), ptr(ptr_), size(size_), aligned_size(0), file_start_offset(file_start_offset_), 
        path(path_), curr_tier_type(tier) { 
            internal_uid = internal_uid_counter++;
        };
    mem_region_t(const mem_region_t* other, TIER_TYPES next_tier): 
        version(other->version), uid(other->uid), ptr(nullptr), size(other->size), aligned_size(other->aligned_size), 
        file_start_offset(other->file_start_offset), path(other->path), curr_tier_type(next_tier), internal_uid(other->internal_uid) {};
    mem_region_t(const std::shared_ptr<mem_region_t> other, TIER_TYPES next_tier): 
        version(other->version), uid(other->uid), ptr(nullptr), size(other->size), aligned_size(other->aligned_size), 
        file_start_offset(other->file_start_offset), path(other->path), curr_tier_type(next_tier), internal_uid(other->internal_uid){};
    mem_region_t& operator=(const mem_region_t&) = delete;
};

}

#endif //__DATASTATES_MEM_REGION_HPP