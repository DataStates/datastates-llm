#ifndef __DATASTATES_POOL_ALLOCATOR_HPP
#define __DATASTATES_POOL_ALLOCATOR_HPP
#include <atomic>
#include <iostream>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <map>
#include <unordered_set>
#include "common/defs.hpp"
#include "common/mem_region.hpp"
#include "common/utils.hpp"

namespace datastates {
class mem_pool_t {
    char* start_ptr_ = nullptr;
    std::atomic<size_t> total_size_{0};
    std::atomic<size_t> curr_size_{0};
    std::atomic<size_t> head_{0};
    std::atomic<size_t> tail_{0};
    
    int device_type_ = -1;
    std::mutex mem_mutex_;
    std::condition_variable mem_cv_;
    std::deque<std::shared_ptr<mem_region_t>> mem_q_;
    bool is_active = true;
    int rank_ = -1;
    int gpu_id_ = -1;
    std::map<std::uint64_t, size_t> alloc_map_;
    void print_trace_();
    void assign_(std::shared_ptr<mem_region_t> m);
    std::unordered_set<std::uint64_t> deferred_deallocations_;
public:
    mem_pool_t(char* start_ptr, size_t total_size, int gpu_id=-1, int rank=-1, TIER_TYPES device_type = HOST_PINNED_TIER);
    ~mem_pool_t();    
    void allocate(std::shared_ptr<mem_region_t> m);
    size_t get_free_size();
    size_t get_capacity();
    void deallocate(std::shared_ptr<mem_region_t> m);
};

}

#endif //__DATASTATES_POOL_ALLOCATOR_HPP