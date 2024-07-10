#ifndef __DATASTATES_POOL_ALLOCATOR_HPP
#define __DATASTATES_POOL_ALLOCATOR_HPP
#include <atomic>
#include <iostream>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <cassert>
#include <map>
#include "common/defs.hpp"
#include "common/mem_region.hpp"
#include "common/utils.hpp"

class mem_pool_t {
    char* start_ptr_ = nullptr;
    std::atomic<size_t> total_size_{0};
    std::atomic<size_t> curr_size_{0};
    std::atomic<size_t> head_{0};
    std::atomic<size_t> tail_{0};
    
    int device_type_ = -1;
    std::mutex mem_mutex_;
    std::condition_variable mem_cv_;
    std::deque<mem_region_t*> mem_q_;
    std::map<uint64_t, size_t> alloc_map_;
    bool is_active = true;
    int rank_ = -1;
    void print_trace_();
    void assign_(mem_region_t* m);
public:
    mem_pool_t(char* start_ptr, size_t total_size, int rank=-1);
    ~mem_pool_t();    
    void allocate(mem_region_t* m);
    size_t get_free_size();
    size_t get_capacity();
    void deallocate(mem_region_t* m);
};


#endif //__DATASTATES_POOL_ALLOCATOR_HPP