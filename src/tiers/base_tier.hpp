#ifndef __DATASTATES_BASE_TIER_HPP
#define __DATASTATES_BASE_TIER_HPP

#include "common/atomic_queue.hpp"
#include "pool/mem_pool.hpp"
#include <thread>


class base_tier_t {
protected:
    // We keep a separate GPU ID because even for host-memory, we need to first `cudaSetDevice` 
    // to map pinned-host memory to a given GPU ID, otherwise the CUDA context for this host-tier will
    // be generated on GPU-0
    int gpu_id_ = -1;
    unsigned int num_threads_ = 0;
    size_t total_size_ = 0;
    base_tier_t* successor_tier_ = nullptr;
    std::thread flush_thread_;
    std::thread fetch_thread_;
    std::atomic<bool> is_active{true};
    atomic_queue_t flush_q;
    atomic_queue_t fetch_q;
public:
    TIER_TYPES tier_type_;
    mem_pool_t* mem_pool = nullptr;
    base_tier_t(TIER_TYPES tier_type, int gpu_id, unsigned int num_threads, size_t total_size): 
        gpu_id_(gpu_id), num_threads_(num_threads), total_size_(total_size), tier_type_(tier_type) {};
    virtual ~base_tier_t() {};
    virtual void flush(mem_region_t* src) = 0;
    virtual void fetch(mem_region_t* src) = 0;
    virtual void wait_for_completion() = 0;
    virtual void set_successor_tier(base_tier_t* tier) {
        successor_tier_  = tier;
    };
    virtual void flush_io_() = 0;
    virtual void fetch_io_() = 0;
};

#endif //__DATASTATES_BASE_TIER_HPP