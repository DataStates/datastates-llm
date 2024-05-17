#ifndef __DATASTATES_GPU_TIER_HPP
#define __DATASTATES_GPU_TIER_HPP

#include "base_tier.hpp"
#include <cuda_runtime.h>

class gpu_tier_t : public base_tier_t {
    char* start_ptr_ = nullptr;
    atomic_queue_t flush_q;
    atomic_queue_t fetch_q;
    cudaStream_t flush_stream;
    cudaStream_t fetch_stream;
public:
    gpu_tier_t(int gpu_id, unsigned int num_threads, size_t total_size);
    ~gpu_tier_t() {
        wait_for_completion();
        is_active = false;
        flush_q.set_inactive();
        fetch_q.set_inactive();
    };
    void flush(mem_region_t* m);
    void fetch(mem_region_t* m);
    void flush_io_();
    void fetch_io_();
    void wait_for_completion();
};

#endif // __DATASTATES_GPU_TIER_HPP