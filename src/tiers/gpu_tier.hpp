#ifndef __DATASTATES_GPU_TIER_HPP
#define __DATASTATES_GPU_TIER_HPP

#include "base_tier.hpp"
#include <cuda_runtime.h>
using namespace datastates;
class gpu_tier_t : public base_tier_t {
    char* start_ptr_ = nullptr;
    cudaStream_t flush_stream;
    cudaStream_t fetch_stream;
    void flush_io_();
    void fetch_io_();
public:
    gpu_tier_t(int gpu_id, unsigned int num_threads, size_t total_size);
    ~gpu_tier_t();
    void flush(std::shared_ptr<mem_region_t> m);
    void fetch(std::shared_ptr<mem_region_t> m);
    void wait_for_completion();
};

#endif // __DATASTATES_GPU_TIER_HPP