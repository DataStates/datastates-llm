#ifndef __DATASTATES_FILE_TIER_HPP
#define __DATASTATES_FILE_TIER_HPP

#include "base_tier.hpp"
#include <cuda_runtime.h>
using namespace datastates;
class file_tier_t : public base_tier_t {
public:
    // This is just a temp placeholder tier for file tier
    // This will be used to manage transfers to/from local to remote file systems
    file_tier_t(int gpu_id, unsigned int num_threads, size_t total_size, int rank=-1);
    ~file_tier_t();
    void flush(std::shared_ptr<mem_region_t> m);
    void fetch(std::shared_ptr<mem_region_t> m);
    void flush_io_();
    void fetch_io_();
    void wait_for_completion();
};

#endif // __DATASTATES_FILE_TIER_HPP