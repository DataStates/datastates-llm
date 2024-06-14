#ifndef __DATASTATES_HOST_TIER_HPP
#define __DATASTATES_HOST_TIER_HPP

#include "base_tier.hpp"
#include <fstream>
#include <filesystem>

class host_tier_t : public base_tier_t {
    char* start_ptr_ = nullptr;
public:
    host_tier_t(int gpu_id, unsigned int num_threads, size_t total_size);
    ~host_tier_t() {
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

#endif // __DATASTATES_HOST_TIER_HPP