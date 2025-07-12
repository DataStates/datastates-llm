#ifndef __DATASTATES_HOST_TIER_HPP
#define __DATASTATES_HOST_TIER_HPP

#include "base_tier.hpp"
#include <fstream>
#include <filesystem>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstring>
#include <cerrno>
#include <stdexcept>
#include <numeric>

class host_tier_t : public base_tier_t {
    char* start_ptr_ = nullptr;
    size_t pwrite_loop_(int fd, const char* ptr, size_t size, size_t file_start_offset);
    void flush_io_();
    void fetch_io_();
public:
    host_tier_t(int gpu_id, unsigned int num_threads, size_t total_size);
    ~host_tier_t();
    void flush(std::shared_ptr<mem_region_t> m);
    void fetch(std::shared_ptr<mem_region_t> m);
    void wait_for_completion();
};

#endif // __DATASTATES_HOST_TIER_HPP