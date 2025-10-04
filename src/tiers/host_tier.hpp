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
#include <stdexcept>
#include <set>
#include "file_handlers/base_file_handler.hpp"
#include "file_handlers/io_uring_handler.hpp"
#include "file_handlers/pwrite_handler.hpp"

using namespace datastates;
class host_tier_t : public base_tier_t {
    char* start_ptr_ = nullptr;
    void flush_io_();
    void fetch_io_();
    std::shared_ptr<base_file_handler_t> file_handler = nullptr;
public:
    host_tier_t(int gpu_id, unsigned int num_threads, size_t total_size, int rank=-1, bool use_io_uring=false);
    ~host_tier_t();
    void flush(std::shared_ptr<mem_region_t> m);
    void fetch(std::shared_ptr<mem_region_t> m);
    void wait_for_completion();
};

#endif // __DATASTATES_HOST_TIER_HPP