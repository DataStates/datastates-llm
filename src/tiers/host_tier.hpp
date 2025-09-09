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
#include <liburing.h>



using namespace datastates;
class host_tier_t : public base_tier_t {
    char* start_ptr_ = nullptr;
    size_t pwrite_loop_(int fd, const char* ptr, size_t size, size_t file_start_offset);
    size_t flush_io_uring_(int fd, std::shared_ptr<mem_region_t> src);
    void fsync_io_uring_();
    void flush_io_();
    void fetch_io_();
    struct io_uring ring;
    int last_version = -1;
    int get_fd_(std::string path, bool is_odirect);
    std::unordered_map<std::string, int> open_direct_files;
    std::unordered_map<std::string, int> open_nondirect_files;
    std::atomic<size_t> num_submitted = 0;
    std::atomic<size_t> num_completed = 0;
    atomic_queue_t<std::shared_ptr<mem_region_t>> pending_fsync_q;
public:
    host_tier_t(int gpu_id, unsigned int num_threads, size_t total_size);
    ~host_tier_t();
    void flush(std::shared_ptr<mem_region_t> m);
    void fetch(std::shared_ptr<mem_region_t> m);
    void wait_for_completion();
};

#endif // __DATASTATES_HOST_TIER_HPP