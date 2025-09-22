#ifndef __DATASTATES_HPP
#define __DATASTATES_HPP


#include <iostream>
#include <cstdint>
#include <string>
#include <memory>
#include "common/mem_region.hpp"
#include "common/utils.hpp"

namespace datastates {
class core_t {
public:
    virtual void ckpt(std::uint64_t version, std::uint64_t uid, const char* ptr, std::uint64_t size, std::uint64_t offset, std::string path) = 0;
    virtual void ckpt_region(std::shared_ptr<mem_region_t> m) = 0;
    virtual void restore(std::uint64_t version, std::uint64_t uid, const char* ptr, std::uint64_t size, std::uint64_t offset, std::string path) = 0;
    virtual void wait(bool persist=false) = 0;
    virtual std::string shutdown() = 0;
    virtual std::string get_queue_stats(bool for_flush_queue=true) = 0;
    virtual ~core_t() = default;
};

core_t* dstates_engine(size_t host_cache_size, int gpu_id, int rank = -1, bool use_io_uring = false, size_t fs_block_alignment = FS_BLOCK_SIZE_ALIGNMENT);
} // namespace datastates

#endif // __DATASTATES_HPP