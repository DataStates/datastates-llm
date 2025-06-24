#ifndef __DATASTATES_CORE_HPP
#define __DATASTATES_CORE_HPP


#include <torch/torch.h>
#include "tiers/host_tier.hpp"
#include "tiers/gpu_tier.hpp"
#include "tiers/file_tier.hpp"
#include "nanobind/nanobind.h"

static volatile uint64_t local_uid = 1;
namespace nb = nanobind;
class datastates_core_t {
    host_tier_t* host_tier;
    gpu_tier_t* gpu_tier;
    file_tier_t* file_tier;
    bool is_active = true;
    int gpu_id = 0;
    int rank = -1;
    
    public:
    datastates_core_t(size_t host_cache_size, int gpu_id, int rank=-1);
    void ckpt(int version, const char* ptr, const std::uint64_t size, const std::uint64_t file_offset, std::string path);
    void restore(int version, const char* ptr, const std::uint64_t size, const std::uint64_t file_offset, std::string path);
    void wait();
    void shutdown();
};

#endif // __DATASTATES_CORE_HPP