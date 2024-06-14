#ifndef __DATASTATES_HPP
#define __DATASTATES_HPP


#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include "tiers/host_tier.hpp"
#include "tiers/gpu_tier.hpp"

namespace py = pybind11;

static volatile uint64_t local_uid = 1;
class datastates_llm_t {
    host_tier_t* host_tier;
    gpu_tier_t* gpu_tier;
    bool is_active = true;
    int gpu_id = 0;
    int rank = -1;
    
    public:
    datastates_llm_t(size_t host_cache_size, int gpu_id, int rank=-1);
    void ckpt_tensor(int version, const torch::Tensor &t, const std::uint64_t size, const std::uint64_t file_offset, std::string path);
    void restore_tensor(int version, const torch::Tensor &t, const std::uint64_t size, const std::uint64_t file_offset, std::string path);
    void wait();
    void shutdown();
};

#endif // __DATASTATES_HPP