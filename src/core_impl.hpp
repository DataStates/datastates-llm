#ifndef DATASTATES_CORE_IMPL_HPP
#define DATASTATES_CORE_IMPL_HPP

#include <cstdint>
#include <string>
#include "datastates.hpp"
#include "tiers/host_tier.hpp"
#include "tiers/gpu_tier.hpp"
#include "tiers/file_tier.hpp"
#include "common/mem_region.hpp"
#include "common/defs.hpp"
#include "common/utils.hpp"

namespace datastates {

class core_impl_t : public core_t {
    host_tier_t* host_tier;
    gpu_tier_t* gpu_tier;
    file_tier_t* file_tier;
    bool is_active = true;
    int gpu_id = 0;
    int rank = -1;
public:
    core_impl_t(size_t host_cache_size, int gpu_id, int rank = -1);
    void ckpt(uint version, uint region_id, const char* ptr, std::uint64_t size, std::uint64_t offset, std::string path) override;
    void ckpt_region(std::shared_ptr<mem_region_t> m) override;
    void restore(uint version, uint region_id, const char* ptr, std::uint64_t size, std::uint64_t offset, std::string path) override;
    void wait(bool persist=false) override;
    std::string shutdown() override;
    ~core_impl_t();
};

} // namespace datastates

#endif // DATASTATES_CORE_IMPL_HPP
