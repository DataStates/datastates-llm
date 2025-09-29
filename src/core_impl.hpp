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
    std::shared_ptr<host_tier_t> host_tier;
    std::shared_ptr<gpu_tier_t> gpu_tier;
    std::shared_ptr<file_tier_t> file_tier;
    int gpu_id = 0;
    int rank = -1;
    bool use_io_uring = false;
    size_t fs_block_alignment = FS_BLOCK_SIZE_ALIGNMENT;
public:
    core_impl_t(size_t host_cache_size, int gpu_id, int rank = -1, bool use_io_uring = false, size_t fs_block_alignment = FS_BLOCK_SIZE_ALIGNMENT);
    void ckpt(std::uint64_t version, std::uint64_t region_id, const char* ptr, std::uint64_t size, std::uint64_t offset, std::string path) override;
    void ckpt_region(std::shared_ptr<mem_region_t> m) override;
    void restore(std::uint64_t version, std::uint64_t region_id, const char* ptr, std::uint64_t size, std::uint64_t offset, std::string path) override;
    void restore_region(std::shared_ptr<mem_region_t> m) override;
    void wait(bool persist=false) override;
    std::string get_queue_stats(bool for_flush_queue=true) override;
    std::string shutdown() override;
    ~core_impl_t();
};

} // namespace datastates

#endif // DATASTATES_CORE_IMPL_HPP
