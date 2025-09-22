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
#include "state_io_engine.hpp"
#include "state_provider/state_manager.hpp"
#include "datastates.hpp"

namespace datastates {

class state_io_engine_impl_t : public state_io_engine_t {
    core_t* core_engine = nullptr;
    int rank = -1;
    bool use_io_uring = true;
    size_t fs_block_alignment = 4096;
public:
    state_io_engine_impl_t(size_t host_cache_size, int gpu_id, int rank = -1, bool use_io_uring = false, size_t fs_block_alignment = 1);
    void ckpt(std::uint64_t version, state_manager_t* state, std::string path) override;
    void restore(std::uint64_t version, state_manager_t* state, std::string path) override;
    void wait(state_manager_t* state, bool persist=false) override;
    std::string get_queue_stats(bool for_flush_queue=true) override;
    std::string shutdown() override;

    ~state_io_engine_impl_t();
};

} // namespace datastates

#endif // DATASTATES_CORE_IMPL_HPP
