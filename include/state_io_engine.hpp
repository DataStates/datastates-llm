#ifndef __STATE_IO_ENGINE_HPP
#define __STATE_IO_ENGINE_HPP


#include <iostream>
#include <cstdint>
#include <string>
#include "state_manager.hpp"

namespace datastates {
class state_io_engine_t {
public:
    virtual void ckpt(std::uint64_t version, state_manager_t* state, std::string path) = 0;
    virtual std::string restore(std::uint64_t version, std::string path) = 0;
    virtual void wait(state_manager_t* state, bool persist=false) = 0;
    virtual std::string shutdown() = 0;
    virtual std::string get_queue_stats(bool for_flush_queue=true) = 0;
    virtual ~state_io_engine_t() = default;
};
static std::shared_ptr<state_io_engine_t> state_io_engine_instance = nullptr;
static std::atomic<bool> is_state_io_engine_active {true};
state_io_engine_t* create_state_io_engine(size_t host_cache_size, int gpu_id, int rank = -1, bool use_io_uring = false, size_t fs_block_alignment = FS_BLOCK_SIZE_ALIGNMENT);
} // namespace datastates

#endif // __STATE_IO_ENGINE_HPP