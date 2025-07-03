#ifndef __STATE_IO_ENGINE_HPP
#define __STATE_IO_ENGINE_HPP


#include <iostream>
#include <cstdint>
#include <string>
#include "state_manager.hpp"

namespace datastates {
class state_io_engine_t {
public:
    virtual void ckpt(uint version, state_manager_t* state, std::string path) = 0;
    virtual void restore(uint version, state_manager_t* state, std::string path) = 0;
    virtual void wait(state_manager_t* state, bool persist=false) = 0;
    virtual void shutdown() = 0;
    virtual ~state_io_engine_t() = default;
};

state_io_engine_t* create_io_engine(size_t host_cache_size, int gpu_id, int rank = -1);
} // namespace datastates

#endif // __STATE_IO_ENGINE_HPP