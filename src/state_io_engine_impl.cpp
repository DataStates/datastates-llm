#include "state_io_engine_impl.hpp"


namespace datastates {

state_io_engine_impl_t::state_io_engine_impl_t(size_t host_cache_size, int gpu_id, int rank) {
    try {
        core_engine = dstates_engine(host_cache_size, gpu_id, rank);
    } catch(std::exception& e) {
        FATAL("Standard exception caught in datastates init: " << e.what());
    }
}

void state_io_engine_impl_t::ckpt(uint version, state_manager_t* state, std::string path) {
    try {
        DBG("Going to checkpoint state...");
        for (TIER_TYPES tier : {GPU_TIER, HOST_PINNED_TIER, HOST_UNPINNED_TIER}) {
            while (state->has_next_chunk(tier)) {
                mem_region_t* m = new mem_region_t(version, 0 /*region_id*/, nullptr /*ptr*/, 0 /*size*/ , 0 /*file_offset*/, path, tier);
                state->get_next_chunk(tier, m);
                DBG("Going to checkpoint memory region with UID " << m->uid << " of size " << m->size << " at file offset " << m->file_start_offset);
                core_engine->ckpt_region(m);
            }
        }
    } catch (std::exception &e) {
        FATAL("Exception caught in ckpt." << e.what());
    }
    return;
}

void state_io_engine_impl_t::restore(uint version, state_manager_t* state, std::string path) {
    try {
        FATAL("Restoring state is not yet implemented.");
    } catch (std::exception &e) {
        FATAL("Exception caught in restore." << e.what());
    }
}

void state_io_engine_impl_t::wait(state_manager_t* state, bool persist) {
    try {
        core_engine->wait(persist);
        state->release();
    }  catch (std::exception &e) {
        FATAL("Exception caught in wait D2H." << e.what());
    }
}

void state_io_engine_impl_t::shutdown() {
    try {
        DBG("Shutting down state I/O engine.");
        core_engine->wait(true);
        core_engine->shutdown();
        DBG("Deleting core engine.");
        return;
    } catch (std::exception &e) {
        FATAL("Exception caught in shutdown." << e.what());
    }
}

state_io_engine_impl_t::~state_io_engine_impl_t() {
    try {
        shutdown();
    } catch (std::exception &e) {
        FATAL("Exception caught in destructor." << e.what());
    }
}

} // namespace datastates