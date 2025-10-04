#include "file_tier.hpp"

file_tier_t::file_tier_t(int gpu_id, unsigned int num_threads, size_t total_size, int rank): 
    base_tier_t(FILE_TIER, gpu_id, num_threads, total_size, rank) {
    assert((num_threads == 1) && "[FILE_TIER] Number of flush and fetch threads should be set to 1.");
    checkCuda(cudaSetDevice(gpu_id_));
    DBG("Started flush and fetch threads_ on file tier for GPU: " << gpu_id);
}

file_tier_t::~file_tier_t() {
    // wait_for_completion();
    is_active = false;
    flush_q.set_inactive();
    fetch_q.set_inactive();
}

void file_tier_t::flush(std::shared_ptr<mem_region_t> src) {
    FATAL("[FILE_TIER] Flush operation is not yet supported on file tier.");
    return;
}

void file_tier_t::fetch(std::shared_ptr<mem_region_t> src) {
    FATAL("[FILE_TIER] Fetch operation is not yet supported on file tier.");
    return;
}

void file_tier_t::wait_for_completion() {
    FATAL("[FILE_TIER] Wait for completion is not yet supported on file tier.");
    return;
}

void file_tier_t::flush_io_() {
    return;
}

void file_tier_t::fetch_io_() {
    return;
}
