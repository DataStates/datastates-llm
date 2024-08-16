#include "host_tier.hpp"

host_tier_t::host_tier_t(int gpu_id, unsigned int num_threads, size_t total_size): 
    base_tier_t(HOST_PINNED_TIER, gpu_id, num_threads, total_size) {
    assert((num_threads == 1) && "[HOST_TIER] Number of flush and fetch threads should be set to 1.");
    checkCuda(cudaSetDevice(gpu_id_));
    checkCuda(cudaMallocHost(&start_ptr_, total_size));
    mem_pool = new mem_pool_t(start_ptr_, total_size, gpu_id);
    flush_thread_ = std::thread([&] { flush_io_(); });
    fetch_thread_ = std::thread([&] { fetch_io_(); });
    flush_thread_.detach();
    fetch_thread_.detach();
    DBG("Started flush and fetch threads_ on Host tier for GPU: " << gpu_id);
}

void host_tier_t::flush(mem_region_t *src) {
    assert((successor_tier_ != nullptr) && "[HOST_TIER] Successor tier is not set.");
    assert((src->curr_tier_type == HOST_PINNED_TIER) && "[HOST_TIER] Source to flush from should be a host memory type.");
    assert((successor_tier_->tier_type == FILE_TIER) && "[HOST_TIER] Only flush from host to file supported.");
    flush_q.push(src);
}

void host_tier_t::fetch(mem_region_t *src) {
    // assert((successor_tier_ != nullptr) && "[HOST_TIER] Successor tier is not set.");
    // assert((src->curr_tier_type == FILE_TIER) && "[HOST_TIER] Only fetch from file to host supported.");
    // assert((successor_tier_->tier_type == FILE_TIER) && "[HOST_TIER] Only fetch from file to host supported.");
    fetch_q.push(src);
}

void host_tier_t::wait_for_completion() {
    DBG("Going to invoke flush_q.wait_for_completeion()");
    flush_q.wait_for_completion();
};

void host_tier_t::flush_io_() {
    checkCuda(cudaSetDevice(gpu_id_));
    while(is_active) {
        bool res = flush_q.wait_for_item();
        if (res == false || is_active == false)
            return;
        mem_region_t* src = flush_q.get_front();
        DBG("[HOST_TIER] Flushing from host to file " << src->uid << " at file_offset " << src->file_start_offset << " at " << src->path << " tensor of size " << src->size);
        try {
            if (!std::filesystem::exists(src->path)) {
                std::ofstream createFile(src->path, std::ios::binary);
                createFile.close();
            }
            std::ofstream f;            
            f.exceptions(std::ofstream::failbit | std::ofstream::badbit);
            f.open(src->path, std::ios::in | std::ios::out | std::ios::binary);
            f.seekp(src->file_start_offset);
            f.write(src->ptr, src->size);
            f.flush();      // This is for consistency guarantee.
            f.close();
            mem_pool->deallocate(src);
            flush_q.pop();
        } catch (const std::exception& ex) {
            FATAL("[HostFlush] Got exception " << ex.what());
        }
    }
}

void host_tier_t::fetch_io_() {
    checkCuda(cudaSetDevice(gpu_id_));
    while(is_active) {
        try {
            bool res = fetch_q.wait_for_item();
            if (res == false || is_active == false)
                return;
            mem_region_t* src = fetch_q.get_front();
            DBG("Starting to fetch in background thread right now " << src->path << " from offset " << src->file_start_offset << " of size " << src->size);
            assert((src->ptr != nullptr) && "[HOST_TIER] Memory not allocated for fetching.");
                    
            std::ifstream f;            
            f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
            f.open(src->path, std::ios::in | std::ios::binary);
            f.seekg(src->file_start_offset);
            f.read(src->ptr, src->size);
            f.close();
            fetch_q.pop();
        } catch (const std::exception& ex) {
            FATAL("[HostFetch] Got exception " << ex.what());
        }
    }
}
