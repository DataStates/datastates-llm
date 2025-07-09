#include "host_tier.hpp"

host_tier_t::host_tier_t(int gpu_id, unsigned int num_threads, size_t total_size): 
    base_tier_t(HOST_PINNED_TIER, gpu_id, num_threads, total_size) {
    assert((num_threads == 1) && "[HOST_TIER] Number of flush and fetch threads should be set to 1.");
    checkCuda(cudaSetDevice(gpu_id_));
    checkCuda(cudaMallocHost(&start_ptr_, total_size));
    mem_pool = new mem_pool_t(start_ptr_, total_size, gpu_id, HOST_PINNED_TIER);
    flush_thread_ = std::thread([&] { flush_io_(); });
    fetch_thread_ = std::thread([&] { fetch_io_(); });
    flush_thread_.detach();
    fetch_thread_.detach();
    DBG("Started flush and fetch threads_ on Host tier for GPU: " << gpu_id);
}

void host_tier_t::flush(mem_region_t *src) {
    assert((successor_tier_ != nullptr) && "[HOST_TIER] Successor tier is not set.");
    assert((src->curr_tier_type == HOST_PINNED_TIER || src->curr_tier_type == HOST_UNPINNED_TIER) && "[HOST_TIER] Source to flush from should be a host memory type.");
    assert((successor_tier_->tier_type_ == FILE_TIER) && "[HOST_TIER] Only flush from host to file supported.");
    flush_q.push(src);
}

void host_tier_t::fetch(mem_region_t *src) {
    assert((successor_tier_ != nullptr) && "[HOST_TIER] Successor tier is not set.");
    assert((successor_tier_->tier_type_ == FILE_TIER) && "[HOST_TIER] Only fetch from file to host supported.");
    fetch_q.push(src);
    fetch_q.wait_for_completion();
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
        size_t curr_size = 0, req_resize = 0;
        std::error_code ec;
        DBG("[HOST_TIER] Flushing from host to file " << src->uid << " internal uid " << src->internal_uid << " at file_offset " << src->file_start_offset << " at " << src->path << " tensor of size " << src->size);
        try {
            std::ofstream f;            
            f.exceptions(std::ofstream::failbit | std::ofstream::badbit);
            f.open(src->path, std::ios::out | std::ios::binary);
            curr_size = std::filesystem::file_size(src->path);
            req_resize = src->file_start_offset + src->size;
            if (req_resize > curr_size) {
                std::filesystem::resize_file(src->path, req_resize, ec);
                curr_size = std::filesystem::file_size(src->path, ec);
            }
            
            f.seekp(src->file_start_offset);
            f.write(const_cast<char*>(src->ptr), src->size);
            f.flush();      // This is for consistency guarantee.
            f.close();
            mem_pool->deallocate(src);
            flush_q.pop();
        } catch (const std::exception& ex) {
            curr_size = std::filesystem::file_size(src->path, ec);
            std::string resize_err = " req resize " + std::to_string(req_resize) + " error code: " 
            + std::to_string(ec.value()) + " error message: " + ec.message();

            FATAL("[HostFlush] Got exception " << "[HOST_TIER] Flushing from host to file region " 
                << src->uid << " internal uid " << src->internal_uid << " at file_offset " << src->file_start_offset << " at " 
                << src->path << " tensor of size " << src->size << " " << " curr file size " << curr_size << " error: " << ex.what() 
                << " resize: " << resize_err);
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
            f.read(const_cast<char*>(src->ptr), src->size);
            f.close();
            fetch_q.pop();
        } catch (const std::exception& ex) {
            FATAL("[HostFetch] Got exception " << ex.what());
        }
    }
}
