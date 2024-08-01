#include "host_tier.hpp"

host_tier_t::host_tier_t(int gpu_id, unsigned int num_threads, size_t total_size): 
    base_tier_t(HOST_PINNED_TIER, gpu_id, num_threads, total_size) {
    assert((num_threads == 1) && "[HOST_TIER] Number of flush and fetch threads should be set to 1.");
    checkCuda(cudaSetDevice(gpu_id_));
    mem_pool = new mem_pool_t<rmm::mr::pinned_host_memory_resource>(HOST_PINNED_TIER, total_size, gpu_id);
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
        DBG("[HOST_TIER] Flushing from host to file at file_offset " << src->file_start_offset);
        try {
            if (!std::filesystem::exists(src->path)) {
                try {   
                    std::ofstream createFile(src->path, std::ios::binary);
                    createFile.close();
                } catch (const std::exception& ex) {
                    FATAL("[HostFlush] Got exception in create file " << ex.what() << " while writing to file " << src->path << " at offset " << src->file_start_offset);
                }
            }
            std::ofstream f;            
            f.exceptions(std::ofstream::failbit | std::ofstream::badbit);
            f.open(src->path, std::ios::in | std::ios::out | std::ios::binary);
            if (!f.is_open()) {
                FATAL("Failed to open the file: " << src->path);
                return;
            }
            try {
                f.seekp(src->file_start_offset);
            } catch (const std::exception& ex) {
                FATAL("[HostFlush] Got exception in seekp " << ex.what() << " while writing to file " << src->path << " at offset " << src->file_start_offset << " curr file size " << std::filesystem::file_size(src->path));
            }
            try{
                f.write(src->ptr, src->size);
            } catch (const std::ofstream::failure& e) {
                std::error_code ec(errno, std::generic_category());
                std::cerr << "Exception writing to file: " << e.what() << std::endl;
                std::cerr << "Error code: " << ec.value() << " - " << ec.message() << std::endl;
            } catch (const std::system_error& e) {
                std::cerr << "System error: " << e.what() << std::endl;
                std::cerr << "Error code: " << e.code() << " - " << e.code().message() << std::endl;
            } catch (const std::exception& ex) {
                FATAL("[HostFlush] Got exception in write " << ex.what() << " while writing to file " << src->path << " at offset " << src->file_start_offset);
            }
            f.flush();      // This is for consistency guarantee.
            f.close();
            mem_pool->deallocate(src);
            flush_q.pop();
        } catch (const std::exception& ex) {
            FATAL("[HostFlush] Got exception " << ex.what() << " while writing to file " << src->path << " at offset " << src->file_start_offset);
        }
    }
}

void host_tier_t::fetch_io_() {
    checkCuda(cudaSetDevice(gpu_id_));
    while(is_active) {
        bool res = fetch_q.wait_for_item();
        if (res == false || is_active == false)
            return;
        mem_region_t* src = fetch_q.get_front();
        DBG("Starting to fetch in background thread right now....");
        assert((src->ptr != nullptr) && "[HOST_TIER] Memory not allocated for fetching.");
                
        std::ifstream f;            
        f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        f.open(src->path, std::ios::in | std::ios::binary);
        f.seekg(src->file_start_offset);
        f.read(src->ptr, src->size);
        f.close();
        fetch_q.pop();
    }
}


void host_tier_t::tier_allocate(mem_region_t* region) {
    mem_pool->allocate(region);
}

void host_tier_t::tier_deallocate(mem_region_t* region) {
    mem_pool->deallocate(region);
}
