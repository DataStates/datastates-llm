#include "host_tier.hpp"

host_tier_t::host_tier_t(int gpu_id, unsigned int num_threads, size_t total_size): 
    base_tier_t(HOST_PINNED_TIER, gpu_id, num_threads, total_size) {
    assert((num_threads == 1) && "[HOST_TIER] Number of flush and fetch threads should be set to 1.");
    checkCuda(cudaSetDevice(gpu_id_));
    // checkCuda(cudaMallocHost(&start_ptr_, total_size));
    int ret = posix_memalign(reinterpret_cast<void**>(&start_ptr_), get_fs_block_alignment(), total_size);
    if (ret != 0) {
        FATAL("posix_memalign failed with error code " + std::to_string(ret));
    }
    checkCuda(cudaHostRegister(start_ptr_, total_size, cudaHostRegisterDefault));
    mem_pool = new mem_pool_t(start_ptr_, total_size, gpu_id, HOST_PINNED_TIER);
    flush_thread_ = std::thread([&] { flush_io_(); });
    fetch_thread_ = std::thread([&] { fetch_io_(); });
    DBG("Started flush and fetch threads_ on Host tier for GPU: " << gpu_id);
}

host_tier_t::~host_tier_t() {
    flush_q.wait_for_completion();
    fetch_q.wait_for_completion();
    checkCuda(cudaHostUnregister(start_ptr_));
    free(start_ptr_);
    is_active = false;
    flush_q.set_inactive();
    fetch_q.set_inactive();
    flush_thread_.join();
    fetch_thread_.join();
}

void host_tier_t::flush(std::shared_ptr<mem_region_t> src) {
    assert((successor_tier_ != nullptr) && "[HOST_TIER] Successor tier is not set.");
    assert((src->curr_tier_type == HOST_PINNED_TIER || src->curr_tier_type == HOST_UNPINNED_TIER) && "[HOST_TIER] Source to flush from should be a host memory type.");
    assert((successor_tier_->tier_type_ == FILE_TIER) && "[HOST_TIER] Only flush from host to file supported.");
    flush_q.push(src);
    perf_profiler.record_event(src, HOST_WAIT_START);
}

void host_tier_t::fetch(std::shared_ptr<mem_region_t> src) {
    assert((successor_tier_ != nullptr) && "[HOST_TIER] Successor tier is not set.");
    assert((successor_tier_->tier_type_ == FILE_TIER) && "[HOST_TIER] Only fetch from file to host supported.");
    fetch_q.push(src);
    fetch_q.wait_for_completion();
}

void host_tier_t::wait_for_completion() {
    DBG("Going to invoke flush_q.wait_for_completion()");
    flush_q.wait_for_completion();
};

void host_tier_t::flush_io_() {
    checkCuda(cudaSetDevice(gpu_id_));
    while(is_active) {
        bool res = flush_q.wait_for_item();
        if (res == false || is_active == false)
            return;
        auto src = flush_q.get_front();
        perf_profiler.record_event(src, HOST_WAIT_END);
        perf_profiler.record_event(src, HOST_START);
        int fd = open(src->path.c_str(), O_WRONLY | O_CREAT, 0644);
        if(src->aligned_size > 0 && get_fs_block_alignment() > 1) { // FS_BLOCK_SIZE_ALIGNMENT==1 means no alignment
            if (!is_aligned(reinterpret_cast<uintptr_t>(src->ptr))) {
                FATAL("[HOST_TIER] Pointer to flush should be aligned to get_fs_block_alignment() " 
                    + std::to_string(reinterpret_cast<uintptr_t>(src->ptr)) 
                    + " is not aligned to " + std::to_string(get_fs_block_alignment())
                    + " for file " + src->path
                    + " of size " + std::to_string(src->size) + " aligned size " + std::to_string(src->aligned_size));
            }
            if (!is_aligned(src->file_start_offset)) {
                FATAL("[HOST_TIER] File start offset to flush should be aligned to get_fs_block_alignment() " 
                    + std::to_string(src->file_start_offset) 
                    + " is not aligned to " + std::to_string(get_fs_block_alignment())
                    + " for file " + src->path
                    + " of size " + std::to_string(src->size) + " aligned size " + std::to_string(src->aligned_size));
            }
            fd = open(src->path.c_str(), O_WRONLY | O_CREAT | O_DIRECT, 0644);
        }

        if (fd < 0) {
            FATAL("[HostFlush] Failed to open file: " + src->path + " Error: " + strerror(errno));
        }
        size_t file_size = src->aligned_size > 0 ? src->aligned_size : src->size;
        ssize_t written = pwrite_loop_(fd, src->ptr, file_size, src->file_start_offset);
        if (written < 0 || static_cast<size_t>(written) < file_size) {
            FATAL("[HostFlush] Incomplete or failed write: written "  + std::to_string(written) + " instead of " + std::to_string(file_size) + " error: " + std::string(strerror(errno)));
        }
        //// Optional: fsync() to ensure consistency
        if (fsync(fd) != 0) {
            close(fd);
            FATAL("[pwrite] fsync failed: " + std::string(strerror(errno)));
        }
        close(fd);
        mem_pool->deallocate(src);
        perf_profiler.record_event(src, HOST_END);
        flush_q.pop();
    }
}

void host_tier_t::fetch_io_() {
    checkCuda(cudaSetDevice(gpu_id_));
    while(is_active) {
        try {
            bool res = fetch_q.wait_for_item();
            if (res == false || is_active == false)
                return;
            auto src = fetch_q.get_front();
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

// Some filesystems do not allow writing more than 2GB (e.g. on ALCF Polaris), so we need this loop
size_t host_tier_t::pwrite_loop_(int fd, const char* ptr, size_t size, size_t file_start_offset) {
    size_t total_written = 0;
    while (total_written < size) {
        size_t to_write = std::min(size - total_written, static_cast<size_t>(MAX_FILE_WRITE_SIZE));
        ssize_t written = pwrite(fd, ptr + total_written, to_write, file_start_offset + total_written);
        if (written < 0 || static_cast<size_t>(written) < to_write) {
            FATAL("[HostFlush] Incomplete or failed write: written "  + std::to_string(total_written) + " instead of " + std::to_string(size) + " error: " + std::string(strerror(errno)));
        }
        total_written += written;
    }
    return total_written;
}