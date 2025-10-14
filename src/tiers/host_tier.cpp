#include "host_tier.hpp"

host_tier_t::host_tier_t(int gpu_id, unsigned int num_threads, size_t total_size, int rank, bool use_io_uring): 
    base_tier_t(HOST_PINNED_TIER, gpu_id, num_threads, total_size, rank) {
    assert((num_threads == 1) && "[HOST_TIER] Number of flush and fetch threads should be set to 1.");
    checkCuda(cudaSetDevice(gpu_id_));
    // checkCuda(cudaMallocHost(&start_ptr_, total_size));
    if (get_fs_block_alignment() <= 1) {
        start_ptr_ = (static_cast<char*>(malloc(total_size)));
    } else {
        int ret = posix_memalign(reinterpret_cast<void**>(&start_ptr_), get_fs_block_alignment(), total_size);
        if (ret != 0) {
            FATAL("posix_memalign failed with error code " + std::to_string(ret));
        }
    }
    checkCuda(cudaHostRegister(start_ptr_, total_size, cudaHostRegisterDefault));
    mem_pool = std::make_shared<mem_pool_t>(start_ptr_, total_size, gpu_id, rank, HOST_PINNED_TIER);

    flush_thread_ = std::thread([&] { flush_io_(); });
    fetch_thread_ = std::thread([&] { fetch_io_(); });
    if (use_io_uring) {
        file_handler = std::make_shared<io_uring_handler_t>(mem_pool);
    } else {
        file_handler = std::make_shared<pwrite_handler_t>(mem_pool);
    }
    DBG("Started flush and fetch threads_ on Host tier for GPU: " << gpu_id);
}

host_tier_t::~host_tier_t() {
    wait_for_completion();
    fetch_q.wait_for_completion();
    is_active = false;
    flush_q.set_inactive();
    fetch_q.set_inactive();
    flush_thread_.join();
    fetch_thread_.join();
    checkCuda(cudaHostUnregister(start_ptr_));
    free(start_ptr_);
}

void host_tier_t::flush(std::shared_ptr<mem_region_t> src) {
    assert((successor_tier_ != nullptr) && "[HOST_TIER] Successor tier is not set.");
    assert((src->curr_tier_type == HOST_PINNED_TIER || src->curr_tier_type == HOST_UNPINNED_TIER) && "[HOST_TIER] Source to flush from should be a host memory type.");
    assert((successor_tier_->tier_type_ == FILE_TIER) && "[HOST_TIER] Only flush from host to file supported.");
    perf_profiler.record_event(src, HOST_WAIT_START);
    flush_q.push(src);
}

void host_tier_t::fetch(std::shared_ptr<mem_region_t> src) {
    assert((successor_tier_ != nullptr) && "[HOST_TIER] Successor tier is not set.");
    assert((successor_tier_->tier_type_ == FILE_TIER) && "[HOST_TIER] Only fetch from file to host supported.");
    fetch_q.push(src);
}

void host_tier_t::wait_for_completion() {
    DBG("Going to invoke flush_q.wait_for_completion()");
    flush_q.wait_for_completion();
    fetch_q.wait_for_completion();
    file_handler->fsync();
}

void host_tier_t::flush_io_() {
    try {
        checkCuda(cudaSetDevice(gpu_id_));
        while (is_active) {
            bool res = flush_q.wait_for_item();
            if (!res)
                break;
            auto src = flush_q.get_front();
            perf_profiler.record_event(src, HOST_WAIT_END);
            perf_profiler.record_event(src, HOST_START);
            bool is_odirect = false;
            if (src->aligned_size > 0 &&
                src->aligned_size % get_fs_block_alignment() == 0 &&
                src->size >= get_fs_block_alignment() &&
                get_fs_block_alignment() > 1) {
                if (!is_aligned(reinterpret_cast<uintptr_t>(src->ptr))) {
                    FATAL("[HOST_TIER][io_uring] Pointer not aligned to fs block size");
                }
                if (!is_aligned(src->file_start_offset)) {
                    FATAL("[HOST_TIER][io_uring] Offset not aligned to fs block size");
                }
                is_odirect = true;
            }
            file_handler->write(src, is_odirect);
            flush_q.pop();
        }

    } catch (const std::exception& ex) {
        FATAL("[HostFlush] Got exception " << ex.what());
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
            assert((src->ptr != nullptr) && "[HOST_TIER] Memory not allocated for fetching.");
            bool is_odirect = false;
            if (src->aligned_size > 0 &&
                src->aligned_size % get_fs_block_alignment() == 0 &&
                src->size >= get_fs_block_alignment() &&
                get_fs_block_alignment() > 1) {
                if (!is_aligned(reinterpret_cast<uintptr_t>(src->ptr))) {
                    FATAL("[HOST_TIER][io_uring] Pointer not aligned to fs block size");
                }
                if (!is_aligned(src->file_start_offset)) {
                    FATAL("[HOST_TIER][io_uring] Offset not aligned to fs block size");
                }
                is_odirect = true;
            }
            file_handler->read(src, is_odirect);
            fetch_q.pop();
        } catch (const std::exception& ex) {
            FATAL("[HostFetch] Got exception " << ex.what());
        }
    }
}
