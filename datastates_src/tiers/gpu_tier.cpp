#include "gpu_tier.hpp"

gpu_tier_t::gpu_tier_t(int gpu_id, unsigned int num_threads, size_t total_size): 
    base_tier_t(GPU_TIER, gpu_id, num_threads, total_size) {
    assert((num_threads == 1) && "[GPU_TIER] Number of flush and fetch threads should be set to 1.");
    checkCuda(cudaSetDevice(gpu_id_));
    checkCuda(cudaMalloc(&start_ptr_, total_size));
    mem_pool = new mem_pool_t(start_ptr_, total_size, gpu_id);
    flush_thread_ = std::thread([&] { flush_io_(); });
    fetch_thread_ = std::thread([&] { fetch_io_(); });
    flush_thread_.detach();
    fetch_thread_.detach();
    checkCuda(cudaStreamCreateWithFlags(&flush_stream, cudaStreamNonBlocking));
    checkCuda(cudaStreamCreateWithFlags(&fetch_stream, cudaStreamNonBlocking));
    DBG("Started flush and fetch threads_ on GPU tier for GPU: " << gpu_id);
}

void gpu_tier_t::flush(mem_region_t *m) {
    assert((successor_tier_ != nullptr) && "[GPU_TIER] Successor tier is not set.");
    assert((m->curr_tier_type == GPU_TIER) && "[GPU_TIER] Source to flush from should be a gpu memory type.");
    assert((successor_tier_->tier_type == HOST_PINNED_TIER) && "[GPU_TIER] Only flush from gpu to pinned host memory is supported.");
    flush_q.push(m);
}

void gpu_tier_t::fetch(mem_region_t *m) {
    assert((successor_tier_ != nullptr) && "[GPU_TIER] Successor tier is not set.");
    assert((m->curr_tier_type == HOST_PINNED_TIER) && "[GPU_TIER] Only fetch from pinned host memory to gpu supported.");
    assert((successor_tier_->tier_type == HOST_PINNED_TIER) && "[GPU_TIER] Only fetch from pinned host memory to gpu supported.");
    fetch_q.push(m);
}

void gpu_tier_t::wait_for_completion() {
    DBG("Going to invoke flush_q.wait_for_completeion()");
    flush_q.wait_for_completion();
};

void gpu_tier_t::flush_io_() {
    checkCuda(cudaSetDevice(gpu_id_));
    while(is_active) {
        bool res = flush_q.wait_for_item();
        if (res == false || is_active == false)
            return;
        mem_region_t* src = flush_q.get_front();
        DBG("In GPU tier got src...." << successor_tier_->tier_type_ );
        mem_region_t* dest = new mem_region_t(src, successor_tier_->tier_type_);
        DBG("In GPU tier got dest....");

        successor_tier_->mem_pool->allocate(dest);
        checkCuda(cudaMemcpyAsync(dest->ptr, src->ptr, src->size, cudaMemcpyDeviceToHost, flush_stream));
        checkCuda(cudaStreamSynchronize(flush_stream));
        DBG("[GPU_TIER] Flushed from GPU to host.");
        successor_tier_->flush(dest);
        mem_pool->deallocate(src);
        flush_q.pop();
    }
}

void gpu_tier_t::fetch_io_() {
    checkCuda(cudaSetDevice(gpu_id_));
    while(is_active) {
        bool res = fetch_q.wait_for_item();
        if (res == false || is_active == false)
            return;
        mem_region_t* src = fetch_q.get_front();
        mem_region_t* dest = new mem_region_t(src, tier_type_);
        
        if (mem_pool->get_capacity()) {
            mem_pool->allocate(dest);
            checkCuda(cudaMemcpyAsync(dest->ptr, src->ptr, src->size, cudaMemcpyHostToDevice, fetch_stream));
            checkCuda(cudaStreamSynchronize(fetch_stream));
        }
        fetch_q.pop();
    }
}
