#include "core_impl.hpp"


namespace datastates {

core_impl_t::core_impl_t(size_t host_cache_size, int gpu_id_, int rank_, bool use_io_uring_, size_t fs_block_alignment_): 
    gpu_id(gpu_id_), rank(rank_), use_io_uring(use_io_uring_), fs_block_alignment(fs_block_alignment_) {
    try {
        DBG("DataStates initing: GPU: " << gpu_id << ", host cache (MB): " << (host_cache_size >> 20));
        checkCuda(cudaSetDevice(gpu_id));
        is_active = true;
        int num_threads = 1;    // For initial prototype, set number of threads=1
        size_t gpu_cache = 1;   // For initial prototype, assume no GPU memory available for checkpointing.
        host_tier = std::make_shared<host_tier_t>(gpu_id, num_threads, host_cache_size, rank, use_io_uring);
        gpu_tier = std::make_shared<gpu_tier_t>(gpu_id, num_threads, gpu_cache, rank);
        file_tier = std::make_shared<file_tier_t>(gpu_id, num_threads, 0, rank);
        gpu_tier->set_successor_tier(host_tier);
        host_tier->set_successor_tier(file_tier);
        
    } catch(std::exception& e) {
        FATAL("Standard exception caught in datastates init: " << e.what());
    }
}

void core_impl_t::ckpt(std::uint64_t version, std::uint64_t uid, const char* ptr, const std::uint64_t size, const std::uint64_t file_offset, std::string path) {
    try {
        DBG("Going to checkpoint tensor of UID " << uid << " and size " << size << " at offset " << file_offset);
        cudaPointerAttributes attr;
        checkCuda(cudaPointerGetAttributes(&attr, ptr));
        if (attr.type == GPU_TIER) {
            assert((attr.device == gpu_id) && "Pointer not on the same GPU as ckpt engine");
            auto m = std::make_shared<mem_region_t>(version, uid, const_cast<char*>(ptr), size, file_offset, path, GPU_TIER);
            ckpt_region(m);
            return;
        } else if (attr.type == HOST_PINNED_TIER || attr.type == HOST_UNPINNED_TIER) {
            auto m = std::make_shared<mem_region_t>(version, uid, const_cast<char*>(ptr), size, file_offset, path, HOST_PINNED_TIER);
            ckpt_region(m);
            return;
        } else {
            FATAL("Checkpointing is not supported on tiers other than GPU, Host unpinned, or Host pinned.");
        }
    } catch (std::exception &e) {
        FATAL("Exception caught in ckpt." << e.what());
    }
}

void core_impl_t::ckpt_region(std::shared_ptr<mem_region_t> m) {
    try {
        DBG("Going to checkpoint memory region with UID " << m->uid << " of size " << m->size << " at file offset " << m->file_start_offset);
        if (m->curr_tier_type == GPU_TIER) {
            assert((m->ptr != nullptr) && "Pointer cannot be null for GPU tier");
            assert((m->size > 0) && "Size must be greater than zero for GPU tier");
            gpu_tier->flush(m);
        } else if (m->curr_tier_type == HOST_PINNED_TIER || m->curr_tier_type == HOST_UNPINNED_TIER) {
            assert((m->ptr != nullptr) && "Pointer cannot be null for Host tier");
            assert((m->size > 0) && "Size must be greater than zero for Host tier");
            host_tier->flush(m);
        } else {
            FATAL("Checkpointing is not supported on tiers other than GPU, Host unpinned, or Host pinned.");
        }
    } catch (std::exception &e) {
        FATAL("Exception caught in ckpt_region." << e.what());
    }
}

void core_impl_t::restore(std::uint64_t version, std::uint64_t uid, const char* ptr, const std::uint64_t size, const std::uint64_t file_offset, std::string path) {
    try {
        cudaPointerAttributes attr;
        checkCuda(cudaPointerGetAttributes(&attr, ptr));
        if (attr.type == GPU_TIER) {
            FATAL("Restoring to GPU memory is not yet supported. Please restore to host memory first.");
        }
        DBG("Going to restore from " << path << " tensor of size " << size << " at file offset " << file_offset);
        auto m = std::make_shared<mem_region_t>(version, uid, const_cast<char*>(ptr), size, file_offset, path, HOST_PINNED_TIER);
        host_tier->fetch(m);
        return;
    } catch (std::exception &e) {
        FATAL("Exception caught in restore." << e.what());
    }
}

void core_impl_t::wait(bool persist) {
    try {
        gpu_tier->wait_for_completion();
        if (persist)
            host_tier->wait_for_completion();
    }  catch (std::exception &e) {
        FATAL("Exception caught in wait D2H." << e.what());
    }
}

std::string core_impl_t::get_queue_stats(bool for_flush_queue) {
    try {
        std::string stats = "Device: " + std::to_string(gpu_id) + ", GPU: " + std::to_string(gpu_tier->get_queue_size(for_flush_queue)) + ", Host: " + std::to_string(host_tier->get_queue_size(for_flush_queue));
        return stats;
    } catch (std::exception &e) {
        FATAL("Exception caught in get_queue_stats." << e.what());
    }
}

std::string core_impl_t::shutdown() {
    try {
        if (is_active) {
            wait(true);
            gpu_tier.reset();
            host_tier.reset();
            is_active = false;
            perf_profiler_t& perf_profiler = perf_profiler_t::get_instance();
            return perf_profiler.report();
        }
        return "";
    } catch (std::exception &e) {
        FATAL("Exception caught in shutdown." << e.what());
    }
}

core_impl_t::~core_impl_t() {
    try {
        shutdown();
    } catch (std::exception &e) {
        FATAL("Exception caught in destructor." << e.what());
    }
}

} // namespace datastates