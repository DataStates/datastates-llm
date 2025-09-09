#include "host_tier.hpp"

host_tier_t::host_tier_t(int gpu_id, unsigned int num_threads, size_t total_size): 
    base_tier_t(HOST_PINNED_TIER, gpu_id, num_threads, total_size) {
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
    mem_pool = new mem_pool_t(start_ptr_, total_size, gpu_id, HOST_PINNED_TIER);
    flush_thread_ = std::thread([&] { flush_io_(); });
    fetch_thread_ = std::thread([&] { fetch_io_(); });
    DBG("Started flush and fetch threads_ on Host tier for GPU: " << gpu_id);
}

host_tier_t::~host_tier_t() {
    flush_q.wait_for_completion();
    fetch_q.wait_for_completion();
    is_active = false;
    flush_q.set_inactive();
    fetch_q.set_inactive();
    flush_thread_.join();
    fetch_thread_.join();
    if (USE_URING) {
        fsync_io_uring_();
        io_uring_queue_exit(&ring);
    }
    checkCuda(cudaHostUnregister(start_ptr_));
    free(start_ptr_);
    delete mem_pool;
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
}

int host_tier_t::get_fd_(std::string path, bool is_odirect) {
    if (is_odirect) {
        if (open_direct_files.find(path) != open_direct_files.end()) {
            return open_direct_files[path];
        } else {
            int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_DIRECT, 0644);
            if (fd < 0) {
                FATAL("[HostFlush][io_uring] Failed to open file: " + path +
                    " Error: " + strerror(errno));
            }
            open_direct_files[path] = fd;
            return fd;
        }
    } else {
        if (open_nondirect_files.find(path) != open_nondirect_files.end()) {
            return open_nondirect_files[path];
        } else {
            int fd = ::open(path.c_str(), O_WRONLY | O_CREAT, 0644);
            if (fd < 0) {
                FATAL("[HostFlush][io_uring] Failed to open file: " + path +
                    " Error: " + strerror(errno));
            }
            open_nondirect_files[path] = fd;
            return fd;
        }
    }
}

void host_tier_t::flush_io_() {
    checkCuda(cudaSetDevice(gpu_id_));
    // Create one io_uring instance per process
    if (USE_URING) {
        std::cout << "[HOST_TIER][io_uring] Initializing io_uring instance." << std::endl;
        if (get_fs_block_alignment() <= 1) {
            FATAL("[HOST_TIER][io_uring] Filesystem block size alignment must be greater than 1 to use io_uring.");
        }
        if (io_uring_queue_init(512, &ring, 0) < 0) {
            FATAL("[io_uring] Failed to initialize queue");
        }
    } else {
        std::cout << "[HOST_TIER][io_uring] Not using io_uring for async I/O operations." << std::endl;
    }

    while (is_active) {
        bool res = flush_q.wait_for_item();
        if (!res)
            break;

        auto src = flush_q.get_front();
        perf_profiler.record_event(src, HOST_WAIT_END);
        perf_profiler.record_event(src, HOST_START);

        if (USE_URING) {
            bool is_odirect = false;
            bool should_fsync = false;
            should_fsync = ( last_version != -1 && last_version != src->version );
            last_version = src->version;
            if (should_fsync) {
                fsync_io_uring_();
            }

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
            int fd = get_fd_(src->path, is_odirect);
            flush_io_uring_(fd, src);
            pending_fsync_q.push(src);
        } else {
            int fd = ::open(src->path.c_str(), O_WRONLY | O_CREAT, 0644);
            if (fd < 0) 
                throw std::system_error(errno, std::generic_category(), "open failed");
            ssize_t ret = pwrite_loop_(fd, src->ptr, src->size, src->file_start_offset);
            if (ret < 0) 
                throw std::system_error(errno, std::generic_category(), "pwrite failed");
            pending_fsync_q.push(src);
            ::close(fd);
            mem_pool->deallocate(src);
            perf_profiler.record_event(src, HOST_END);
            bool should_fsync = true;
            should_fsync = ( last_version != -1 && last_version != src->version );
            last_version = src->version;
            if (should_fsync) {
                while (pending_fsync_q.get_size()) {
                    auto entry = pending_fsync_q.get_front();
                    int fd = get_fd_(entry->path, false);
                    if (::fsync(fd) != 0)
                        throw std::system_error(errno, std::generic_category(), "fsync failed");
                    pending_fsync_q.pop();
                }
            }
        }
        flush_q.pop();
    }
}

size_t host_tier_t::flush_io_uring_(int fd, std::shared_ptr<mem_region_t> src) {
    size_t file_size = src->aligned_size > 0 ? src->aligned_size : src->size;
    size_t total_written = 0;

    // Enqueue all writes
    while (total_written < file_size) {
        size_t remaining = file_size - total_written;
        size_t to_write = std::min(remaining, MAX_FILE_WRITE_SIZE);

        struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
        if (!sqe) {
            FATAL("[io_uring] Failed to get SQE");
        }
        io_uring_prep_write(sqe,
                            fd,
                            src->ptr + total_written,
                            to_write,
                            src->file_start_offset + total_written);
        sqe->user_data = to_write;
        total_written += to_write;
        num_submitted++;
    }
    // Submit all enqueued writes
    int ret = io_uring_submit(&ring);
    if (ret < 0) {
        FATAL("[io_uring] io_uring_submit failed: " +
              std::string(strerror(-ret)));
    }
    return file_size;
}


void host_tier_t::fsync_io_uring_() {
    if (open_direct_files.size() == 0 && open_nondirect_files.size() == 0)
        return;
    std::cout << "[io_uring] Draining outstanding requests: " << num_submitted - num_completed << std::endl;

    struct io_uring_cqe *cqe;
    int ret;

    for(auto &entry: pending_fsync_q.q) {
        int fd = get_fd_(entry->path, false);
        struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
        if (!sqe) {
            FATAL("[io_uring] Failed to get SQE for fsync");
        }
        io_uring_prep_fsync(sqe, fd, 0);
        sqe->user_data = 0; 
        sqe->flags |= IOSQE_IO_DRAIN;
        num_submitted++;
    }

    ret = io_uring_submit(&ring);
    if (ret < 0) {
        FATAL("[io_uring] io_uring_submit failed: " +
              std::string(strerror(-ret)));
    }

    while (num_completed < num_submitted) {
        ret = io_uring_wait_cqe(&ring, &cqe);
        if (ret < 0) {
            FATAL("wait_cqe failed: " + std::string(strerror(-ret)));
        }
        if (cqe->res < 0) {
            FATAL("SQE failed: " + std::string(strerror(-cqe->res)) + " for num_completed=" + std::to_string(num_completed) 
                + " out of num_submitted=" + std::to_string(num_submitted) + " for fd= " + std::to_string(cqe->user_data));
        }
        io_uring_cqe_seen(&ring, cqe);
        num_completed++;
        if (cqe->user_data != 0) {
            if (cqe->res > MAX_FILE_WRITE_SIZE) {
                FATAL("[io_uring] Write size returned greater than MAX_FILE_WRITE_SIZE: " + std::to_string(cqe->res));
            }
            if (cqe->res != static_cast<int>(cqe->user_data)) {
                FATAL("[io_uring] Incomplete write: " + std::to_string(cqe->res) + " instead of total size " + std::to_string(cqe->user_data));
            }
        }
    }
    num_submitted = 0;
    num_completed = 0;

    while(pending_fsync_q.get_size()) {
        auto entry = pending_fsync_q.get_front();
        mem_pool->deallocate(entry);
        perf_profiler.record_event(entry, HOST_END);
        pending_fsync_q.pop();
    }

    for (auto &entry : open_direct_files) {
        ::close(entry.second);
    }
    open_direct_files.clear();
    for (auto &entry : open_nondirect_files) {
        ::close(entry.second);
    }
    open_nondirect_files.clear();
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
        size_t to_write = std::min(size - total_written, MAX_FILE_WRITE_SIZE);
        ssize_t written = pwrite(fd, ptr + total_written, to_write, file_start_offset + total_written);
        if (written <= 0) {
            throw std::runtime_error("[HostFlush] Incomplete or failed write: written "  + std::to_string(written) + 
                " instead of total size " + std::to_string(size) + " out of "  + std::to_string(to_write) + " in curr iteration, error: " + std::string(strerror(errno)));
        }
        total_written += written;
    }
    return total_written;
}