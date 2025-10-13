#include "io_uring_handler.hpp"

using namespace datastates;
io_uring_handler_t::io_uring_handler_t(std::shared_ptr<mem_pool_t> pool): base_file_handler_t(pool) {
    std::cout << "[HOST_TIER][io_uring] Initializing io_uring instance." << std::endl;
    if (get_fs_block_alignment() <= 1) {
        FATAL("[HOST_TIER][io_uring] Filesystem block size alignment must be greater than 1 to use io_uring.");
    }
    int ret = io_uring_queue_init(512, &ring, 0);
    if (ret < 0) {
        FATAL("[io_uring] io_uring_queue_init failed: " +
              std::string(strerror(-ret)));
    }
    io_uring_wait_thread_ = std::thread([&] { wait_on_io_uring_(); });
    DBG("[io_uring] Initialized io_uring module");
}

io_uring_handler_t::~io_uring_handler_t() {
    fsync();
    is_active = false;
    io_uring_wait_cv_.notify_all();
    io_uring_wait_thread_.join();
    io_uring_queue_exit(&ring);
    DBG("[io_uring] Destroyed io_uring module");
}

void io_uring_handler_t::write(std::shared_ptr<mem_region_t> m, bool is_odirect) {
    assert((chunk_counter.find(m->internal_uid) == chunk_counter.end() || chunk_counter[m->internal_uid] == 0) 
        && "[HOST_TIER][io_uring] Previous chunks for this memory region are still being processed.");
    size_t file_size = m->aligned_size > 0 ? m->aligned_size : m->size;
    size_t total_written = 0;
    struct io_uring_sqe *sqe;
    std::unique_lock<std::mutex> io_uring_lock_(io_uring_wait_mutex_, std::defer_lock);
    int num_ops = 0;
    int fd = get_fd_(m->path, is_odirect);
    while (total_written < file_size) {
        size_t remaining = file_size - total_written;
        size_t to_write = std::min(remaining, MAX_FILE_WRITE_SIZE);

        sqe = io_uring_get_sqe(&ring);
        if (!sqe) {
            FATAL("[io_uring] Failed to get SQE");
        }
        io_uring_prep_write(sqe,
                            fd,
                            m->ptr + total_written,
                            to_write,
                            m->file_start_offset + total_written);
        total_written += to_write;
        io_uring_lock_.lock();
        sqe->user_data = uring_submission_id_++;
        num_ops  += 1;
        io_status_map[sqe->user_data] = {m, to_write, fd};
        io_uring_lock_.unlock();
    }
    // Submit all enqueued writes
    int ret = io_uring_submit(&ring);
    if (ret < 0) {
        FATAL("[io_uring] io_uring_submit failed: " + std::string(strerror(ret)));
    }
    if (ret != num_ops) {
        FATAL("[io_uring] io_uring_submit submitted " + std::to_string(ret) + " out of " + std::to_string(num_ops) + " operations");
    }
    io_uring_lock_.lock();
    chunk_counter[m->internal_uid] = num_ops;
    num_submitted += num_ops;
    io_uring_lock_.unlock();
    io_uring_wait_cv_.notify_all();
}


void io_uring_handler_t::wait_on_io_uring_() {
    try {
        struct io_uring_cqe* cqe; // Batch processing buffer
        struct io_uring_sqe* sqe;
        std::unique_lock<std::mutex> io_uring_lock_(io_uring_wait_mutex_, std::defer_lock);
        while (true) {
            io_uring_lock_.lock();
            while (is_active && num_completed >= num_submitted) {
                io_uring_wait_cv_.wait(io_uring_lock_);
            }
            if (!is_active) return;
            io_uring_lock_.unlock();
            int ret = 0;
            ret = io_uring_wait_cqe(&ring, &cqe);
            if (ret < 0) {
                if (-ret == EINTR) continue; // Interrupted, just retry the loop.
                FATAL("[io_uring_handler] io_uring_wait_cqe failed: " + std::string(strerror(-ret)));
            }
            if (cqe->res < 0) 
                FATAL("[io_uring_handler] SQE failed: " + std::string(strerror(-cqe->res)));
            uint64_t user_data = cqe->user_data;
            io_uring_lock_.lock();
            auto it = io_status_map.find(user_data);
            if (it == io_status_map.end()) {
                FATAL("[io_uring_handler] Could not find user_data in io_status_map " + std::to_string(user_data) + " of size " + std::to_string(cqe->res)
                    + " num_completed " + std::to_string(num_completed) + " num_submitted " + std::to_string(num_submitted) 
                    + " io_status_map size " + std::to_string(io_status_map.size()));
            }

            io_chunk_status& info = it->second;
            if (info.size < static_cast<size_t>(cqe->res)) 
                FATAL("[io_uring_handler] Overwrite error, written " + std::to_string(cqe->res) + " expected max " + std::to_string(info.size));
            
            if (info.size > static_cast<size_t>(cqe->res)) {
                size_t written = cqe->res;
                size_t remaining = info.size - written;
                off_t new_off = info.mem_region->file_start_offset + (info.size - remaining);

                sqe = io_uring_get_sqe(&ring);
                io_uring_prep_write(sqe,
                                    info.fd,
                                    (char*)info.mem_region->ptr + (info.size - remaining),
                                    remaining,
                                    new_off);
                sqe->user_data = uring_submission_id_++;
                io_status_map[sqe->user_data] = {info.mem_region, remaining, info.fd};
                num_submitted += 1;
                chunk_counter[info.mem_region->internal_uid] += 1;
                io_uring_submit(&ring);
            }

            io_uring_cqe_seen(&ring, cqe);
            num_completed += 1;
            chunk_counter[info.mem_region->internal_uid] -= 1;
            if (chunk_counter[info.mem_region->internal_uid] == 0) {
                chunk_counter.erase(info.mem_region->internal_uid);
                mem_pool->deallocate(info.mem_region);
                perf_profiler.record_event(info.mem_region, HOST_END);
            }
            io_status_map.erase(it);
            io_uring_lock_.unlock();
            io_uring_wait_cv_.notify_all();
        }
    } catch (const std::exception& ex) {
        FATAL("[io_uring_handler] Got exception " << ex.what());
    }
}

void io_uring_handler_t::read(std::shared_ptr<mem_region_t> m, bool is_odirect) {
    int fd = get_fd_(m->path, is_odirect);
    size_t file_size = m->aligned_size > 0 ? m->aligned_size : m->size;
    size_t total_read = 0;
    std::unique_lock<std::mutex> io_uring_lock_(io_uring_wait_mutex_, std::defer_lock);
    while (total_read < file_size) {
        size_t remaining = file_size - total_read;
        size_t to_read = std::min(remaining, MAX_FILE_WRITE_SIZE);

        struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
        if (!sqe) {
            FATAL("[io_uring] Failed to get SQE");
        }
        io_uring_prep_read(sqe,
                           fd,
                           m->ptr + total_read,
                           to_read,
                           m->file_start_offset + total_read);
        total_read += to_read;
        io_uring_lock_.lock();
        sqe->user_data = uring_submission_id_++;
        io_status_map[sqe->user_data] = {m, to_read, fd};
        num_submitted++;
        io_uring_lock_.unlock();
    }
    int ret = io_uring_submit(&ring);
    if (ret < 0) {
        FATAL("[io_uring] io_uring_submit failed: " +
              std::string(strerror(-ret)));
    }
    io_uring_wait_cv_.notify_all();
    // We can wait for all reads to complete here since we don't use the buffer until the read is done
    // This can be optimized later to overlap reads with computation if needed
    fsync();
}

void io_uring_handler_t::fsync() {
    std::unique_lock<std::mutex> io_uring_lock_(io_uring_wait_mutex_);
    while (is_active && num_submitted != num_completed) {
        io_uring_wait_cv_.wait(io_uring_lock_);
    }
    io_uring_lock_.unlock();
}