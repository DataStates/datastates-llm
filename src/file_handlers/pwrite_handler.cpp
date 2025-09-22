#include "pwrite_handler.hpp"

using namespace datastates;
pwrite_handler_t::pwrite_handler_t(std::shared_ptr<mem_pool_t> pool): base_file_handler_t(pool) {
    std::cout << "Not using io_uring, using pwrite file handler" << std::endl;
    DBG("[pwrite_handler] Initialized pwrite file handler");
}

pwrite_handler_t::~pwrite_handler_t() {
    DBG("[pwrite_handler] Destroyed pwrite file handler");
}

void pwrite_handler_t::write(std::shared_ptr<mem_region_t> m, bool is_odirect) {
    try {
        int fd = get_fd_(m->path, is_odirect);
        size_t file_size = m->aligned_size > 0 ? m->aligned_size : m->size;
        size_t total_written = 0;
        while (total_written < file_size) {
            size_t remaining = file_size - total_written;
            size_t to_write = std::min(remaining, MAX_FILE_WRITE_SIZE);
            ssize_t ret = pwrite(fd,
                                m->ptr + total_written,
                                to_write,
                                m->file_start_offset + total_written);
            if (ret < 0) {
                if (errno == EINTR) continue; // Interrupted, just retry the write.
                FATAL("[pwrite_handler] pwrite failed: " + std::string(strerror(errno)));
            }
            total_written += ret;
        }
        if (::fsync(fd) != 0) {
            throw std::system_error(errno, std::generic_category(), "fsync failed");
        }
        mem_pool->deallocate(m);
        perf_profiler.record_event(m, HOST_END);
        return;
    } catch (const std::exception& ex) {
        FATAL("[ERROR][pwrite_handler] I/O operation failed for " << m->path 
                    << " with exception: " << ex.what());
    }
}

void pwrite_handler_t::read(std::shared_ptr<mem_region_t> m, bool is_odirect) {
    int fd = get_fd_(m->path, is_odirect);
    size_t file_size = m->aligned_size > 0 ? m->aligned_size : m->size;
    size_t total_read = 0;
    while (total_read < file_size) {
        size_t remaining = file_size - total_read;
        size_t to_read = std::min(remaining, MAX_FILE_WRITE_SIZE);
        ssize_t ret = pread(fd,
                            m->ptr + total_read,
                            to_read,
                            m->file_start_offset + total_read);
        if (ret < 0) {
            if (errno == EINTR) continue; // Interrupted, just retry the read.
            FATAL("[pwrite_handler] pread failed: " + std::string(strerror(errno)));
        }
        total_read += ret;
    }
    ::close(fd);
    return;
}

void pwrite_handler_t::fsync() {
    return;
}