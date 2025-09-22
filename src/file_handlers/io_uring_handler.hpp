#ifndef __DATASTATES_FILE_HANDLERS_IO_URING_HANDLER_HPP
#define __DATASTATES_FILE_HANDLERS_IO_URING_HANDLER_HPP

#include "base_file_handler.hpp"
#include <liburing.h>
#include <unordered_map>
#include <string>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace datastates {


class io_uring_handler_t: public base_file_handler_t {
    private:
        struct io_chunk_status {
            std::shared_ptr<mem_region_t> mem_region;
            size_t size;
        };

        struct io_uring ring;
        std::atomic<size_t> num_submitted = 0; 
        std::atomic<size_t> num_completed = 0;
        std::map<uint64_t, io_chunk_status> io_status_map;
        std::map<uint64_t, int> chunk_counter;
        std::thread io_uring_wait_thread_;
        std::mutex io_uring_wait_mutex_;
        std::condition_variable io_uring_wait_cv_;
        size_t flush_io_uring_(int fd, std::shared_ptr<mem_region_t> src);
        void wait_on_io_uring_();
    public:
        io_uring_handler_t(std::shared_ptr<mem_pool_t> pool);
        ~io_uring_handler_t();
        void write(std::shared_ptr<mem_region_t> m, bool is_odirect=false) override;
        void read(std::shared_ptr<mem_region_t> m, bool is_odirect=false) override;
        void fsync() override;
};
}

#endif // __DATASTATES_FILE_HANDLERS_IO_URING_HANDLER_HPP