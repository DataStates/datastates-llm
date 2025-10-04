#ifndef __DATASTATES_FILE_HANDLER_BASE_HPP
#define __DATASTATES_FILE_HANDLER_BASE_HPP

#include "common/atomic_queue.hpp"
#include "pool/mem_pool.hpp"
#include "common/defs.hpp"
#include "common/mem_region.hpp"
#include "common/atomic_queue.hpp"
#include "common/perf_profiler.hpp"
#include <unordered_map>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <list>
#include <deque>
#include <unordered_set>

namespace datastates {
    class base_file_handler_t {
    protected:
        std::map<std::pair<std::string, bool>, int> open_fds_;
        std::list<std::pair<std::string, bool>> fifo_fds_;
        perf_profiler_t& perf_profiler;
        bool is_active = true;
        std::shared_ptr<mem_pool_t> mem_pool;
        // Filesystems such as /tmp and /dev/shm may not support O_DIRECT.
        // We optimistically assume that all paths will go to the same filesystem.
        int supports_odirect_ = -1; // -1: unknown, 0: no, 1: yes
    public:
        base_file_handler_t(std::shared_ptr<mem_pool_t> pool);
        ~base_file_handler_t();
        bool check_odirect_support_(const std::string& path);
        int get_fd_(const std::string& path, bool is_odirect);
        void close_fds_();
        virtual void write(std::shared_ptr<mem_region_t> m, bool is_odirect=false) = 0;
        virtual void read(std::shared_ptr<mem_region_t> m, bool is_odirect=false) = 0;
        virtual void fsync() { return ; };
        
    };
}


#endif // __DATASTATES_FILE_HANDLER_BASE_HPP