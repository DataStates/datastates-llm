#include "base_file_handler.hpp"

using namespace datastates;
base_file_handler_t::base_file_handler_t(std::shared_ptr<mem_pool_t> p): 
    perf_profiler(perf_profiler_t::get_instance()), mem_pool(p) {
    // Nothing to do
}

base_file_handler_t::~base_file_handler_t() {
    close_fds_();
}


void base_file_handler_t::close_fds_() {
    for (auto& entry : open_fds_) {
        ::fsync(entry.second);
        ::close(entry.second);
    }
    open_fds_.clear();
    fifo_fds_.clear();
}

bool base_file_handler_t::check_odirect_support_(const std::string& path) {
    if (supports_odirect_ == -1) {
            // Create a temporary file path in the same directory as the target file.
        std::filesystem::path p(path);
        std::string temp_file_path = p.parent_path().string() + "/.odirect_check_datastates_tmp";

        // Open the temp file with O_CREAT to ensure it exists for the check.
        int test_fd = ::open(temp_file_path.c_str(), O_WRONLY | O_CREAT | O_DIRECT, 0644);

        if (test_fd >= 0) {
            // Success! The filesystem supports O_DIRECT.
            supports_odirect_ = 1;
            ::close(test_fd);
            ::unlink(temp_file_path.c_str()); // Clean up the temporary file.
            DBG("[BaseFileHandler] Filesystem supports O_DIRECT.");
        } else {
            if (errno == EINVAL) {
                // This is the expected error on filesystems that don't support O_DIRECT.
                supports_odirect_ = 0;
                WARN("[BaseFileHandler] Filesystem does not support O_DIRECT. Tested path: " + path);
            } else {
                // Any other error is unexpected. We can warn but shouldn't be fatal.
                // We'll conservatively assume no support.
                supports_odirect_ = 0;
                WARN("[BaseFileHandler] Could not verify O_DIRECT support. Error: " + std::string(strerror(errno)) + ". Disabling O_DIRECT. Tested path: " + path);
                // We don't need to unlink here because the file was never created.
            }
        }
    }
    return supports_odirect_ == 1;
}

int base_file_handler_t::get_fd_(const std::string& path, bool is_odirect) {
    // Create a composite key from the path and the O_DIRECT flag.
    const auto key = std::make_pair(path, is_odirect);

    // Check if the file descriptor is already cached in our single map.
    auto map_it = open_fds_.find(key);
    if (map_it != open_fds_.end()) {
        return map_it->second;
    }

    // The file is not in the cache. Check if we need to evict the oldest FD before opening a new one.
    if (open_fds_.size() >= MAX_OPEN_FDS) {
        // Get the oldest item's key (from the front of the queue).
        const auto& oldest_key = fifo_fds_.front();
        // Find the corresponding entry in the main map.
        auto oldest_it = open_fds_.find(oldest_key);
        if (oldest_it != open_fds_.end()) {
            ::close(oldest_it->second);
            open_fds_.erase(oldest_it);
        }
        // Clean up the FIFO tracking queue.
        fifo_fds_.pop_front();
    }

    // Now, it's safe to open the new file.
    int flags = O_RDWR | O_CREAT;
    if (is_odirect && check_odirect_support_(path)) {
        flags |= O_DIRECT;
    }
    int fd = ::open(path.c_str(), flags, 0644);
    if (fd < 0) {
        FATAL("[BaseFileHandler] Failed to open file: " + path + " Error: " + strerror(errno));
    }
    // Add the new file to the single cache and to the back of the FIFO queue.
    open_fds_[key] = fd;
    fifo_fds_.push_back(key);
    return fd;
}
