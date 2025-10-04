#ifndef __DATASTATES_PERF_PROFILER_HPP
#define __DATASTATES_PERF_PROFILER_HPP

#include <chrono>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include "mem_region.hpp"
#include <stdexcept>
#include <cerrno>
#include <unistd.h>
#include <cassert>
#include <filesystem>
#include "json.hpp"

namespace datastates {
inline static bool ENABLE_PROFILING = true;
enum PERF_PROFILER_EVENT: int {
    GPU_WAIT_START=0,
    GPU_WAIT_END=1,
    HOST_WAIT_START=2,
    HOST_WAIT_END=3,
    GPU_START=4,
    GPU_END=5,
    HOST_START=6,
    HOST_END=7,
    FILE_START=8,
    FILE_END=9
};

struct profile_info_t {
    std::uint64_t gpu_wait_start_time = 0; // GPU wait start time in nanoseconds
    std::uint64_t gpu_wait_end_time = 0;   // GPU wait end time in nanoseconds
    std::uint64_t gpu_start_time = 0; // GPU start time in nanoseconds
    std::uint64_t gpu_end_time = 0;   // GPU end time in nanoseconds
    std::uint64_t host_wait_start_time = 0; // Host wait start time in nanoseconds
    std::uint64_t host_wait_end_time = 0;   // Host wait end time in nanoseconds
    std::uint64_t host_start_time = 0; // Host start time in nanoseconds
    std::uint64_t host_end_time = 0;   // Host end time in nanoseconds
    std::uint64_t host_start_time_version = 0; // Host start time version
    std::uint64_t host_end_time_version = 0;   // Host end time version
    size_t size = 0; // Size of the memory region
    std::uint64_t version = 0; // Version of the memory region
    std::uint64_t uid = 0;     // Unique identifier for the memory region
    std::uint64_t internal_uid = 0; // Internal unique identifier for tracking
    std::string path; // Path of the memory region
    profile_info_t() = default;
    profile_info_t(std::uint64_t internal_uid_) : internal_uid(internal_uid_) {}
};

class perf_profiler_t {
    using clock = std::chrono::high_resolution_clock;
    using duration = std::chrono::duration<uint64_t, std::nano>;
    std::unordered_map<std::uint64_t, profile_info_t> perf_profiles;
    std::mutex profiler_mutex;

public:
    perf_profiler_t() = default;
    perf_profiler_t(const perf_profiler_t&) = delete;
    perf_profiler_t& operator=(const perf_profiler_t&) = delete;
    static perf_profiler_t& get_instance() {
        static perf_profiler_t perf_profiler_instance;
        return perf_profiler_instance;
    }

    std::uint64_t get_current_time() const {
        if (!ENABLE_PROFILING) return 0;
        return std::chrono::duration_cast<duration>(clock::now().time_since_epoch()).count();
    }

    void record_event(std::shared_ptr<mem_region_t> m, PERF_PROFILER_EVENT e) {
        if (!ENABLE_PROFILING) return;
        std::unique_lock<std::mutex> lock(profiler_mutex);
        assert(m != nullptr && "Memory region cannot be null");
        assert(m->size > 0 && "Memory region size must be greater than zero");
        // assert(m->version > 0 && "Memory region version must be greater than zero");
        assert(m->internal_uid > 0 && "Memory region internal UID must be greater than zero");

        if (perf_profiles.find(m->internal_uid) == perf_profiles.end()) {
            profile_info_t info(m->internal_uid);
            info.size = m->size;
            info.version = m->version;
            info.uid = m->uid;
            info.path = m->path;
            info.internal_uid = m->internal_uid;
            perf_profiles[m->internal_uid] = info;
        }

        profile_info_t& info = perf_profiles[m->internal_uid];

        switch (e) {
            case GPU_WAIT_START: info.gpu_wait_start_time = get_current_time(); break;
            case GPU_WAIT_END: info.gpu_wait_end_time = get_current_time(); break;
            case HOST_WAIT_START: info.host_wait_start_time = get_current_time(); break;
            case HOST_WAIT_END: info.host_wait_end_time = get_current_time(); break;
            case GPU_START: info.gpu_start_time = get_current_time(); break;
            case GPU_END: info.gpu_end_time = get_current_time(); break;
            case HOST_START: info.host_start_time = get_current_time(); break;
            case HOST_END: info.host_end_time = get_current_time(); break;
            default: FATAL("Unknown PERF_PROFILER_EVENT type"); break;
        }
    }

    std::string report() const {
        try {
            nlohmann::json j_report = nlohmann::json{};
            for (const auto& [uid, info] : perf_profiles) {
                try {
                    std::filesystem::path filep(info.path);
                    std::string path = filep.filename().string();
                    std::string version = std::to_string(info.version);
                    if (!j_report.contains(version)) {
                        j_report[version] = nlohmann::json{};
                    }
                    if (!j_report[version].contains(path)) {
                        j_report[version][path] = nlohmann::json{
                            {"min_gpu_wait_start", std::numeric_limits<std::uint64_t>::max()},
                            {"max_gpu_wait_end", 0ULL},
                            {"min_host_wait_start", std::numeric_limits<std::uint64_t>::max()},
                            {"max_host_wait_end", 0ULL},
                            {"min_gpu_start", std::numeric_limits<std::uint64_t>::max()},
                            {"max_gpu_end", 0ULL},
                            {"min_host_start", std::numeric_limits<std::uint64_t>::max()},
                            {"max_host_end", 0ULL},
                            {"total_size", 0ULL}
                        };
                    }
                    // Some datastructures might be on host and might have 0 for GPU times. Avoid updating min/max in such cases.
                    if (info.gpu_wait_start_time > 0) {
                        j_report[version][path]["min_gpu_wait_start"] = std::min(j_report[version][path]["min_gpu_wait_start"].get<std::uint64_t>(), info.gpu_wait_start_time);
                    }
                    if (info.gpu_wait_end_time > 0) {
                        j_report[version][path]["max_gpu_wait_end"] = std::max(j_report[version][path]["max_gpu_wait_end"].get<std::uint64_t>(), info.gpu_wait_end_time);
                    }
                    if (info.host_wait_start_time > 0) {
                        j_report[version][path]["min_host_wait_start"] = std::min(j_report[version][path]["min_host_wait_start"].get<std::uint64_t>(), info.host_wait_start_time);
                    }
                    if (info.host_wait_end_time > 0) {
                        j_report[version][path]["max_host_wait_end"] = std::max(j_report[version][path]["max_host_wait_end"].get<std::uint64_t>(), info.host_wait_end_time);
                    }
                    if (info.gpu_start_time > 0) {
                        j_report[version][path]["min_gpu_start"] = std::min(j_report[version][path]["min_gpu_start"].get<std::uint64_t>(), info.gpu_start_time);
                    }
                    if (info.gpu_end_time > 0) {
                        j_report[version][path]["max_gpu_end"] = std::max(j_report[version][path]["max_gpu_end"].get<std::uint64_t>(), info.gpu_end_time);
                    }
                    if (info.host_start_time > 0) {
                        j_report[version][path]["min_host_start"] = std::min(j_report[version][path]["min_host_start"].get<std::uint64_t>(), info.host_start_time);
                    }
                    if (info.host_end_time > 0) {
                        j_report[version][path]["max_host_end"] = std::max(j_report[version][path]["max_host_end"].get<std::uint64_t>(), info.host_end_time);
                    }
                    j_report[version][path]["total_size"] = j_report[version][path]["total_size"].get<std::uint64_t>() + info.size;
                } catch (const std::exception& e) {
                    FATAL("Error reporting performance for UID " << uid << ": " << e.what());
                }
            }
            return write_to_shm(j_report.dump());
        } catch (const std::exception& e) {
            FATAL("Error generating performance report: " << e.what());
            return "";
        }
    }


    static std::string write_to_shm(const std::string& contents) {
        try {
            // Pattern must end with "XXXXXX" for mkstemp to replace in-place
            std::string pattern = "/dev/shm/ds_report_XXXXXX";
            std::vector<char> path(pattern.begin(), pattern.end());
            path.push_back('\0');

            int fd = mkstemp(path.data()); // creates and opens a unique file
            if (fd == -1) {
                throw std::runtime_error(std::string("mkstemp failed: ") + std::strerror(errno));
            }

            // Write all bytes (handle partial writes)
            const char* buf = contents.data();
            size_t to_write = contents.size();
            while (to_write > 0) {
                ssize_t n = ::write(fd, buf, to_write);
                if (n <= 0) {
                    int e = errno;
                    ::close(fd);
                    ::unlink(path.data()); // cleanup
                    throw std::runtime_error(std::string("write failed: ") + std::strerror(e));
                }
                buf += n;
                to_write -= static_cast<size_t>(n);
            }

            ::close(fd);
            return std::string(path.data()); // absolute path in /dev/shm
        } catch (const std::exception& e) {
            FATAL("Error writing to shared memory: " << e.what());
            return "";
        }
    }
};

} // namespace datastates

#endif // __DATASTATES_PERF_PROFILER_HPP