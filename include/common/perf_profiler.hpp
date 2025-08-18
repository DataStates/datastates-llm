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
#define NDEBUG
#include <cassert>
#include "json.hpp"

namespace datastates {

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
    uint gpu_wait_start_time = 0; // GPU wait start time in nanoseconds
    uint gpu_wait_end_time = 0;   // GPU wait end time in nanoseconds
    uint gpu_start_time = 0; // GPU start time in nanoseconds
    uint gpu_end_time = 0;   // GPU end time in nanoseconds
    uint host_wait_start_time = 0; // Host wait start time in nanoseconds
    uint host_wait_end_time = 0;   // Host wait end time in nanoseconds
    uint host_start_time = 0; // Host start time in nanoseconds
    uint host_end_time = 0;   // Host end time in nanoseconds
    size_t size = 0; // Size of the memory region
    uint version = 0; // Version of the memory region
    uint uid = 0;     // Unique identifier for the memory region
    uint internal_uid = 0; // Internal unique identifier for tracking
    std::string path; // Path of the memory region
    profile_info_t() = default;
    profile_info_t(uint internal_uid_) : internal_uid(internal_uid_) {}
    uint get_duration(TIER_TYPES tier=FILE_TIER) const {
        if (tier == FILE_TIER && (gpu_start_time + gpu_end_time + host_start_time + host_end_time) > 0) {
            // Return the total time from GPU to the File tier
            return gpu_end_time - gpu_start_time + host_end_time - host_start_time;
        } else if (tier == GPU_TIER && (gpu_start_time + gpu_end_time) > 0) {
            // Return the total time from Host to the GPU tier
            return gpu_end_time - gpu_start_time;
        } else if ((tier == HOST_PINNED_TIER || tier == HOST_UNPINNED_TIER) && (host_start_time + host_end_time) > 0) {
            // Return the total time from File to the Host tier
            return host_end_time - host_start_time;
        }
        FATAL("Invalid tier type for duration calculation for " + path + " with times (gpu_start_time: "
            + std::to_string(gpu_start_time) + ", gpu_end_time: " + std::to_string(gpu_end_time) + ", "
            + "host_start_time: " + std::to_string(host_start_time) + ", host_end_time: " + std::to_string(host_end_time) + ")");
        return 0; // This line will never be reached, but added to avoid compiler warnings
    }
};

inline void to_json(nlohmann::json& j, const profile_info_t& info) {
    j = nlohmann::json{
        {"gpu_time", info.gpu_end_time - info.gpu_start_time},
        {"host_time", info.host_end_time - info.host_start_time},
        {"gpu_wait_time", info.gpu_wait_end_time - info.gpu_wait_start_time},
        {"host_wait_time", info.host_wait_end_time - info.host_wait_start_time},
        {"size", info.size},
        {"version", info.version},
        {"uid", info.uid},
        {"internal_uid", info.internal_uid},
        {"path", info.path.substr(info.path.find_last_of("/\\") + 1)}
    };
}

class perf_profiler_t {
    using clock = std::chrono::high_resolution_clock;
    using duration = std::chrono::duration<unsigned long long, std::nano>;
    std::unordered_map<uint, profile_info_t> perf_profiles;

public:
    perf_profiler_t() = default;
    perf_profiler_t(const perf_profiler_t&) = delete;
    perf_profiler_t& operator=(const perf_profiler_t&) = delete;
    static perf_profiler_t& get_instance() {
        static perf_profiler_t perf_profiler_instance;
        return perf_profiler_instance;
    }

    uint get_current_time() const {
        return std::chrono::duration_cast<duration>(clock::now().time_since_epoch()).count();
    }

    void record_event(std::shared_ptr<mem_region_t> m, PERF_PROFILER_EVENT e) {
        assert(m != nullptr && "Memory region cannot be null");
        assert(m->size > 0 && "Memory region size must be greater than zero");
        assert(m->version > 0 && "Memory region version must be greater than zero");
        assert(m->uid > 0 && "Memory region UID must be greater than zero");
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
        if (e == GPU_WAIT_START) {
            info.gpu_wait_start_time = get_current_time();
        } else if (e == GPU_WAIT_END) {
            assert(info.gpu_wait_start_time > 0 && "GPU wait start time must be set before GPU wait end time");
            info.gpu_wait_end_time = get_current_time();
        } else if (e == HOST_WAIT_START) {
            info.host_wait_start_time = get_current_time();
        } else if (e == HOST_WAIT_END) {
            assert(info.host_wait_start_time > 0 && "Host wait start time must be set before Host wait end time");
            info.host_wait_end_time = get_current_time();
        } else if (e == GPU_START) {
            assert(info.gpu_start_time == 0 && "GPU start time should not be set before GPU_START event");
            info.gpu_start_time = get_current_time();
        } else if (e == GPU_END) {
            assert(info.gpu_end_time == 0 && "GPU end time should not be set before GPU_END event");
            assert(info.gpu_start_time > 0 && "GPU start time must be set before GPU end time");
            info.gpu_end_time = get_current_time();
        } else if (e == HOST_START) {
            assert(info.host_start_time == 0 && "Host start time should not be set before HOST_START event");
            info.host_start_time = get_current_time();
        } else if (e == HOST_END) {
            assert(info.host_end_time == 0 && "Host end time should not be set before HOST_END event");
            assert(info.host_start_time > 0 && "Host start time must be set before Host end time");
            info.host_end_time = get_current_time();
        } else {
            FATAL("Unknown PERF_PROFILER_EVENT type");
        }
    }

    std::string report() const {
        nlohmann::json j_report = nlohmann::json{};

        for (const auto& [uid, info] : perf_profiles) {
            try {
                std::string path = info.path;
                if (!j_report.contains(path)) {
                    j_report[path] = nlohmann::json{
                        {"gpu_wait_time", 0ULL},
                        {"host_wait_time", 0ULL},
                        {"gpu_time", 0ULL},
                        {"host_time", 0ULL}
                    };
                }
                j_report[path]["gpu_wait_time"] = j_report[path]["gpu_wait_time"].get<uint64_t>()  + (info.gpu_wait_end_time - info.gpu_wait_start_time);
                j_report[path]["host_wait_time"] = j_report[path]["host_wait_time"].get<uint64_t>()  + (info.host_wait_end_time - info.host_wait_start_time);
                j_report[path]["gpu_time"] = j_report[path]["gpu_time"].get<uint64_t>()  + (info.gpu_end_time - info.gpu_start_time);
                j_report[path]["host_time"] = j_report[path]["host_time"].get<uint64_t>()  + (info.host_end_time - info.host_start_time);
            } catch (const std::exception& e) {
                std::cerr << "Error reporting performance for UID " << uid << ": " << e.what() << std::endl;
            }
        }
        return write_to_shm(j_report.dump());
    }


    static std::string write_to_shm(const std::string& contents) {
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
    }
};

} // namespace datastates

#endif // __DATASTATES_PERF_PROFILER_HPP
