#ifndef __DATASTATES_UTILS_HPP
#define __DATASTATES_UTILS_HPP
#include <iostream>
#include <cuda_runtime.h>
#include <mutex>
#include <chrono>
#include <cassert>
#include <atomic>

#define checkCuda(ans) { checkCudaFunc((ans), __FILE__, __LINE__); }
inline void checkCudaFunc(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"========= GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


// Global logging mutex
inline std::mutex& log_mutex() {
    static std::mutex mtx;
    return mtx;
}
#define MESSAGE(level, message) \
{ \
    std::lock_guard<std::mutex> lock(log_mutex()); \
    std::cout << "[" << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << "] " << message << std::endl; \
}

#define COLOR_RED     "\033[1;31m"
#define COLOR_YELLOW  "\033[1;33m"
#define COLOR_RESET   "\033[0m"

#define FATAL(message) {\
    std::cout << COLOR_RED << " [!!ERROR!!] \t [" << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << "] " \
              << message << COLOR_RESET << std::endl << std::endl; \
    std::abort(); \
}

#define WARN(message) {\
    std::cout << COLOR_YELLOW << " [!!WARN!!] \t [" << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << "] " \
              << message << COLOR_RESET << std::endl; \
}

#define __PROFILE
#ifndef __PROFILE
    #define TIMER_START(t) {}
    #define TIMER_STOP(t, m, s) {}
#else
    static auto beginning = std::chrono::steady_clock::now();
    #define TIMER_START(timer) auto timer = std::chrono::steady_clock::now();
    #define TIMER_STOP(timer, message, size) {\
        auto now = std::chrono::steady_clock::now();\
        auto d = std::chrono::duration_cast<std::chrono::nanoseconds>(now - timer).count(); \
        auto t = std::chrono::duration_cast<std::chrono::seconds>(now - beginning).count();\
        std::cout << "[BENCHMARK] [" << message << "] [time elapsed: " << d << " ns] [size: " << size \
            << "] [throughput: " << (double)((double)size/(double)d) << "]" << std::endl; \
    }
#endif

// #define __DBG
#ifndef __DBG
    #define DBG(m) {}
#else
    #define DBG(message) MESSAGE("DEBUG", message)
#endif


inline const size_t get_fs_block_alignment() {
    return datastates::FS_BLOCK_SIZE_ALIGNMENT;
}

inline size_t get_aligned_offset(size_t offset, size_t alignment = get_fs_block_alignment()) {
    return (offset + alignment - 1) / alignment * alignment;
}

inline bool is_aligned(size_t offset) {
    return offset % get_fs_block_alignment() == 0;
}

#endif //__DATASTATES_UTILS_HPP