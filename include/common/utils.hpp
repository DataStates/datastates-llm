#ifndef __DATASTATES_UTILS_HPP
#define __DATASTATES_UTILS_HPP
#include <iostream>
#include <cuda_runtime.h>
#include <mutex>
#include <chrono>

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

// #define MESSAGE(level, message) std::cout << "[" << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << "] " << message << std::endl
// #define MESSAGE(level, message) py::print("[", __FILE__, ":", __LINE__, ":", __FUNCTION__, "] ", message)
#define FATAL(message) {\
    MESSAGE("FATAL", message);\
    std::abort(); \
}

// #define __PROFILE
#ifndef __PROFILE
    #define TIMER_START(t) {}
    #define TIMER_STOP(t, m, s) {}
    #define DBG(m) {}
#else
    static auto beginning = std::chrono::steady_clock::now();
    #define TIMER_START(timer) auto timer = std::chrono::steady_clock::now();
    #define TIMER_STOP(timer, message, size) {\
        auto now = std::chrono::steady_clock::now();\
        auto d = std::chrono::duration_cast<std::chrono::nanoseconds>(now - timer).count(); \
        auto t = std::chrono::duration_cast<std::chrono::seconds>(now - beginning).count();\
        std::cout << "[BENCHMARK " << t << "] [" << __FILE__ << ":" << __LINE__ << ":" \
            << __FUNCTION__ << "] [time elapsed: " << d << " ns] " << message \
            << " [THRU: " << (double)((double)size/(double)d) << "]" << std::endl; \
    }
    #define DBG(message) MESSAGE("DEBUG", message)
#endif


extern size_t FS_BLOCK_SIZE_ALIGNMENT;
inline void set_fs_block_alignment(size_t alignment) {
    FS_BLOCK_SIZE_ALIGNMENT = alignment;
}
inline const size_t get_fs_block_alignment() {
    return FS_BLOCK_SIZE_ALIGNMENT;
}

inline size_t get_aligned_offset(size_t offset, size_t alignment = get_fs_block_alignment()) {
    return (offset + alignment - 1) / alignment * alignment;
}

inline bool is_aligned(size_t offset) {
    return offset % get_fs_block_alignment() == 0;
}

#endif //__DATASTATES_UTILS_HPP