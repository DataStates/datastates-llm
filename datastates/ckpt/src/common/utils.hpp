#ifndef __DATASTATES_UTILS_HPP
#define __DATASTATES_UTILS_HPP
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define checkCuda(ans) { checkCudaFunc((ans), __FILE__, __LINE__); }
inline void checkCudaFunc(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"========= GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#define MESSAGE(level, message) std::cout << "[" << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << "] " << message << std::endl
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

#endif //__DATASTATES_UTILS_HPP