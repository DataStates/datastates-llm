#ifndef __DATASTATES_CACHE_HPP
#define __DATASTATES_CACHE_HPP
#include <atomic>
#include <iostream>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <cuda_runtime.h>
#include <deque>
#include <cassert>
#include <cstdlib>
#include <chrono>
#include <fstream>  
#include <cstdint>
#include <cmath>
#include <thread>
#include <cstdlib>
#include <sys/mman.h>
#include <cstring>

#define checkCuda(ans) { checkCudaFunc((ans), __FILE__, __LINE__); }
inline void checkCudaFunc(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"========= GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


static auto beginning = std::chrono::steady_clock::now();
#define TIMER_START(t) {}
#define TIMER_STOP(t, m, s) {}
// #define TIMER_START(timer) auto timer = std::chrono::steady_clock::now();
// #define TIMER_STOP(timer, message, size) {\
//     auto now = std::chrono::steady_clock::now();\
//     auto d = std::chrono::duration_cast<std::chrono::nanoseconds>(now - timer).count(); \
//     auto t = std::chrono::duration_cast<std::chrono::seconds>(now - beginning).count();\
//     std::cout << "[BENCHMARK " << t << "] [" << __FILE__ << ":" << __LINE__ << ":" \
//         << __FUNCTION__ << "] [time elapsed: " << d << " ns] " << message \
//         << " [THRU: " << (double)((double)size/(double)d) << "]" << std::endl; \
// }

#define MESSAGE(level, message) \
    std::cout << "[" \
        << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << "] " << message << std::endl

#define DBG(message) MESSAGE("DEBUG", message)
// #define DBG(m) {}

#define FATAL(message) {\
    MESSAGE("FATAL", message);\
    std::abort(); \
}

struct mem_region_t {
    uint64_t    uid;
    char        *ptr;
    size_t      start_offset;
    size_t      end_offset;
    mem_region_t(uint64_t u, char *p, size_t s, size_t e): uid(u), ptr(p), start_offset(s), end_offset(e) {};
};

class host_cache_t {
    int _device_id;
    std::atomic<size_t> _total_memory;
    std::atomic<size_t> _curr_size;
    std::atomic<size_t> _head;
    std::atomic<size_t> _tail;
    char* _start_ptr;

    std::mutex _mem_mutex;
    std::condition_variable _mem_cv;
    std::deque<mem_region_t*> _mem_q;
    size_t max_allocated = 0;
    bool is_active = true;
    int _rank = -1;
    void _print_trace();
    mem_region_t* _assign(const uint64_t uid, size_t h, size_t s);
public:
    host_cache_t(size_t t, int d, int rank);
    ~host_cache_t();    
    mem_region_t* allocate(const uint64_t uid, size_t s);
    void deallocate(uint64_t _uid, size_t s);
    void shutdown();
};


#endif //__DATASTATES_CACHE_HPP