#ifndef __DATASTATES_HPP
#define __DATASTATES_HPP

#include <stdlib.h>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <cuda_runtime.h>
#include <tuple>
#include <iostream>
#include <exception>
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <thread>
#include <tuple>
#include <iostream>
#include <cstdint>
#include <fstream>
#include <chrono>
#include <unistd.h>
#include "host_cache.hpp"
#include "omp.h"
#include <algorithm>
#include <stdlib.h>   // posix_memalign
#include <sys/mman.h> // madvise
#include <cstddef>     // static_cast<size_t>
#include <fcntl.h>
#include <algorithm>
#include <stdlib.h>   // posix_memalign
#include <sys/mman.h> // madvise
#include <cstddef>     // static_cast<size_t>
#include <fcntl.h>

#define MIN_TENSOR_SIZE static_cast<size_t>(1<<25)
#define HUGEPAGES_SIZE static_cast<size_t>(1<<21)
#define CHUNK_SIZE static_cast<size_t>(1<<27)
#define READ_THREADS (4)
namespace py = pybind11;


static volatile uint64_t local_uid = 1;
class datastates_llm_t {
    // Store the <ckpt_version, local_uid, tensor reference, tensor size, file_start_offset, file_path>
    std::deque<std::tuple<int, uint64_t, const torch::Tensor, size_t, size_t, std::string>> _pending_d2h;
    std::mutex              _mutex_d2h;
    std::condition_variable _cv_d2h;
    std::thread             _thread_d2h;

    // Store the <ckpt_version, local_uid, host_pointer, tensor size, file_start_offset, file_path>
    std::deque<std::tuple<int, uint64_t, char *, size_t, size_t, std::string>> _pending_h2f;
    std::mutex              _mutex_h2f;
    std::condition_variable _cv_h2f;
    std::thread             _thread_h2f;

    cudaStream_t _cpy_stream;    
    host_cache_t *mem;
    bool is_active = true;
    int _gpu_id = 0;
    int _rank = -1;
    void _d2h_trf();
    void _h2f_trf();

    // Restart functions and datastructures.
    void _alloc_tensor();
    // Store the tensor_pointer, total_size
    std::deque<std::tuple<void*, size_t>> _pending_alloc;
    // Store the tensor_pointer, allocated_size
    std::map<void *, size_t> _alloc_map;
    std::mutex              _mutex_alloc;
    std::condition_variable _cv_alloc;
    std::thread             _thread_alloc;

    void _read_file(int thread_id);
    // Store the tensor_pointer, start_offset, end_offset, file descriptor
    std::deque<std::tuple<void *, size_t, size_t, int>> _pending_read[READ_THREADS];
    std::mutex              _mutex_read[READ_THREADS];
    std::condition_variable _cv_read[READ_THREADS];
    std::thread             _thread_read[READ_THREADS];
    
    public:
    datastates_llm_t(size_t host_cache_size, int gpu_id, int rank=-1);
    void ckpt_tensor(int version, const torch::Tensor &t, const std::uint64_t size, const std::uint64_t file_offset, std::string path);
    
    void alloc_tensor_queue(const torch::Tensor &t);
    void restore_tensor(const torch::Tensor &t, std::string path, const std::uint64_t start_offset, const std::uint64_t end_offset);
    void wait();
    void shutdown();
};

#endif // __DATASTATES_HPP