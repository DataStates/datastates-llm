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
// #include <torch/csrc/python_headers.h>
// #include <torch/csrc/utils/pybind.h>
#include <torch/torch.h>
// #include <torch/extension.h>
// #include <torch/script.h>
// #include <pybind11/stl.h>
#include <thread>
#include <tuple>
#include <iostream>
#include <cstdint>
#include <fstream>
#include <chrono>
#include <unistd.h>
#include "host_cache.hpp"

namespace py = pybind11;


static volatile uint64_t local_uid = 1;
class datastates_llm_t {    
    std::deque<std::tuple<int, uint64_t, const torch::Tensor, size_t, size_t, std::string>> _pending_d2h;
    std::mutex _mutex_d2h;
    std::condition_variable _cv_d2h;
    std::thread _thread_d2h;

    std::deque<std::tuple<int, uint64_t, char *, size_t, size_t, std::string>> _pending_h2f;
    std::mutex _mutex_h2f;
    std::condition_variable _cv_h2f;
    std::thread _thread_h2f;

    cudaStream_t _cpy_stream;    
    host_cache_t *mem;
    bool is_active = true;
    int _gpu_id = 0;
    int _rank = -1;
    void _d2h_trf();
    void _h2f_trf();
    
    public:
    datastates_llm_t(size_t host_cache_size, int gpu_id, int rank=-1);
    void ckpt_tensor(int version, const torch::Tensor &t, const std::uint64_t size, const std::uint64_t file_offset, std::string path);
    void wait();
    void shutdown();
};

#endif // __DATASTATES_HPP