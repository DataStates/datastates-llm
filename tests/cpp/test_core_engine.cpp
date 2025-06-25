#include <iostream>
#include <vector>
#include <numeric>
#include "datastates.hpp"
#include <cuda_runtime.h>
#define __PROFILE
#include "common/utils.hpp"

int main() {
    std::cout << "Starting DataStates-LLM C++ Core Engine Test..." << std::endl;
    size_t host_buffer_size = 1ULL<<30; // 1 GB
    int gpu_id = 0; 
    int rank = -1; 
    datastates::core_t* engine = datastates::dstates_engine(host_buffer_size, gpu_id, rank);
    if (!engine) {
        std::cerr << "Failed to initialize DataStates-LLM core engine." << std::endl;
        return 1;
    }

    // Initializing dummy application datastructures.
    size_t nelements = 1<<20; // 1 million elements
    std::string ckpt_path = "/tmp/test_datastates_core.ckpt";
    float dummy_data = 12345;
    std::vector<float> cpu_data(nelements, dummy_data);
    float expected_sum = std::accumulate(cpu_data.begin(), cpu_data.end(), 0.0f);

    float* gpu_data;
    checkCuda(cudaMalloc(&gpu_data, nelements * sizeof(float)));
    checkCuda(cudaMemcpy(gpu_data, cpu_data.data(), nelements * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaDeviceSynchronize());

    // Start the checkpointing process
    TIMER_START(t);
    size_t curr_size = 0;
    engine->ckpt(1, 0, reinterpret_cast<char*>(cpu_data.data()), nelements * sizeof(float), curr_size, ckpt_path);
    curr_size += nelements * sizeof(float);
    engine->ckpt(1, 1, reinterpret_cast<char*>(gpu_data), nelements * sizeof(float), curr_size, ckpt_path);
    curr_size += nelements * sizeof(float);
    TIMER_STOP(t, "Async checkpointing launched", curr_size);

    TIMER_START(t_wait);
    engine->wait(true);
    TIMER_STOP(t_wait, "Wait for checkpointing to persist", curr_size);

    // Start the restore process
    cpu_data.assign(nelements, 0.0f);
    std::vector<float> gpu_data_on_host(nelements, 0.0f); /* restore the GPU data on host-buffer because direct GPU restore is not yet supported */
    TIMER_START(t_restore);
    curr_size = 0;
    engine->restore(1, 0, reinterpret_cast<char*>(cpu_data.data()), nelements * sizeof(float), curr_size, ckpt_path);
    curr_size += nelements * sizeof(float);
    engine->restore(1, 1, reinterpret_cast<char*>(gpu_data_on_host.data()), nelements * sizeof(float), curr_size, ckpt_path);
    curr_size += nelements * sizeof(float);
    TIMER_STOP(t_restore, "Restore process completed", curr_size);

    // Start the verification process
    float cpu_sum = std::accumulate(cpu_data.begin(), cpu_data.end(), 0.0f);
    if (cpu_sum != expected_sum) {
        std::cerr << "CPU data verification failed: expected sum " << expected_sum << ", got " << cpu_sum << std::endl;
        return -1;
    }
    float gpu_data_on_host_sum = std::accumulate(gpu_data_on_host.begin(), gpu_data_on_host.end(), 0.0f);
    if (gpu_data_on_host_sum != expected_sum) {
        std::cerr << "GPU data verification failed: expected sum " << expected_sum << ", got " << gpu_data_on_host_sum << std::endl;
        return -1;
    }

    return 0;
}
