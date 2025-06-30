#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <state_provider.hpp>

int main() {
    std::cout << "VLCC States C++ Test" << std::endl;

    
    char* gpu_ptr = nullptr;
    size_t size = 1ULL<<30; 
    cudaMalloc(&gpu_ptr, size);
    cudaMemset(gpu_ptr, 1, size);

    auto provider = vlcc_states::create_rawptr_provider("GPUProvider", vlcc_states::TIER_TYPES::GPU_TIER, gpu_ptr, size);

    if (provider) {
        std::cout << "Provider created successfully: " << provider->get_name() << std::endl;
        auto chunk = provider->get_next_chunk(vlcc_states::TIER_TYPES::GPU_TIER);
        if (chunk) {
            std::cout << "Chunk size: " << chunk->size << " bytes" << std::endl;
            provider->release();
        } else {
            std::cout << "No chunk available." << std::endl;
        }
    } else {
        std::cerr << "Failed to create provider." << std::endl;
    }

    return 0;
}