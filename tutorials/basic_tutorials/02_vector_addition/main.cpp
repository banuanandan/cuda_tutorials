#include "kernel.h"
#include "vector_add.hpp"
#include "constants.hpp"

#include <iostream>
#include <vector>
    

int main(int argc, char **argv) {

    // Initialize input vectors on the host
    std::vector<float> some_vector_h(NUM_ELEMENTS);
    std::vector<float> some_other_vector_h(NUM_ELEMENTS);
    std::vector<float> result_vector_h(NUM_ELEMENTS);

    for (size_t idx = 0; idx < NUM_ELEMENTS; ++idx) {
        some_vector_h.at(idx) = static_cast<float>(idx * 0.5);
        some_other_vector_h.at(idx) = static_cast<float>(idx * 2.0);
    }

    // Copy input vectors to the device
    float *some_vector_d;
    float *some_other_vector_d;
    float *result_vector_d;

    cudaMalloc(&some_vector_d, NUM_ELEMENTS * sizeof(float));
    cudaMalloc(&some_other_vector_d, NUM_ELEMENTS * sizeof(float));
    cudaMalloc(&result_vector_d, NUM_ELEMENTS * sizeof(float));

    cudaMemcpy(some_vector_d, some_vector_h.data(), NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(some_other_vector_d, some_other_vector_h.data(), NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);

    // Perform vector addition on the GPU
    vector_add_kernel<<<BLOCK_SIZE, THREADS_PER_BLOCK>>>(some_vector_d, some_other_vector_d, result_vector_d, NUM_ELEMENTS);
    cudaMemcpy(result_vector_h.data(), result_vector_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(some_vector_d);
    cudaFree(some_other_vector_d);
    cudaFree(result_vector_d);

    // Verify the result
    for (size_t idx = 0; idx < NUM_ELEMENTS; ++idx) {
        bool result_match = (result_vector_h.at(idx) == some_vector_h.at(idx) + some_other_vector_h.at(idx));
        if (!result_match) {
            std::cerr << "Error at index " << idx << ": "
                      << result_vector_h.at(idx) << " != "
                      << some_vector_h.at(idx) + some_other_vector_h.at(idx)
                      << std::endl;
            return -1;
        } else {
            // Uncomment the following line to see all results

            // std::cout << result_vector_h.at(idx) << " == "
            //           << some_vector_h.at(idx) + some_other_vector_h.at(idx)
            //           << std::endl;
        }
    }

    // Perform vector addition on the CPU for comparison (optional)
    std::vector<float> cpu_result_vector_h(NUM_ELEMENTS);
    vector_add(some_vector_h.data(), some_other_vector_h.data(), cpu_result_vector_h.data(), NUM_ELEMENTS);

    std::cout << "Vector addition successful!" << std::endl;

    return 0;
}