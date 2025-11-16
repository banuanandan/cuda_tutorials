#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                \
  do {                                                                  \
    cudaError_t err = (expr);                                           \
    if (err != cudaSuccess) {                                           \
      fprintf(stderr, "CUDA Error Code  : %d\n     Error String: %s\n", \
              err, cudaGetErrorString(err));                            \
      exit(err);                                                        \
    }                                                                   \
  } while (0)


__global__ void image_filter_kernel(unsigned char* input_image,
                        unsigned char* output_image,
                        int width, int height);