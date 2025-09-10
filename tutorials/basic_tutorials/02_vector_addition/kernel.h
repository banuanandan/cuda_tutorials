#include <cuda_runtime.h>

__global__ void vector_add_kernel(float *first, float *second, float *result, int n);