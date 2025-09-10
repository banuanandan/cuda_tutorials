#include "kernel.h"
#include <iostream>


__global__ void vector_add_kernel(float *first, float *second, float *result, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    result[idx] = first[idx] + second[idx];
  }
}