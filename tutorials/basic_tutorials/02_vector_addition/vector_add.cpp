#include "vector_add.hpp"
#include <cstddef>

void vector_add(float *first, float *second, float *result, int n)
{
    for(size_t idx = 0; idx < n; ++idx) {
        result[idx] = first[idx] + second[idx];
    }
}