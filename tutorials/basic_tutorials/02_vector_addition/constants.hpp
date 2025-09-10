#include <cstddef>
#include <cmath>

constexpr size_t NUM_ELEMENTS = 100'000'000; // 1,000,000 elements
constexpr size_t THREADS_PER_BLOCK = 256;
constexpr size_t BLOCK_SIZE = ceil((float)NUM_ELEMENTS / THREADS_PER_BLOCK);