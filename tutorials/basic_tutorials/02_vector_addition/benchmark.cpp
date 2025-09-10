
#include "vector_add.hpp"
#include "constants.hpp"

#include <cstddef>
#include <vector>

#include "benchmark/benchmark.h"

void benchmark_vector_addition(benchmark::State& state)
{
    std::vector<float> a(NUM_ELEMENTS);
    std::vector<float> b(NUM_ELEMENTS);
    std::vector<float> c(NUM_ELEMENTS);

    for (auto _ : state) {
        vector_add(a.data(), b.data(), c.data(), NUM_ELEMENTS);
    }
}

// Register the function as a benchmark
BENCHMARK(benchmark_vector_addition);

// Run the benchmark
BENCHMARK_MAIN();