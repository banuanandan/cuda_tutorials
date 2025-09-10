load("@rules_cuda//cuda:defs.bzl", "cuda_library")
load("@rules_cuda//cuda:defs.bzl", "cuda_binary")

package(default_visibility = ["//visibility:public"])

cuda_library(
    name = "vector_addition_gpu",
    srcs = ["kernel.cc"],
    hdrs = ["kernel.h"],
    deps = [],
)

cc_library(
    name = "vector_addition_cpu",
    srcs = ["vector_add.cpp"],
    hdrs = [
        "vector_add.hpp",
        "constants.hpp",
    ],
)

cuda_binary(
    name = "vector_addition",
    srcs = ["main.cpp"],
    deps = [
        ":vector_addition_gpu",
        ":vector_addition_cpu",
    ],
)

cc_binary(
    name = "benchmark",
    srcs = ["benchmark.cpp"],
    deps = [
        ":vector_addition_cpu",
        "@google_benchmark//:benchmark",
    ],
)