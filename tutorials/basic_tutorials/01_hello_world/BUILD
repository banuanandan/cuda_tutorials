load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])


cuda_library(
    name = "mycudalib",
    srcs = ["kernel.cc"],
    hdrs = ["kernel.h"],
    deps = [],
)

cc_binary(
    name = "main",
    srcs = ["main.cpp"],
    deps = [":mycudalib"],
)