load("@rules_cuda//cuda:defs.bzl", "cuda_library", "cuda_binary")
load("@opencv//:opencv.bzl", "opencv_module")

package(default_visibility = ["//visibility:public"])

cuda_library(
    name = "mycudalib",
    srcs = ["kernel.cc"],
    hdrs = ["kernel.h"],
    deps = [],
)

cuda_binary(
    name = "image_filter",
    srcs = ["main.cpp"],
    deps = [
        "@opencv",
        ":mycudalib",
    ],
)