load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//monolith/native_training/runtime:__subpackages__"])

cc_library(
    name = "cpu_info",
    srcs = ["cpu_info.cc"],
    hdrs = ["cpu_info.h"],
    deps = [
    ],
)

cc_library(
    name = "metrics_internal_deps",
)

cc_library(
    name = "metrics",
    srcs = ["metrics.cc"],
    hdrs = ["metrics.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":metrics_internal_deps",
        "@com_google_glog//:glog",
    ],
)

