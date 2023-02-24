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

cc_library(
    name = "linalg_utils",
    hdrs = ["linalg_utils.h"],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "linalg_utils_test",
    srcs = [
        "linalg_utils_test.cc",
    ],
    deps = [
        ":linalg_utils",
        "@com_google_glog//:glog",
        "@com_google_googletest//:gtest_main",
    ],
)

