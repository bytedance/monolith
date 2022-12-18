load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//monolith/native_training/runtime:__subpackages__"])

cc_library(
    name = "block_allocator",
    srcs = ["block_allocator.cc"],
    hdrs = ["block_allocator.h"],
    deps = [
        "//monolith/native_training/runtime/concurrency:xorshift",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_glog//:glog",
    ],
)

cc_test(
    name = "block_allocator_test",
    srcs = ["block_allocator_test.cc"],
    deps = [
        ":block_allocator",
        "@com_google_googletest//:gtest_main",
    ],
)
