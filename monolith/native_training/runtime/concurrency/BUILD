load("@rules_cc//cc:defs.bzl", "cc_library", "cc_proto_library", "cc_test")

package(default_visibility = ["//monolith/native_training/runtime:__subpackages__"])

cc_library(
    name = "thread_pool",
    srcs = ["thread_pool.cc"],
    hdrs = ["thread_pool.h"],
    deps = [
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "queue",
    hdrs = ["queue.h"],
    deps = [],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "queue_test",
    srcs = ["queue_test.cc"],
    deps = [
        ":queue",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "sleeper",
    hdrs = ["sleeper.h"],
    deps = [],
)

cc_library(
    name = "micro_one_bit_spin_lock",
    hdrs = ["micro_one_bit_spin_lock.h"],
    deps = [
        ":sleeper",
        "@com_google_glog//:glog"
    ],
)

cc_library(
    name = "xorshift",
    hdrs = ["xorshift.h"],
    srcs = ["xorshift.cc"],
    deps = [],
)

cc_binary(
    name = "xorshift_test",
    srcs = ["xorshift_test.cc"],
    deps = [
        ":xorshift",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "random_number_generator_benchmark",
    srcs = ["random_number_generator_benchmark.cc"],
    deps = [
        ":xorshift",
        "@com_google_absl//absl/random",
        "@com_github_google_benchmark//:benchmark",
    ],
)
