load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//monolith/native_training/runtime:__subpackages__"])

cc_library(
    name = "types",
    hdrs = ["types.h"],
)

cc_library(
    name = "filter",
    hdrs = ["filter.h"],
    deps = [
        ":types",
        "//monolith/native_training/runtime/hash_table:embedding_hash_table_cc_proto",
        "//monolith/native_training/runtime/hash_table:embedding_hash_table_factory",
    ],
)

cc_library(
    name = "hash_filter",
    hdrs = ["hash_filter.h"],
    srcs = ["hash_filter.cc"],
    deps = [
        ":filter",
        ":types",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/types:span",
        "@com_google_glog//:glog",
    ],
)

cc_test(
    name = "hash_filter_test",
    srcs = ["hash_filter_test.cc"],
    deps = [
        ":hash_filter",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "sliding_hash_filter",
    srcs = ["sliding_hash_filter.cc"],
    hdrs = ["sliding_hash_filter.h"],
    deps = [
        ":hash_filter",
        "//monolith/native_training/runtime/hash_table:embedding_hash_table_cc_proto",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "dummy_hash_filter",
    hdrs = ["dummy_hash_filter.h"],
    deps = [
        ":hash_filter",
    ],
)

cc_test(
    name = "sliding_hash_filter_test",
    srcs = ["sliding_hash_filter_test.cc"],
    deps = [
        ":sliding_hash_filter",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "probabilistic_filter",
    hdrs = ["probabilistic_filter.h"],
    srcs = ["probabilistic_filter.cc"],
    deps = [
        ":filter",
        ":hash_filter",
        "//monolith/native_training/runtime/concurrency:xorshift",
    ],
)

cc_binary(
    name = "probabilistic_filter_test",
    srcs = ["probabilistic_filter_test.cc"],
    deps = [
        ":probabilistic_filter",
        "@com_google_googletest//:gtest_main",
    ],
)
