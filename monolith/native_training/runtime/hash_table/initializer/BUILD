load("@rules_cc//cc:defs.bzl", "cc_library", "cc_proto_library", "cc_test")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

package(default_visibility = ["//monolith/native_training/runtime/hash_table:__subpackages__"])

cc_library(
    name = "initializer_interface",
    hdrs = ["initializer_interface.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

proto_library(
    name = "initializer_config_proto",
    srcs = ["initializer_config.proto"],
)

cc_proto_library(
    name = "initializer_config_cc_proto",
    deps = [":initializer_config_proto"],
)

py_proto_library(
    name = "initializer_config_py_proto",
    srcs = ["initializer_config.proto"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "initializer_factory",
    srcs = ["initializer_factory.cc"],
    hdrs = ["initializer_factory.h"],
    deps = [
        "random_uniform_initializer",
        ":constants_initializer",
        ":initializer_config_cc_proto",
        ":initializer_interface",
    ],
)

cc_library(
    name = "constants_initializer",
    srcs = ["constants_initializer.cc"],
    hdrs = ["constants_initializer.h"],
    deps = [
        ":initializer_config_cc_proto",
        ":initializer_interface",
    ],
)

cc_library(
    name = "random_uniform_initializer",
    srcs = ["random_uniform_initializer.cc"],
    hdrs = ["random_uniform_initializer.h"],
    deps = [
        ":initializer_config_cc_proto",
        ":initializer_interface",
    ],
)

cc_test(
    name = "random_uniform_initializer_test",
    srcs = ["random_uniform_initializer_test.cc"],
    deps = [
        ":initializer_config_cc_proto",
        ":initializer_interface",
        ":random_uniform_initializer",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "initializer_combination",
    srcs = ["initializer_combination.cc"],
    hdrs = ["initializer_combination.h"],
    deps = [
        ":initializer_interface",
    ],
)

cc_test(
    name = "initializer_combination_test",
    srcs = ["initializer_combination_test.cc"],
    deps = [
        ":constants_initializer",
        ":initializer_combination",
        ":initializer_interface",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
