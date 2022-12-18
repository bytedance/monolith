load("@rules_cc//cc:defs.bzl", "cc_library", "cc_proto_library", "cc_test")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

package(default_visibility = ["//monolith/native_training/runtime/hash_table:__subpackages__"])

cc_library(
    name = "retriever_interface",
    hdrs = ["retriever_interface.h"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "retriever_base",
    hdrs = ["retriever_base.h"],
    deps = [
        ":retriever_interface",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "raw_retriever",
    srcs = ["raw_retriever.cc"],
    hdrs = ["raw_retriever.h"],
    deps = [
        ":retriever_base",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_test(
    name = "raw_retriever_test",
    srcs = ["raw_retriever_test.cc"],
    deps = [
        ":raw_retriever",
        "@com_google_absl//absl/random",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "fake_quant_retriever",
    srcs = ["fake_quant_retriever.cc"],
    hdrs = ["fake_quant_retriever.h"],
    deps = [
        ":retriever_base",
        "//monolith/native_training/runtime/hash_table/compressor:fake_quantizer",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_test(
    name = "fake_quant_retriever_test",
    srcs = ["fake_quant_retriever_test.cc"],
    deps = [
        ":fake_quant_retriever",
        "@com_google_absl//absl/random",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "retriever_combination",
    srcs = ["retriever_combination.cc"],
    hdrs = ["retriever_combination.h"],
    deps = [
        ":retriever_base",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_test(
    name = "retriever_combination_test",
    srcs = ["retriever_combination_test.cc"],
    deps = [
        ":raw_retriever",
        ":fake_quant_retriever",
        ":retriever_combination",
        "@com_google_absl//absl/random",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "hash_net_retriever",
    srcs = ["hash_net_retriever.cc"],
    hdrs = ["hash_net_retriever.h"],
    deps = [
        ":retriever_base",
        "//monolith/native_training/runtime/hash_table/compressor:hash_net_quantizer",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_test(
    name = "hash_net_retriever_test",
    srcs = ["hash_net_retriever_test.cc"],
    deps = [
        ":hash_net_retriever",
        "@com_google_absl//absl/random",
        "@com_google_googletest//:gtest_main",
    ],
)
