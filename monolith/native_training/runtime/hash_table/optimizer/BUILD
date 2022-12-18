load("@rules_cc//cc:defs.bzl", "cc_library", "cc_proto_library", "cc_test")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

package(default_visibility = ["//monolith/native_training/runtime:__subpackages__"])

proto_library(
    name = "optimizer_proto",
    srcs = ["optimizer.proto"],
)

cc_proto_library(
    name = "optimizer_cc_proto",
    deps = [":optimizer_proto"],
)

py_proto_library(
    name = "optimizer_py_proto",
    srcs = ["optimizer.proto"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "optimizer_interface",
    hdrs = ["optimizer_interface.h"],
    deps = [
        ":optimizer_cc_proto",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "optimizer_decorator",
    hdrs = ["optimizer_decorator.h"],
    deps = [
        ":optimizer_cc_proto",
        ":optimizer_interface",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "stochastic_rounding",
    srcs = ["stochastic_rounding.cc"],
    hdrs = ["stochastic_rounding.h"],
    deps = [
        ":optimizer_cc_proto",
        ":optimizer_interface",
        "//third_party/half_sourceforge_net:half",
    ],
)

cc_test(
    name = "stochastic_rounding_test",
    srcs = ["stochastic_rounding_test.cc"],
    deps = [
        ":optimizer_cc_proto",
        ":optimizer_factory",
        ":stochastic_rounding",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "optimizer_factory",
    srcs = ["optimizer_factory.cc"],
    hdrs = ["optimizer_factory.h"],
    deps = [
        ":adadelta_optimizer",
        ":adagrad_optimizer",
        ":adam_optimizer",
        ":amsgrad_optimizer",
        ":batch_softmax_optimizer",
        ":dynamic_wd_adagrad_optimizer",
        ":ftrl_optimizer",
        ":group_ftrl_optimizer",
        ":momentum_optimizer",
        ":moving_average_optimizer",
        ":rmsprop_optimizer",
        ":sgd_optimizer",
        ":stochastic_rounding",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "adagrad_optimizer_internal_deps",
)

cc_library(
    name = "adagrad_optimizer",
    srcs = ["adagrad_optimizer.cc"],
    hdrs = ["adagrad_optimizer.h"],
    copts = [
        "-D_ENABLE_AVX",
        "-mavx",
        "-mfma",
    ],
    deps = [
        ":adagrad_optimizer_internal_deps",
        ":avx_utils",
        ":optimizer_cc_proto",
        ":optimizer_interface",
    ],
)

cc_library(
    name = "batch_softmax_optimizer",
    srcs = ["batch_softmax_optimizer.cc"],
    hdrs = ["batch_softmax_optimizer.h"],
    deps = [
        ":optimizer_cc_proto",
        ":optimizer_interface",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "dynamic_wd_adagrad_optimizer_internal_deps",
)

cc_library(
    name = "dynamic_wd_adagrad_optimizer",
    srcs = ["dynamic_wd_adagrad_optimizer.cc"],
    hdrs = ["dynamic_wd_adagrad_optimizer.h"],
    copts = [
        "-D_ENABLE_AVX",
        "-mavx",
        "-mfma",
    ],
    deps = [
        ":dynamic_wd_adagrad_optimizer_internal_deps",
        ":dynamic_wd_avx_utils",
        ":optimizer_cc_proto",
        ":optimizer_interface",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "ftrl_optimizer",
    srcs = ["ftrl_optimizer.cc"],
    hdrs = ["ftrl_optimizer.h"],
    deps = [
        ":optimizer_cc_proto",
        ":optimizer_interface",
    ],
)

cc_test(
    name = "ftrl_optimizer_test",
    srcs = ["ftrl_optimizer_test.cc"],
    deps = [
        ":ftrl_optimizer",
        ":optimizer_cc_proto",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "adagrad_optimizer_test",
    srcs = ["adagrad_optimizer_test.cc"],
    deps = [
        ":adagrad_optimizer",
        ":avx_utils",
        ":optimizer_cc_proto",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "batch_softmax_optimizer_test",
    srcs = ["batch_softmax_optimizer_test.cc"],
    deps = [
        ":batch_softmax_optimizer",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "dynamic_wd_adagrad_optimizer_test",
    srcs = ["dynamic_wd_adagrad_optimizer_test.cc"],
    deps = [
        ":dynamic_wd_adagrad_optimizer",
        ":dynamic_wd_avx_utils",
        ":optimizer_cc_proto",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "sgd_optimizer",
    srcs = ["sgd_optimizer.cc"],
    hdrs = ["sgd_optimizer.h"],
    deps = [
        ":optimizer_cc_proto",
        ":optimizer_interface",
    ],
)

cc_test(
    name = "sgd_optimizer_test",
    srcs = ["sgd_optimizer_test.cc"],
    deps = [
        ":optimizer_cc_proto",
        ":sgd_optimizer",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "adadelta_optimizer",
    srcs = ["adadelta_optimizer.cc"],
    hdrs = ["adadelta_optimizer.h"],
    deps = [
        ":optimizer_cc_proto",
        ":optimizer_interface",
    ],
)

cc_test(
    name = "adadelta_optimizer_test",
    srcs = ["adadelta_optimizer_test.cc"],
    deps = [
        ":adadelta_optimizer",
        ":optimizer_cc_proto",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "adam_optimizer",
    srcs = ["adam_optimizer.cc"],
    hdrs = ["adam_optimizer.h"],
    deps = [
        ":optimizer_cc_proto",
        ":optimizer_interface",
    ],
)

cc_test(
    name = "adam_optimizer_test",
    srcs = ["adam_optimizer_test.cc"],
    deps = [
        ":adam_optimizer",
        ":optimizer_cc_proto",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "amsgrad_optimizer",
    srcs = ["amsgrad_optimizer.cc"],
    hdrs = ["amsgrad_optimizer.h"],
    deps = [
        ":optimizer_cc_proto",
        ":optimizer_interface",
    ],
)

cc_test(
    name = "amsgrad_optimizer_test",
    srcs = ["amsgrad_optimizer_test.cc"],
    deps = [
        ":amsgrad_optimizer",
        ":optimizer_cc_proto",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "momentum_optimizer",
    srcs = ["momentum_optimizer.cc"],
    hdrs = ["momentum_optimizer.h"],
    deps = [
        ":optimizer_cc_proto",
        ":optimizer_interface",
    ],
)

cc_test(
    name = "momentum_optimizer_test",
    srcs = ["momentum_optimizer_test.cc"],
    deps = [
        ":momentum_optimizer",
        ":optimizer_cc_proto",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "moving_average_optimizer",
    srcs = ["moving_average_optimizer.cc"],
    hdrs = ["moving_average_optimizer.h"],
    deps = [
        ":optimizer_cc_proto",
        ":optimizer_interface",
    ],
)

cc_test(
    name = "moving_average_optimizer_test",
    srcs = ["moving_average_optimizer_test.cc"],
    deps = [
        ":moving_average_optimizer",
        ":optimizer_cc_proto",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "rmsprop_optimizer",
    srcs = ["rmsprop_optimizer.cc"],
    hdrs = ["rmsprop_optimizer.h"],
    deps = [
        ":optimizer_cc_proto",
        ":optimizer_interface",
    ],
)

cc_test(
    name = "rmsprop_optimizer_test",
    srcs = ["rmsprop_optimizer_test.cc"],
    deps = [
        ":optimizer_cc_proto",
        ":rmsprop_optimizer",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "dc_optimizer",
    srcs = ["dc_optimizer.cc"],
    hdrs = ["dc_optimizer.h"],
    deps = [
        ":optimizer_cc_proto",
        ":optimizer_decorator",
        ":optimizer_interface",
    ],
)

cc_test(
    name = "dc_optimizer_test",
    srcs = ["dc_optimizer_test.cc"],
    deps = [
        ":adadelta_optimizer",
        ":dc_optimizer",
        ":optimizer_cc_proto",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "optimizer_combination",
    srcs = ["optimizer_combination.cc"],
    hdrs = ["optimizer_combination.h"],
    deps = [
        ":optimizer_interface",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_test(
    name = "optimizer_combination_test",
    srcs = ["optimizer_combination_test.cc"],
    deps = [
        ":adagrad_optimizer",
        ":optimizer_cc_proto",
        ":optimizer_combination",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "test_utils",
    testonly = 1,
    hdrs = ["test_utils.h"],
    deps = [
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "dynamic_wd_avx_utils",
    hdrs = ["dynamic_wd_avx_utils.h"],
    copts = [
        "-D_ENABLE_AVX",
        "-mavx",
        "-mfma",
    ],
    deps = [
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "dynamic_wd_avx_test",
    testonly = 1,
    srcs = ["dynamic_wd_avx_test.cc"],
    copts = [
        "-D_ENABLE_AVX",
        "-mavx",
        "-mfma",
    ],
    deps = [
        ":dynamic_wd_avx_utils",
        "@com_google_absl//absl/random",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "group_ftrl_optimizer",
    srcs = ["group_ftrl_optimizer.cc"],
    hdrs = ["group_ftrl_optimizer.h"],
    deps = [
        ":avx_utils",
        ":optimizer_cc_proto",
        ":optimizer_interface",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "group_ftrl_optimizer_test",
    srcs = ["group_ftrl_optimizer_test.cc"],
    deps = [
        ":avx_utils",
        ":group_ftrl_optimizer",
        ":optimizer_cc_proto",
        ":test_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "avx_utils",
    hdrs = ["avx_utils.h"],
    copts = [
        "-D_ENABLE_AVX",
        "-mavx",
        "-mfma",
    ],
    deps = [
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "avx_test",
    testonly = 1,
    srcs = ["avx_test.cc"],
    copts = [
        "-D_ENABLE_AVX",
        "-mavx",
        "-mfma",
    ],
    deps = [
        ":avx_utils",
        "@com_google_absl//absl/random",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "avx_benchmark",
    testonly = 1,
    srcs = ["avx_benchmark.cc"],
    copts = [
        "-D_ENABLE_AVX",
        "-mavx",
        "-mfma",
    ],
    deps = [
        ":avx_utils",
        "//monolith/native_training/runtime/allocator:block_allocator",
        "//monolith/native_training/runtime/common:cpu_info",
        "@com_github_google_benchmark//:benchmark",
        "@com_google_absl//absl/random",
        "@com_google_glog//:glog",
        "@com_google_googletest//:gtest_main",
    ],
)
