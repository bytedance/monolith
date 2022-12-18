load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_custom_op_library")

package(default_visibility = ["//monolith/native_training:__subpackages__"])

cc_library(
    name = "deep_insight_internal_deps",
)

cc_library(
    name = "deep_insight",
    srcs = ["deep_insight.cc"],
    hdrs = ["deep_insight.h"],
    deps = [
        ":deep_insight_internal_deps",
        "//monolith/native_training/runtime/common:metrics",
        "//third_party/nlohmann:json",
        "@com_google_glog//:glog",
    ],
)

cc_test(
    name = "deep_insight_test",
    srcs = ["deep_insight_test.cc"],
    deps = [
        ":deep_insight",
        "//monolith/native_training/runtime/common:metrics",
        "@com_google_glog//:glog",
        "@com_google_googletest//:gtest_main",
    ],
)
