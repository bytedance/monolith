load("@rules_cc//cc:defs.bzl", "cc_library", "cc_proto_library", "cc_test")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

package(default_visibility = ["//monolith/native_training/runtime/hash_table:__subpackages__"])

proto_library(
    name = "float_compressor_proto",
    srcs = ["float_compressor.proto"],
)

cc_proto_library(
    name = "float_compressor_cc_proto",
    deps = [
        ":float_compressor_proto",
    ],
)

py_proto_library(
    name = "float_compressor_py_proto",
    srcs = ["float_compressor.proto"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "float_compressor",
    srcs = ["float_compressor.cc"],
    hdrs = ["float_compressor.h"],
    defines = ["HALF_ENABLE_F16C_INTRINSICS=0"],
    deps = [
        ":float_compressor_cc_proto",
        ":fake_quantizer",
        ":hash_net_quantizer",
        "//third_party/half_sourceforge_net:half",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "float_compressor_test",
    srcs = ["float_compressor_test.cc"],
    deps = [
        ":float_compressor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "fake_quantizer",
    hdrs = ["fake_quantizer.h"],
    deps = [],
)

cc_test(
    name = "fake_quantizer_test",
    srcs = ["fake_quantizer_test.cc"],
    deps = [
        ":fake_quantizer",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "hash_net_quantizer",
    hdrs = ["hash_net_quantizer.h"],
    deps = [
        ":float_compressor_cc_proto",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_glog//:glog",
        "@org_tensorflow//tensorflow/core/platform:logging",
    ],
)

cc_test(
    name = "hash_net_quantizer_test",
    srcs = ["hash_net_quantizer_test.cc"],
    deps = [
        ":hash_net_quantizer",
        "@com_google_googletest//:gtest_main",
    ],
)
