load("@rules_cc//cc:defs.bzl", "cc_library", "cc_proto_library", "cc_test")
load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_test")
load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")

package(default_visibility = ["//monolith/native_training/runtime:__subpackages__"])

cc_library(
    name = "sync_client_interface",
    hdrs = ["sync_client_interface.h"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
    ],
)

cc_library(
    name = "dummy_sync_client",
    hdrs = ["dummy_sync_client.h"],
    deps = [
        ":sync_client_interface",
    ],
)

proto_library(
    name = "parameter_sync_proto",
    srcs = ["parameter_sync.proto"],
)

cc_proto_library(
    name = "parameter_sync_cc_proto",
    deps = [":parameter_sync_proto"],
)

py_proto_library(
    name = "parameter_sync_py_proto",
    srcs = ["parameter_sync.proto"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [],
)

cc_grpc_library(
    name = "parameter_sync_cc_grpc",
    srcs = [":parameter_sync_proto"],
    grpc_only = True,
    deps = [":parameter_sync_cc_proto"],
)

cc_library(
    name = "parameter_sync_client",
    srcs = ["parameter_sync_client.cc"],
    hdrs = ["parameter_sync_client.h"],
    deps = [
        ":parameter_sync_cc_grpc",
        ":sync_client_interface",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_glog//:glog",
        "@org_tensorflow//tensorflow/core/platform:logging",
        "@org_tensorflow_serving//tensorflow_serving/apis:prediction_service_proto",
    ],
)

tf_cc_test(
    name = "parameter_sync_client_test",
    srcs = ["parameter_sync_client_test.cc"],
    deps = [
        ":parameter_sync_client",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "dummy_sync_server",
    srcs = ["dummy_sync_server.cc"],
    hdrs = ["dummy_sync_server.h"],
    deps = [
        ":parameter_sync_cc_grpc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_github_grpc_grpc//:grpc++_reflection",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_glog//:glog",
        "@org_tensorflow_serving//tensorflow_serving/apis:prediction_service_proto",
    ],
)

cc_library(
    name = "request_splitter",
    srcs = ["request_splitter.cc"],
    hdrs = ["request_splitter.h"],
    deps = [
        ":parameter_sync_cc_grpc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_github_grpc_grpc//:grpc++_reflection",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_glog//:glog",
    ],
)

cc_test(
    name = "request_splitter_test",
    srcs = ["request_splitter_test.cc"],
    deps = [
        ":request_splitter",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "sync_client_manager",
    srcs = ["sync_client_manager.cc"],
    hdrs = ["sync_client_manager.h"],
    deps = [
        ":parameter_sync_cc_grpc",
        ":sync_client_interface",
        "//monolith/native_training/runtime/common:metrics",
        "//monolith/native_training/runtime/parameter_sync:parameter_sync_client",
        "//monolith/native_training/runtime/parameter_sync:request_splitter",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_glog//:glog",
        "@org_tensorflow//tensorflow/core/platform:logging",
        "@org_tensorflow_serving//tensorflow_serving/apis:prediction_service_proto",
    ],
)
