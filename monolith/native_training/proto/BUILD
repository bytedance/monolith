load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_grpc_library", "py_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "primus_am_service_proto",
    srcs = ["primus_am_service.proto"],
    deps = ["@com_google_protobuf//:wrappers_proto"],
)

py_proto_library(
    name = "primus_am_service_py_proto",
    deps = [
        ":primus_am_service_proto",
    ],
)

py_grpc_library(
    name = "primus_am_service_py_proto_grpc",
    srcs = ["primus_am_service_proto"],
    deps = [":primus_am_service_py_proto"],
)

proto_library(
    name = "debugging_info_proto",
    srcs = ["debugging_info.proto"],
)

py_proto_library(
    name = "debugging_info_py_proto",
    deps = [
        ":debugging_info_proto",
    ],
)

proto_library(
    name = "ckpt_info_proto",
    srcs = ["ckpt_info.proto"],
)

py_proto_library(
    name = "ckpt_info_py_proto",
    deps = [
        ":ckpt_info_proto",
    ],
)
