load("@pip_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_grpc_library", "py_proto_library")

package(
    default_visibility = ["//visibility:public"],
)

proto_library(
    name = "service_proto",
    srcs = ["service.proto"],
)

py_proto_library(
    name = "service_py_proto",
    deps = [":service_proto"],
)

py_grpc_library(
    name = "service_py_grpc",
    srcs = [":service_proto"],
    deps = [":service_py_proto"],
)

py_library(
    name = "constants",
    srcs = ["constants.py"],
)

py_library(
    name = "server_lib",
    srcs = ["server_lib.py"],
    deps = [
        ":constants",
        ":service_py_grpc",
        "//monolith/native_training:barrier_ops",
        "//monolith/native_training:net_utils",
        "//monolith/native_training:save_utils",
        requirement("grpcio"),
    ],
)

py_library(
    name = "client_lib",
    srcs = ["client_lib.py"],
    deps = [
        ":constants",
        ":service_py_grpc",
        requirement("grpcio"),
    ],
)

py_test(
    name = "server_lib_test",
    srcs = ["server_lib_test.py"],
    deps = [
        ":client_lib",
        ":server_lib",
    ],
)
