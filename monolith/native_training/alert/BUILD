load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

package(default_visibility = ["//monolith/native_training/alert:__subpackages__"])

# Including all public interfaces.
py_library(
    name = "alert",
    visibility = ["//visibility:public"],
    deps = [
        ":alert_manager",
        ":alert_py_proto",
    ],
)

py_proto_library(
    name = "alert_py_proto",
    srcs = ["alert.proto"],
)

py_library(
    name = "alert_manager_internal_deps",
)

py_library(
    name = "alert_manager",
    srcs = ["alert_manager.py"],
    deps = [
        ":alert_manager_internal_deps",
        ":alert_py_proto",
    ],
)

py_test(
    name = "alert_manager_test",
    srcs = ["alert_manager_test.py"],
    deps = [
        ":alert_manager",
    ],
)
