load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_proto_library")

package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "session_hooks",
    srcs = ["session_hooks.py"],
)

py_test(
    name = "session_hooks_test",
    srcs = ["session_hooks_test.py"],
    deps = [
        ":session_hooks",
    ],
)

proto_library(
    name = "ckpt_hooks_proto",
    srcs = ["ckpt_hooks.proto"],
)

py_proto_library(
    name = "ckpt_hooks_py_proto",
    deps = [
        ":ckpt_hooks_proto",
    ],
)

py_library(
    name = "ckpt_hooks",
    srcs = ["ckpt_hooks.py"],
    deps = [
        ":ckpt_hooks_py_proto",
        "//monolith/native_training:barrier_ops",
        "//monolith/native_training:graph_meta",
    ],
)

py_test(
    name = "ckpt_hooks_test",
    srcs = ["ckpt_hooks_test.py"],
    deps = [
        ":ckpt_hooks",
        "//monolith/native_training:save_utils",
    ],
)

proto_library(
    name = "controller_hooks_proto",
    srcs = ["controller_hooks.proto"],
)

py_proto_library(
    name = "controller_hooks_py_proto",
    deps = [
        ":controller_hooks_proto",
    ],
)

py_library(
    name = "controller_hooks",
    srcs = ["controller_hooks.py"],
    deps = [
        ":controller_hooks_py_proto",
        "//monolith/native_training:barrier_ops",
        "//monolith/native_training:utils",
    ],
)

py_test(
    name = "controller_hooks_test",
    srcs = ["controller_hooks_test.py"],
    deps = [
        ":controller_hooks",
    ],
)

py_library(
    name = "ckpt_info",
    srcs = ["ckpt_info.py"],
    deps = [
        "//monolith/native_training:hash_table_ops",
        "//monolith/native_training:hash_table_utils",
        "//monolith/native_training:multi_hash_table_ops",
        "//monolith/native_training/proto:ckpt_info_py_proto",
    ],
)

py_test(
    name = "ckpt_info_test",
    srcs = ["ckpt_info_test.py"],
    deps = [
        ":ckpt_info",
    ],
)

py_library(
    name = "hook_utils",
    srcs = ["hook_utils.py"],
)

py_test(
    name = "hook_utils_test",
    srcs = ["hook_utils_test.py"],
    deps = [
        ":hook_utils",
    ],
)

py_library(
    name = "ps_check_hooks",
    srcs = ["ps_check_hooks.py"],
    deps = [
        "//monolith/native_training:barrier_ops",
        "//monolith/native_training:logging_ops",
        "//monolith/native_training:utils",
        "//monolith/native_training/runtime/ops:logging_ops_py_proto",
    ],
)

py_test(
    name = "ps_check_hooks_test",
    srcs = ["ps_check_hooks_test.py"],
    deps = [
        ":ps_check_hooks",
    ],
)

py_library(
    name = "feature_engineering_hooks",
    srcs = ["feature_engineering_hooks.py"],
    deps = [
        "//idl:example_py_proto",
    ],
)
