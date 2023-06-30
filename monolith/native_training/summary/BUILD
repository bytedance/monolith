load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test", "tf_kernel_library")
load("@pip_deps//:requirements.bzl", "requirement")

package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
    deps = [requirement('tensorboard')],
)


py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":utils",
    ],
)


py_library(
    name = "summary_ops",
    srcs = ["summary_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":utils",
        "//monolith/native_training/layers:layer_ops",
    ]
)


py_test(
    name = "summary_ops_test",
    srcs = ["summary_ops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":summary_ops",
    ],
)
