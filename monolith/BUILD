load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//visibility:public"])


py_library(
    name = "path_utils",
    srcs = ["path_utils.py"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        ":path_utils",
    ],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    deps = [
        ":utils",
    ],
)

py_library(
    name = "init",
    srcs = ["__init__.py"],
    deps = [
        "//monolith/native_training:entry",
        "//monolith/native_training:estimator",
        "//monolith/native_training:native_model",
        "//monolith/native_training/data",
        "//monolith/native_training/layers",
        "//monolith/native_training/model_export",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
