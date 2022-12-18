load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@pip_deps//:requirements.bzl", "requirement")

package(
    default_visibility = [
        "//monolith/native_training:__subpackages__",
        "//monolith/sail:__subpackages__",
    ],
)

py_library(
    name = "deep_insight_ops",
    srcs = ["deep_insight_ops.py"],
    deps = [
        "//monolith/native_training/runtime/ops:gen_monolith_ops",
    ],
)

py_test(
    name = "deep_insight_ops_test",
    srcs = ["deep_insight_ops_test.py"],
    deps = [":deep_insight_ops"],
)

py_library(
    name = "cli",
    srcs = ["cli.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "exit_hook",
    srcs = ["exit_hook.py"],
    deps = [
        ":cli",
        "//monolith/native_training:native_task_context",
    ],
)

py_library(
    name = "metric_hook",
    srcs = [
        "kafka_utils.py",
        "metric_hook.py",
    ],
    deps = [
        ":cli",
        ":exit_hook",
        "//monolith/native_training:utils",
        "//monolith/native_training/alert",
        "@org_tensorflow//tensorflow:tensorflow_py",
        requirement("kafka-python"),
    ],
)

py_test(
    name = "metric_hook_test",
    srcs = ["metric_hook_test.py"],
    deps = [
        ":metric_hook",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        ":deep_insight_ops",
    ],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    deps = [
        ":utils",
    ],
)
