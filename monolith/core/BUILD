load("@pip_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "base_embedding_task",
    srcs = ["base_embedding_task.py"],
    srcs_version = "PY3",
    deps = [
        ":auto_checkpoint_feed_hook",
        ":base_embedding_host_call",
        ":base_task",
        ":feature",
        ":util",
    ],
)

py_library(
    name = "base_layer",
    srcs = ["base_layer.py"],
    srcs_version = "PY3",
    deps = [
        ":hyperparams",
        ":py_utils",
    ],
)

py_library(
    name = "base_host_call",
    srcs = ["base_host_call.py"],
    srcs_version = "PY3",
    deps = [
    ],
)

py_library(
    name = "base_embedding_host_call",
    srcs = ["base_embedding_host_call.py"],
    srcs_version = "PY3",
    deps = [
        ":base_host_call",
        ":tpu_variable",
    ],
)

py_test(
    name = "base_embedding_host_call_test",
    srcs = ["base_embedding_host_call_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":base_embedding_host_call",
    ],
)

py_library(
    name = "host_call",
    srcs = ["host_call.py"],
    srcs_version = "PY3",
)

py_library(
    name = "mixed_emb_op_comb_nws",
    srcs = ["mixed_emb_op_comb_nws.py"],
    srcs_version = "PY3",
)

py_test(
    name = "base_layer_test",
    srcs = ["base_layer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":base_layer",
    ],
)

py_library(
    name = "base_model_params",
    srcs = ["base_model_params.py"],
    srcs_version = "PY3",
)

py_library(
    name = "base_task",
    srcs = ["base_task.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":hyperparams",
    ],
)

py_test(
    name = "core_test_suite",
    srcs = ["core_test_suite.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":base_embedding_host_call_test",
        ":base_layer_test",
        ":hyperparams_test",
        ":util_test",
    ],
)

py_library(
    name = "dense",
    srcs = ["dense.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":variance_scaling",
    ],
)

py_test(
    name = "dense_test",
    srcs = ["dense_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dense",
        ":testing_utils",
    ],
)

py_library(
    name = "feature",
    srcs = ["feature.py"],
    srcs_version = "PY3",
)

py_test(
    name = "hyperparams_test",
    srcs = ["hyperparams_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":hyperparams",
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    srcs_version = "PY3",
    deps = [
        ":feature",
    ],
)

py_library(
    name = "hyperparams",
    srcs = ["hyperparams.py"],
    srcs_version = "PY3",
)

py_library(
    name = "model_imports_no_params",
    srcs = ["model_imports.py"],
    srcs_version = "PY3",
)

py_library(
    name = "model_imports",
    srcs_version = "PY3",
    deps = [
        ":model_imports_no_params",
    ],
)

py_library(
    name = "model_registry",
    srcs = ["model_registry.py"],
    srcs_version = "PY3",
    deps = [
        ":base_model_params",
        ":model_imports_no_params",
    ],
)

py_library(
    name = "optimizers",
    srcs = ["optimizers.py"],
    srcs_version = "PY3",
)

py_library(
    name = "py_utils",
    srcs = ["py_utils.py"],
    srcs_version = "PY3",
    deps = [],
)

py_library(
    name = "tpu_variable",
    srcs = ["tpu_variable.py"],
    srcs_version = "PY3",
    deps = [],
)

py_library(
    name = "testing_utils",
    srcs = ["testing_utils.py"],
    srcs_version = "PY3",
)

py_library(
    name = "util",
    srcs = ["util.py"],
    srcs_version = "PY3",
    deps = [
        requirement("google-cloud-storage"),
    ],
)

py_library(
    name = "variance_scaling",
    srcs = ["variance_scaling.py"],
    srcs_version = "PY3",
)


py_library(
    name = "auto_checkpoint_feed_hook",
    srcs = ["auto_checkpoint_feed_hook.py"],
    srcs_version = "PY3",
)

py_test(
    name = "feature_test",
    srcs = ["feature_test.py"],
    srcs_version = "PY3",
    deps = [
        "feature",
        "hyperparams",
    ],
)

py_library(
    name = "util_test",
    srcs = ["util_test.py"],
    srcs_version = "PY3",
    deps = [":util"],
)
