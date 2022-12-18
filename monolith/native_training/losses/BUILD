load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "batch_softmax_loss",
    srcs = ["batch_softmax_loss.py"],
    srcs_version = "PY3",
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "batch_softmax_loss_test",
    srcs = ["batch_softmax_loss_test.py"],
    srcs_version = "PY3",
    deps = [
        ":batch_softmax_loss",
    ],
)

py_binary(
    name = "inbatch_auc_loss",
    srcs = ["inbatch_auc_loss.py"],
    deps = [
        "//monolith/native_training/runtime/ops:gen_monolith_ops",
    ],
)

py_test(
    name = "inbatch_auc_loss_test",
    srcs = ["inbatch_auc_loss_test.py"],
    deps = [
        ":inbatch_auc_loss",
    ],
)

py_binary(
    name = "ltr_losses",
    srcs = ["ltr_losses.py"],
    deps = [
        "//monolith:utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "losses",
    srcs = [],
    srcs_version = "PY3",
    deps = [
        ":batch_softmax_loss",
        ":inbatch_auc_loss",
        ":ltr_losses",
    ],
)
