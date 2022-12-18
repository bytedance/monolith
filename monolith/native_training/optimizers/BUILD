load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_custom_op_library", "tf_kernel_library")

package(
    default_visibility = ["//visibility:public"],
)


tf_kernel_library(
    name = "training_ops",
    srcs = [
        "cc/kernels/training_op_helpers.h",
        "cc/kernels/training_ops.h",
        "cc/kernels/training_ops.cc",
        "cc/training_ops.cc",
    ],
    gpu_srcs = [
        "cc/kernels/training_op_helpers.h",
        "cc/kernels/training_ops.h",
        "cc/kernels/training_ops_gpu.cu.cc",
    ],
    deps = [
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core/kernels:gpu_device_array_for_custom_op",
    ],
    alwayslink = 1,
)

py_library(
    name = "adamom",
    srcs = ["adamom.py"],
    deps = [
        "//monolith/native_training/runtime/ops:gen_monolith_ops",
    ],
)

py_test(
    name = "adamom_test",
    srcs = ["adamom_test.py"],
    deps = [
        ":adamom",
    ],
)

py_library(
    name = "shampoo",
    srcs = ["shampoo.py"],
    deps = [
        "//monolith:utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "rmsprop",
    srcs = ["rmsprop.py"],
    deps = [
        "//monolith/native_training/runtime/ops:gen_monolith_ops",
    ],
)

py_test(
    name = "rmsprop_test",
    srcs = ["rmsprop_test.py"],
    deps = [
        ":rmsprop",
    ],
)

py_test(
    name = "rmspropv2_test",
    srcs = ["rmspropv2_test.py"],
    deps = [
        ":rmsprop",
    ],
)
