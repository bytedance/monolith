load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library", "py_proto_library")
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

cc_proto_library(
    name = "transform_config_cc_proto",
    srcs = ["transform_config.proto"],
)

py_proto_library(
    name = "transform_config_py_proto",
    srcs = ["transform_config.proto"],
    srcs_version = "PY2AND3",
)


cc_library(
    name = "transforms",
    srcs = ["cc/transforms.cc"],
    hdrs = ["cc/transforms.h"],
    deps = [
        ":transform_config_cc_proto",
        "//monolith/native_training/data/training_instance:instance_utils",
        "//monolith/native_training/data/kernels/internal:label_utils",
        "//monolith/native_training/runtime/common:linalg_utils",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)


py_library(
    name = "transforms_py",
    srcs = [
        "transforms.py",
    ],
    deps = [
        ":transform_config_py_proto",
    ],
)

py_test(
    name = "transforms_test",
    srcs = ["transforms_test.py"],
    deps = [
        ":transforms_py",
    ],
)


exports_files([
    "cc/transforms.cc",
])
