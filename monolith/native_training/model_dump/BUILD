load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

py_proto_library(
    name = "monolith_model_py_proto",
    srcs = ["monolith_model.proto"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//monolith/native_training/runtime/hash_table/compressor:float_compressor_py_proto",
        "//monolith/native_training/runtime/hash_table/initializer:initializer_config_py_proto",
        "//monolith/native_training/runtime/hash_table/optimizer:optimizer_py_proto",
    ],
)

py_library(
    name = "graph_utils",
    srcs = ["graph_utils.py"],
    deps = [
        "//idl:line_id_py_proto",
        "//monolith/native_training:utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "dump_utils",
    srcs = ["dump_utils.py"],
    deps = [
        ":graph_utils",
        ":monolith_model_py_proto",
        "//monolith/native_training/data:feature_list",
        "//monolith/native_training/data:item_pool_hook",
        "//monolith/native_training/data:parsers_py",
    ],
)

