load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test", "tf_custom_op_library")

package(default_visibility = ["//visibility:public"])


cc_library(
    name = "reader_util",
    srcs = ["cc/reader_util.cc"],
    hdrs = ["cc/reader_util.h"],
    deps = [
        "//third_party/nlohmann:json",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core/kernels:ops_util_hdrs",
        "@org_tensorflow//tensorflow/core/platform:logging",
    ],
)

cc_test(
    name = "reader_util_test",
    srcs = ["cc/reader_util_test.cc"],
    deps = [
        ":reader_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "cached_mem_pool",
    srcs = ["cc/cached_mem_pool.cc"],
    hdrs = ["cc/cached_mem_pool.h"],
    deps = [
        "@com_google_glog//:glog",
    ],
)

cc_test(
    name = "cached_mem_pool_test",
    srcs = ["cc/cached_mem_pool_test.cc"],
    deps = [
        ":cached_mem_pool",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "snappy_inputbuffer",
    srcs = [
        "cc/snappy_inputbuffer.cc",
    ],
    hdrs = [
        "cc/snappy_inputbuffer.h",
    ],
    deps = [
        ":cached_mem_pool",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ],
)

cc_library(
    name = "ue_compress",
    srcs = ["cc/ue_compress.cc"],
    hdrs = ["cc/ue_compress.h"],
    deps = [
        "//idl:compression_qtz8mm",
        "//idl:proto_parser_cc_proto",
        "@com_google_glog//:glog",
    ],
)

cc_test(
    name = "ue_compress_test",
    srcs = ["cc/ue_compress_test.cc"],
    deps = [
        ":ue_compress",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "data_format_options",
    hdrs = ["cc/data_format_options.h"],
)

cc_library(
    name = "data_reader",
    srcs = [
        "cc/data_reader.cc",
        "cc/data_reader.h",
        "cc/pb_variant.cc",
        "cc/pb_variant.h",
    ],
    deps = [
        ":data_format_options",
        ":reader_util",
        ":snappy_inputbuffer",
        ":ue_compress",
        "//idl:example_cc_proto",
        "//idl:proto_parser_cc_proto",
        "//monolith/native_training/runtime/ops:traceme",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ],
)

cc_library(
    name = "data_writer",
    srcs = [
        "cc/data_writer.cc",
    ],
    hdrs = [
        "cc/data_writer.h",
    ],
    deps = [
        ":data_format_options",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ],
)

tf_cc_test(
    name = "data_read_write_test",
    srcs = [
        "cc/data_read_write_test.cc",
    ],
    deps = [
        ":data_reader",
        ":data_writer",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "parse_instance_lib",
    srcs = ["cc/parse_instance_lib.cc"],
    hdrs = ["cc/parse_instance_lib.h"],
    deps = [
        ":data_reader",
        ":reader_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:span",
        "@com_google_glog//:glog",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core/kernels:ops_util_hdrs",
    ],
)

cc_library(
    name = "instance_utils",
    srcs = ["cc/instance_utils.cc"],
    hdrs = ["cc/instance_utils.h"],
    deps = [
        ":reader_util",
        "//idl:example_cc_proto",
        "//idl:proto_parser_cc_proto",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "instance_utils_test",
    srcs = ["cc/instance_utils_test.cc"],
    deps = [
        ":instance_utils",
        "@com_google_absl//absl/time",
        "@com_google_glog//:glog",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_binary(
    name = "instance_processor",
    srcs = [
        "cc/instance_processor.cc",
    ],
    copts = ["-fexceptions"],
    deps = [
        ":data_reader",
        ":instance_utils",
        "//third_party/nlohmann:json",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

cc_library(
    name = "pb_datasource_lib",
    srcs = [
        "cc/instance_dataset_kernel.cc",
        "cc/parse_instance_kernel.cc",
    ],
    deps = [
        ":data_reader",
        ":instance_utils",
        ":parse_instance_lib",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "pb_datasource_ops",
    srcs = [
        "cc/instance_dataset_ops.cc",
        "cc/parse_instance_ops.cc",
    ],
    copts = ["-DNDEBUG"],
    deps = [
        ":pb_datasource_lib",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ],
    alwayslink = 1,
)

py_library(
    name = "instance_dataset_ops_py",
    srcs = [
        "python/instance_dataset_op.py",
    ],
    deps = [
        "//monolith:utils",
        "//monolith/native_training:runner_utils",
        "//monolith/native_training/distribute:distributed_dataset",
        "//monolith/native_training/hooks:ckpt_hooks",
        "//monolith/native_training/runtime/ops:gen_monolith_ops",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "parse_instance_ops_py_test",
    srcs = [
        "python/parse_instance_ops_test.py",
    ],
    main = "python/parse_instance_ops_test.py",
    deps = [
        ":instance_dataset_ops_py",
        ":parse_instance_ops_py",
        "//idl:proto_parser_py_proto",
    ],
)

py_binary(
    name = "instance_dataset_op_py_test_stdin",
    srcs = [
        "python/instance_dataset_op_test_stdin.py",
    ],
    main = "python/instance_dataset_op_test_stdin.py",
    deps = [
        ":instance_dataset_ops_py",
        ":parse_instance_ops_py",
        "//idl:proto_parser_py_proto",
    ],
)


py_library(
    name = "parser_utils",
    srcs = [
        "python/parser_utils.py",
    ],
    deps = [
        "//monolith/native_training:ragged_utils",
    ],
)

py_library(
    name = "parse_instance_ops_py",
    srcs = [
        "python/parse_instance_ops.py",
    ],
    deps = [
        ":parser_utils",
        "//idl:proto_parser_py_proto",
        "//monolith:utils",
        "//monolith/native_training/runtime/ops:gen_monolith_ops",
    ],
)

exports_files([
    "cc/parse_instance_kernel.cc",
    "cc/parse_instance_ops.cc",
])
