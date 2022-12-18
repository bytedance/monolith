load("@rules_cc//cc:defs.bzl", "cc_library", "cc_proto_library", "cc_test")
load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")
load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "if_cuda_is_configured_compat",
    "tf_gpu_kernel_library_allow_except",
)

package(default_visibility = ["//monolith/native_training/runtime:__subpackages__"])

proto_library(
    name = "embedding_hash_table_proto",
    srcs = ["embedding_hash_table.proto"],
    deps = [
        "//monolith/native_training/runtime/hash_table/compressor:float_compressor_proto",
        "//monolith/native_training/runtime/hash_table/initializer:initializer_config_proto",
        "//monolith/native_training/runtime/hash_table/optimizer:optimizer_proto",
    ],
)

cc_proto_library(
    name = "embedding_hash_table_cc_proto",
    visibility = ["//visibility:public"],
    deps = [":embedding_hash_table_proto"],
)

py_proto_library(
    name = "embedding_hash_table_py_proto",
    srcs = ["embedding_hash_table.proto"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//monolith/native_training/runtime/hash_table/compressor:float_compressor_py_proto",
        "//monolith/native_training/runtime/hash_table/initializer:initializer_config_py_proto",
        "//monolith/native_training/runtime/hash_table/optimizer:optimizer_py_proto",
    ],
)

cc_library(
    name = "entry_accessor",
    srcs = ["entry_accessor.cc"],
    hdrs = [
        "entry_accessor.h",
        "entry_accessor_decorator.h",
        "quantized_entry_accessor.h",
    ],
    deps = [
        ":embedding_hash_table_cc_proto",
        ":utils",
        "//monolith/native_training/runtime/hash_table/compressor:float_compressor",
        "//monolith/native_training/runtime/hash_table/initializer:initializer_combination",
        "//monolith/native_training/runtime/hash_table/initializer:initializer_factory",
        "//monolith/native_training/runtime/hash_table/optimizer:optimizer_combination",
        "//monolith/native_training/runtime/hash_table/optimizer:optimizer_factory",
        "//monolith/native_training/runtime/hash_table/retriever:fake_quant_retriever",
        "//monolith/native_training/runtime/hash_table/retriever:hash_net_retriever",
        "//monolith/native_training/runtime/hash_table/retriever:raw_retriever",
        "//monolith/native_training/runtime/hash_table/retriever:retriever_combination",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_glog//:glog",
    ],
)

cc_test(
    name = "entry_accessor_test",
    srcs = ["entry_accessor_test.cc"],
    deps = [
        ":embedding_hash_table_cc_proto",
        ":entry_accessor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "quantized_entry_accessor",
    hdrs = [
        "entry_accessor_decorator.h",
        "quantized_entry_accessor.h",
    ],
    deps = [
        ":entry_accessor",
        "//monolith/native_training/runtime/hash_table/compressor:fake_quantizer",
    ],
)

cc_test(
    name = "quantized_entry_accessor_test",
    srcs = ["quantized_entry_accessor_test.cc"],
    deps = [
        ":quantized_entry_accessor",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_gpu_kernel_library_allow_except(
    name = "embedding_hash_table_interface",
    srcs = [],
    hdrs = ["embedding_hash_table_interface.h"],
    deps = [
        ":embedding_hash_table_cc_proto",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "embedding_hash_table_factory_internal_deps",
)

tf_gpu_kernel_library_allow_except(
    name = "embedding_hash_table_factory",
    srcs = ["embedding_hash_table_factory.cc"],
    hdrs = ["embedding_hash_table_factory.h"],
    deps = [
        ":embedding_hash_table_cc_proto",
        ":embedding_hash_table_factory_internal_deps",
        ":embedding_hash_table_interface",
        ":entry_accessor",
        "//monolith/native_training/runtime/hash_table/cuckoohash:cuckoo_embedding_hash_table",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "embedding_hash_table_test",
    hdrs = ["embedding_hash_table_test.h"],
    deps = [
        ":embedding_hash_table_cc_proto",
        ":embedding_hash_table_factory",
        ":embedding_hash_table_interface",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "hash_table_benchmark",
    srcs = ["hash_table_benchmark.cc"],
    deps = [
        ":embedding_hash_table_cc_proto",
        ":embedding_hash_table_factory",
        ":embedding_hash_table_interface",
        "//monolith/native_training/runtime/concurrency:thread_pool",
        "@com_github_google_benchmark//:benchmark",
        "@com_google_absl//absl/random",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "utils",
    hdrs = ["utils.h"],
)

cc_library(
    name = "entry_defs",
    hdrs = ["entry_defs.h"],
    deps = [
        "//monolith/native_training/runtime/allocator:block_allocator",
    ],
)

cc_test(
    name = "entry_defs_test",
    srcs = ["entry_defs_test.cc"],
    deps = [
        ":entry_defs",
        "@com_google_googletest//:gtest_main",
    ],
)
