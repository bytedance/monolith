load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test", "tf_custom_op_library")
load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library", "py_proto_library")
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

cc_proto_library(
    name = "data_op_config_cc_proto",
    srcs = ["data_op_config.proto"],
)

py_proto_library(
    name = "data_op_config_py_proto",
    srcs = ["data_op_config.proto"],
    srcs_version = "PY2AND3",
)

cc_library(
    name = "pb_data_internal_lib",
    alwayslink = 1,
)

cc_library(
    name = "pb_data_lib",
    srcs = [
        "kernels/add_action_kernel.cc",
        "kernels/add_label_kernel.cc",
        "kernels/df_resource_kernel.cc",
        "kernels/df_resource_kernel.h",
        "kernels/dynamic_match_file_dataset_kernel.cc",
        "kernels/extract_fid_kernel.cc",
        "kernels/feature_hash.cc",
        "kernels/feature_name_mapper_tf_bridge.cc",
        "kernels/feature_name_mapper_tf_bridge.h",
        "kernels/fill_multi_rank_output_kernel.cc",
        "kernels/filter_by_label_kernel.cc",
        "kernels/instance_reweight_dataset_kernel.cc",
        "kernels/instance_reweight_dataset_kernel.h",
        "kernels/item_pool_kernels.cc",
        "kernels/item_pool_kernels.h",
        "kernels/kafka_kernels.cc",
        "kernels/label_normalization_kernel.cc",
        "kernels/label_upper_bound_kernel.cc",
        "kernels/map_id_kernels.cc",
        "kernels/merge_flow_dataset_kernel.cc",
        "kernels/multi_label_gen_kernel.cc",
        "kernels/negative_gen_dataset_kernel.cc",
        "kernels/negative_gen_dataset_kernel.h",
        "kernels/parse_example_lib.cc",
        "kernels/parse_example_lib.h",
        "kernels/parse_input_data_kernel.cc",
        "kernels/parse_sparse_feature.cc",
        "kernels/parse_sparse_feature.h",
        "kernels/pb_dataset_kernel.cc",
        "kernels/ragged_feature_kernel.cc",
        "kernels/scatter_label_kernel.cc",
        "kernels/split_flow_dataset_kernel.cc",
        "kernels/string_to_variant.cc",
        "kernels/variant_filter_kernel.cc",
    ],
    deps = [
        ":data_op_config_cc_proto",
        ":pb_data_internal_lib",
        "//monolith/native_training/data/kernels/internal:cache_mgr",
        "//monolith/native_training/data/kernels/internal:datasource_utils",
        "//monolith/native_training/data/kernels/internal:file_match_split_provider",
        "//monolith/native_training/data/kernels/internal:label_utils",
        "//monolith/native_training/data/kernels/internal:relational_utils",
        "//monolith/native_training/data/training_instance:data_reader",
        "//monolith/native_training/data/training_instance:instance_utils",
        "//monolith/native_training/data/training_instance:parse_instance_lib",
        "//monolith/native_training/data/training_instance:reader_util",
        "//monolith/native_training/runtime/common:metrics",
        "//monolith/native_training/runtime/concurrency:queue",
        "//monolith/native_training/runtime/ops:traceme",
        "//third_party/nlohmann:json",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash:city",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings:str_format",
        "@kafka",
    ],
    alwayslink = 1,
)

cc_library(
    name = "pb_data_ops",
    srcs = [
        "ops/feature_utils_ops.cc",
        "ops/parse_input_data_ops.cc",
        "ops/pb_dataset_ops.cc",
    ],
    copts = ["-DNDEBUG"],
    deps = [
        ":pb_data_lib",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ],
    alwayslink = 1,
)

py_library(
    name = "datasets_py",
    srcs = [
        "datasets.py",
    ],
    deps = [
        ":feature_list",
        "//monolith:utils",
        "//monolith/native_training:monolith_export",
        "//monolith/native_training/distribute:distributed_dataset",
        "//monolith/native_training/hooks:ckpt_hooks",
        "//monolith/native_training/runtime/ops:gen_monolith_ops",
        "@org_tensorflow//tensorflow:tensorflow_py",
        requirement("kafka_python"),
    ],
)

py_library(
    name = "parsers_py",
    srcs = [
        "parsers.py",
    ],
    deps = [
        ":data_op_config_py_proto",
        ":feature_list",
        "//monolith:utils",
        "//monolith/native_training:monolith_export",
        "//monolith/native_training:utils",
        "//monolith/native_training/runtime/ops:gen_monolith_ops",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "feature_utils_py",
    srcs = [
        "feature_utils.py",
    ],
    deps = [
        ":data_op_config_py_proto",
        "//idl:example_py_proto",
        "//idl:proto_parser_py_proto",
        "//monolith:utils",
        "//monolith/native_training:monolith_export",
        "//monolith/native_training/runtime/ops:gen_monolith_ops",
    ],
)


py_library(
    name = "feature_list",
    srcs = ["feature_list.py"],
    deps = [
        ":utils",
        "//monolith/native_training:utils",
    ],
)

py_test(
    name = "feature_list_test",
    srcs = ["feature_list_test.py"],
    data = ["//monolith/native_training/data/test_data:test_feature_lists"],
    deps = [
        ":feature_list",
    ],
)


py_library(
    name = "data",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":datasets_py",
        ":feature_utils_py",
        ":parsers_py",
    ],
)

py_test(
    name = "extract_fid_test",
    srcs = [
        "extract_fid_test.py",
    ],
    main = "extract_fid_test.py",
    deps = [
        "//idl:example_py_proto",
        "//idl:proto_parser_py_proto",
        "//monolith:utils",
        "//monolith/native_training/runtime/ops:gen_monolith_ops",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
)

py_library(
    name = "item_pool_hook",
    srcs = ["item_pool_hook.py"],
    deps = [
        ":datasets_py",
        ":feature_utils_py",
    ],
)

py_test(
    name = "item_pool_test",
    srcs = [
        "item_pool_test.py",
    ],
    deps = [
        ":feature_utils_py",
        "//monolith:utils",
    ],
)

py_test(
    name = "multi_flow_test",
    srcs = [
        "multi_flow_test.py",
    ],
    main = "multi_flow_test.py",
    deps = [
        ":datasets_py",
        ":feature_utils_py",
        ":parsers_py",
        "//idl:example_py_proto",
        "//idl:proto_parser_py_proto",
    ],
)

py_test(
    name = "negative_gen_test",
    srcs = [
        "negative_gen_test.py",
    ],
    main = "negative_gen_test.py",
    deps = [
        ":datasets_py",
        ":feature_utils_py",
        ":parsers_py",
        "//idl:example_py_proto",
        "//idl:proto_parser_py_proto",
    ],
)


py_binary(
    name = "kafka_dataset_test",
    srcs = [
        "kafka_dataset_test.py",
    ],
    deps = [
        ":datasets_py",
        ":feature_utils_py",
        ":parsers_py",
        "//idl:example_py_proto",
        "//idl:proto_parser_py_proto",
        "//monolith/native_training/model_export:data_gen_utils",
    ],
)

py_binary(
    name = "data_service_test",
    srcs = [
        "data_service_test.py",
    ],
    deps = [
        ":datasets_py",
        ":feature_utils_py",
    ],
)

exports_files([
    "kernels/add_action_kernel.cc",
    "kernels/instance_reweight_dataset_kernel.cc",
    "kernels/instance_reweight_dataset_kernel.h",
    "kernels/negative_gen_dataset_kernel.cc",
    "kernels/negative_gen_dataset_kernel.h",
    "kernels/df_resource_kernel.h",
    "kernels/df_resource_kernel.cc",
    "kernels/split_flow_dataset_kernel.cc",
    "kernels/merge_flow_dataset_kernel.cc",
    "kernels/parse_example_lib.cc",
    "kernels/parse_example_lib.h",
    "kernels/parse_input_data_kernel.cc",
    "kernels/pb_dataset_kernel.cc",
    "kernels/ragged_feature_kernel.cc",
    "kernels/variant_filter_kernel.cc",
    "kernels/item_pool_kernels.h",
    "kernels/item_pool_kernels.cc",
    "ops/feature_utils_ops.cc",
    "ops/parse_input_data_ops.cc",
    "ops/pb_dataset_ops.cc",
])
