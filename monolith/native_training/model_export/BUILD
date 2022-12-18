load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "export_proto",
    srcs = ["export.proto"],
)

py_proto_library(
    name = "export_py_proto",
    srcs = ["export.proto"],
)

py_library(
    name = "export_context",
    srcs = ["export_context.py"],
    deps = [
        "//monolith:utils",
        "//monolith/native_training:monolith_export",
        "//monolith/native_training:utils",
    ],
)

py_library(
    name = "saved_model_exporters",
    srcs = ["saved_model_exporters.py"],
    deps = [
        ":data_gen_utils",
        ":export_context",
        "//monolith/native_training:device_utils",
        "//monolith/native_training:hash_table_ops",
        "//monolith/native_training:monolith_export",
        "//monolith/native_training:multi_hash_table_ops",
        "//monolith/native_training:multi_type_hash_table",
        "//monolith/native_training/model_dump:dump_utils",
    ],
)

py_test(
    name = "saved_model_exporters_test",
    srcs = ["saved_model_exporters_test.py"],
    deps = [
        ":saved_model_exporters",
        "//monolith/native_training:test_utils",
    ],
)

py_library(
    name = "export_state_utils",
    srcs = ["export_state_utils.py"],
    deps = [
        ":export_py_proto",
    ],
)

py_test(
    name = "export_state_utils_test",
    srcs = ["export_state_utils_test.py"],
    deps = [
        ":export_state_utils",
    ],
)

py_library(
    name = "export_hooks",
    srcs = ["export_hooks.py"],
    deps = [
        ":export_py_proto",
        ":export_state_utils",
        ":saved_model_exporters",
        "//monolith/native_training:save_utils",
    ],
)

py_test(
    name = "export_hooks_test",
    srcs = ["export_hooks_test.py"],
    deps = [
        ":export_hooks",
        ":saved_model_exporters",
        "//monolith/native_training:save_utils",
    ],
)

py_binary(
    name = "saved_model_visulizer",
    srcs = ["saved_model_visulizer.py"],
    deps = [
        "//monolith/native_training:distribution_ops",
        "//monolith/native_training:hash_table_ops",
    ],
)

py_binary(
    name = "warmup_data_gen",
    srcs = ["warmup_data_gen.py"],
    deps = [
        ":data_gen_utils",
        "//monolith/native_training:cpu_training",
        "@org_tensorflow_serving//tensorflow_serving/apis:predict_proto_py_pb2",
        "@org_tensorflow_serving//tensorflow_serving/apis:prediction_log_proto_py_pb2",
    ],
)

py_binary(
    name = "warmup_data_decoder",
    srcs = ["warmup_data_decoder.py"],
    deps = [
        ":data_gen_utils",
        "//monolith/native_training:cpu_training",
        "@org_tensorflow_serving//tensorflow_serving/apis:predict_proto_py_pb2",
        "@org_tensorflow_serving//tensorflow_serving/apis:prediction_log_proto_py_pb2",
    ],
)

py_binary(
    name = "warmup_example_batch",
    srcs = ["warmup_example_batch.py"],
    deps = [
        ":data_gen_utils",
        "//monolith/native_training:cpu_training",
        "@org_tensorflow_serving//tensorflow_serving/apis:predict_proto_py_pb2",
        "@org_tensorflow_serving//tensorflow_serving/apis:prediction_log_proto_py_pb2",
    ],
)

py_binary(
    name = "demo_export",
    srcs = ["demo_export.py"],
    deps = [
        ":saved_model_exporters",
        "//monolith/native_training:cpu_training",
        "//monolith/native_training:model",
        "//monolith/native_training/data/training_instance:parse_instance_ops_py",
    ],
)

py_test(
    name = "demo_export_test",
    srcs = ["demo_export_test.py"],
    deps = [
        ":demo_export",
    ],
)

py_binary(
    name = "demo_predictor",
    srcs = ["demo_predictor.py"],
    deps = [
        "//idl:proto_parser_py_proto",
        "//monolith/native_training:distribution_ops",
        "//monolith/native_training:hash_filter_ops",
        "//monolith/native_training:hash_table_ops",
        "//monolith/native_training:logging_ops",
        "//monolith/native_training:model",
        "//monolith/native_training/data/training_instance:parse_instance_ops_py",
    ],
)

py_binary(
    name = "demo_predictor_client",
    srcs = ["demo_predictor_client.py"],
    deps = [
        ":demo_predictor",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow_serving//tensorflow_serving/apis:predict_proto_py_pb2",
        "@org_tensorflow_serving//tensorflow_serving/apis:prediction_service_proto_py_pb2",
    ],
)

py_library(
    name = "model_export",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":export_context",
        ":saved_model_exporters",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "data_gen_utils",
    srcs = ["data_gen_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":export_context",
        "//idl:example_py_proto",
        "//idl:line_id_py_proto",
        "//monolith/native_training:env_utils",
        "//monolith/native_training:utils",
        "//monolith/native_training/data:feature_list",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow_serving//tensorflow_serving/apis:predict_proto_py_pb2",
        "@org_tensorflow_serving//tensorflow_serving/apis:prediction_log_proto_py_pb2",
    ],
)

py_test(
    name = "data_gen_utils_test",
    srcs = ["data_gen_utils_test.py"],
    data = ["//monolith/native_training/data/test_data:test_feature_lists"],
    srcs_version = "PY3",
    deps = [
        ":data_gen_utils",
        "//monolith/native_training/data:datasets_py",
        "//monolith/native_training/data:feature_utils_py",
        "//monolith/native_training/data:parsers_py",
    ],
)

py_library(
    name = "export_utils",
    srcs = ["export_utils.py"],
    deps = [
        ":export_context",
        "//monolith/native_training:distributed_serving_ops",
        "//monolith/native_training:nested_tensors",
    ],
)

py_test(
    name = "export_utils_test",
    srcs = ["export_utils_test.py"],
    deps = [
        ":export_utils",
    ],
)
