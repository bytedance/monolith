load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_grpc_library", "py_proto_library")
load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library")
load("@pip_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(default_visibility = [
    "//monolith/agent_service:__subpackages__",
    "//monolith/integration_test:__subpackages__",
    "//monolith/native_training:__subpackages__",
])

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
    deps = [
        "//idl:proto_parser_py_proto",
        "@org_tensorflow//tensorflow/core:protos_all_py",
        "@org_tensorflow//tensorflow/core/example:protos_all_py",
        "@org_tensorflow_serving//tensorflow_serving/apis:model_service_proto_py_pb2",
        "@org_tensorflow_serving//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "@org_tensorflow_serving//tensorflow_serving/apis:session_service_proto_py_pb2",
        "@org_tensorflow_serving//tensorflow_serving/config:platform_config_py_pb2",
        "@org_tensorflow_serving//tensorflow_serving/servables/tensorflow:saved_model_bundle_source_adapter_py_pb2",
        "@org_tensorflow_serving//tensorflow_serving/servables/tensorflow:session_bundle_config_py_pb2",
        requirement("dataclasses_json"),
        ":constants",
        "//monolith/native_training:env_utils",
        "//monolith/native_training:zk_utils",
    ],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    data = [
        "agent.conf",
        "//monolith/agent_service/test_data",
    ],
    srcs_version = "PY3",
    deps = [
        ":utils",
    ],
)

py_library(
    name = "backends",
    srcs = ["backends.py"],
    srcs_version = "PY3",
    deps = [
        requirement("kazoo"),
        requirement("dataclasses_json"),
        ":utils",
    ],
)

py_test(
    name = "backends_test",
    srcs = ["backends_test.py"],
    srcs_version = "PY3",
    deps = [
        ":backends",
        ":mocked_zkclient",
        ":utils",
    ],
)

py_library(
    name = "constants",
    srcs = ["constants.py"],
)

py_binary(
    name = "agent_controller",
    srcs = [
        "agent_controller.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":backends",
        ":utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)


py_library(
    name = "replica_manager",
    srcs = ["replica_manager.py"],
    srcs_version = "PY3",
    deps = [
        ":agent_service_py_pb2",
        ":backends",
        ":data_def",
        ":resource_utils",
        ":tfs_monitor",
        ":utils",
        "//monolith/native_training/metric:cli",
        "//monolith/native_training/model_export:export_state_utils",
        requirement("dataclasses_json"),
    ],
)

py_library(
    name = "model_manager",
    srcs = ["model_manager.py"],
    srcs_version = "PY3",
    deps = [
        "//monolith/native_training/metric:cli",
    ],
)

py_test(
    name = "model_manager_test",
    srcs = ["model_manager_test.py"],
    srcs_version = "PY3",
    deps = [
        ":model_manager",
    ],
)

py_test(
    name = "replica_manager_test",
    srcs = ["replica_manager_test.py"],
    srcs_version = "PY3",
    deps = [
        ":constants",
        ":mocked_tfserving",
        ":mocked_zkclient",
        ":replica_manager",
        ":tfs_monitor",
        ":utils",
    ],
)

py_library(
    name = "agent_base",
    srcs = ["agent_base.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
    ],
)

py_library(
    name = "agent_v1",
    srcs = ["agent_v1.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":agent_base",
        ":agent_service",
        ":replica_manager",
        ":tfs_monitor",
    ],
)

py_library(
    name = "mocked_tfserving",
    srcs = [
        "mocked_tfserving.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":utils",
    ],
)

py_test(
    name = "mocked_tfserving_test",
    srcs = [
        "mocked_tfserving_test.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":mocked_tfserving",
    ],
)

py_library(
    name = "mocked_zkclient",
    srcs = [
        "mocked_zkclient.py",
    ],
    srcs_version = "PY3",
    deps = [
        requirement("kazoo"),
    ],
)

py_test(
    name = "mocked_zkclient_test",
    srcs = [
        "mocked_zkclient_test.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":mocked_zkclient",
    ],
)

proto_library(
    name = "agent_service_proto",
    srcs = ["agent_service.proto"],
)

py_proto_library(
    name = "agent_service_py_pb2",
    deps = [":agent_service_proto"],
)

py_grpc_library(
    name = "agent_service_py_pb2_grpc",
    srcs = [":agent_service_proto"],
    deps = [":agent_service_py_pb2"],
)

py_library(
    name = "agent_service",
    srcs = ["agent_service.py"],
    srcs_version = "PY3",
    deps = [
        ":agent_service_py_pb2",
        ":agent_service_py_pb2_grpc",
        ":data_def",
        ":replica_manager",
        ":resource_utils",
        ":utils",
        ":zk_mirror",
    ],
)

py_binary(
    name = "agent_client",
    srcs = ["agent_client.py"],
    srcs_version = "PY3",
    deps = [
        ":agent_service_py_pb2",
        ":agent_service_py_pb2_grpc",
        ":utils",
        requirement("kazoo"),
        ":client",
        ":data_def",
        ":resource_utils",
    ],
)

py_test(
    name = "agent_service_test",
    srcs = [
        "agent_service_test.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":agent_service",
        ":mocked_zkclient",
        ":svr_client",
    ],
)

py_binary(
    name = "tfs_client",
    srcs = ["tfs_client.py"],
    deps = [
        ":client",
        ":utils",
        "//idl:example_py_proto",
        "//idl:line_id_py_proto",
        "//idl:proto_parser_py_proto",
        "//monolith/native_training/data:feature_list",
        "//monolith/native_training/model_export",
    ],
)


cc_proto_library(
    name = "agent_service_cc_proto",
    srcs = [
        "agent_service.proto",
    ],
    visibility = ["//visibility:public"],
)

cc_grpc_library(
    name = "agent_service_cc_proto_grpc",
    srcs = [
        ":agent_service_proto",
    ],
    generate_mocks = True,
    grpc_only = True,
    visibility = ["//visibility:public"],
    deps = [
        ":agent_service_cc_proto",
    ],
)

py_library(
    name = "data_def",
    srcs = ["data_def.py"],
    srcs_version = "PY3",
    deps = [
        ":utils",
        "//monolith/native_training:net_utils",
    ],
)

py_test(
    name = "data_def_test",
    srcs = ["data_def_test.py"],
    srcs_version = "PY3",
    deps = [
        ":data_def",
    ],
)

py_library(
    name = "tfs_monitor",
    srcs = ["tfs_monitor.py"],
    srcs_version = "PY3",
    deps = [
        ":data_def",
        ":utils",
    ],
)

py_test(
    name = "tfs_monitor_test",
    srcs = ["tfs_monitor_test.py"],
    srcs_version = "PY3",
    deps = [
        ":constants",
        ":mocked_tfserving",
        ":tfs_monitor",
    ],
)

py_library(
    name = "zk_mirror",
    srcs = ["zk_mirror.py"],
    deps = [
        ":data_def",
        ":utils",
        requirement("kazoo"),
    ],
)

py_test(
    name = "zk_mirror_test",
    srcs = ["zk_mirror_test.py"],
    deps = [
        ":agent_service_py_pb2",
        ":constants",
        ":mocked_tfserving",
        ":mocked_zkclient",
        ":zk_mirror",
    ],
)

py_library(
    name = "resource_utils",
    srcs = ["resource_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":data_def",
        ":utils",
        "//monolith/native_training/model_export:export_py_proto",
        "//monolith/native_training/model_export:export_state_utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
        requirement("psutil"),
    ],
)

py_test(
    name = "resource_utils_test",
    srcs = ["resource_utils_test.py"],
    deps = [
        ":resource_utils",
    ],
)

py_library(
    name = "tfs_wrapper",
    srcs = ["tfs_wrapper.py"],
    data = [
        "//conf:serving",
    ],
    deps = [
        ":utils",
        "//monolith:utils",
    ],
)

py_library(
    name = "agent_v3",
    srcs = ["agent_v3.py"],
    srcs_version = "PY3",
    deps = [
        ":agent_base",
        ":agent_service",
        ":backends",
        ":data_def",
        ":resource_utils",
        ":tfs_wrapper",
    ],
)

py_test(
    name = "agent_v3_test",
    srcs = ["agent_v3_test.py"],
    srcs_version = "PY3",
    deps = [
        ":agent_v3",
        ":mocked_zkclient",
    ],
)

py_binary(
    name = "client",
    srcs = ["client.py"],
    srcs_version = "PY3",
    deps = [
        ":data_def",
        ":utils",
        ":zk_mirror",
        requirement("kazoo"),
    ],
)

filegroup(
    name = "agent_internal_data",
)

py_binary(
    name = "agent",
    srcs = ["agent.py"],
    data = [
        ":agent_internal_data",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":agent_v1",
        ":agent_v3",
        ":mocked_zkclient",
        ":model_manager",
    ],
)

py_binary(
    name = "svr_client",
    srcs = ["svr_client.py"],
    srcs_version = "PY3",
    deps = [
        ":agent_service_py_pb2",
        ":agent_service_py_pb2_grpc",
        ":utils",
    ],
)

filegroup(
    name = "agent_exported",
)
