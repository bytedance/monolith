load("@rules_cc//cc:defs.bzl", "cc_library", "cc_proto_library", "cc_test")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "cc_header_only_library", "tf_cc_test", "tf_custom_op_library", "tf_gen_op_wrapper_py", "tf_gpu_kernel_library_allow_except", "tf_kernel_library")
load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(default_visibility = [
    "//monolith:__subpackages__",
    "@org_tensorflow//:__subpackages__",
])

cc_header_only_library(
    name = "traceme",
    deps = [
        "@org_tensorflow//tensorflow/core/profiler/lib:traceme",
    ],
)

cc_library(
    name = "tracelib",
    deps = [
        ":traceme",
        "//monolith/native_training/runtime/common:metrics",
        "@com_google_glog//:glog",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ],
)

tf_gpu_kernel_library_allow_except(
    name = "embedding_hash_table_tf_bridge",
    srcs = ["embedding_hash_table_tf_bridge.cc"],
    hdrs = ["embedding_hash_table_tf_bridge.h"],
    deps = [
        ":hash_filter_tf_bridge",
        "//monolith/native_training/runtime/common:metrics",
        "//monolith/native_training/runtime/hash_filter:filter",
        "//monolith/native_training/runtime/hash_filter:probabilistic_filter",
        "//monolith/native_training/runtime/hash_filter:sliding_hash_filter",
        "//monolith/native_training/runtime/hash_table:embedding_hash_table_factory",
        "//monolith/native_training/runtime/hopscotch:hopscotch_hash_set",
        "@com_google_absl//absl/memory",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core/kernels:ops_util_hdrs",
    ],
)

cc_library(
    name = "hash_filter_tf_bridge",
    srcs = ["hash_filter_tf_bridge.cc"],
    hdrs = ["hash_filter_tf_bridge.h"],
    deps = [
        ":file_utils",
        "//monolith/native_training/data/training_instance:reader_util",
        "//monolith/native_training/runtime/hash_filter:filter",
        "//monolith/native_training/runtime/hash_table:embedding_hash_table_cc_proto",
        "@com_google_absl//absl/strings:str_format",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core/kernels:ops_util_hdrs",
    ],
)

cc_library(
    name = "touched_key_set_tf_bridge",
    srcs = [],
    hdrs = ["touched_key_set_tf_bridge.h"],
    deps = [
        "//monolith/native_training/runtime/hopscotch:hopscotch_hash_set",
        "@com_google_absl//absl/strings:str_format",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core/kernels:ops_util_hdrs",
    ],
)

cc_library(
    name = "parameter_sync_tf_bridge",
    srcs = ["parameter_sync_tf_bridge.cc"],
    hdrs = ["parameter_sync_tf_bridge.h"],
    deps = [
        ":embedding_hash_table_tf_bridge",
        ":multi_hash_table",
        "//monolith/native_training/runtime/parameter_sync:dummy_sync_client",
        "//monolith/native_training/runtime/parameter_sync:dummy_sync_server",
        "//monolith/native_training/runtime/parameter_sync:sync_client_manager",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core/kernels:ops_util_hdrs",
    ],
)

cc_library(
    name = "file_utils",
    srcs = ["file_utils.cc"],
    hdrs = ["file_utils.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_protobuf//:protobuf_lite",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core/kernels:ops_util_hdrs",
    ],
)

tf_cc_test(
    name = "file_utils_test",
    srcs = ["file_utils_test.cc"],
    deps = [
        ":file_utils",
        "//monolith/native_training/data/training_instance:reader_util",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

tf_kernel_library(
    name = "clip_ops",
    srcs = [
        "clip_by_global_norm.h",
        "clip_by_global_norm_op.cc",
    ],
    copts = [
        "-D_ENABLE_AVX",
        "-mavx",
        "-mfma",
    ],
    gpu_srcs = [
        "clip_by_global_norm.h",
        "clip_by_global_norm.cu.cc",
        "global_norm.cu.cc",
    ],
    linkopts = [],
    deps = [
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core/kernels:gpu_device_array_for_custom_op",
        "@org_tensorflow//tensorflow/core/kernels:gpu_prim_hdrs",
    ],
)

cc_library(
    name = "multi_hash_table",
    hdrs = ["multi_hash_table.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ],
)

tf_gpu_kernel_library_allow_except(
    name = "hash_table_ops",
    srcs = [
        "hash_table/misc_ops.cc",
        "hash_table_lookup_op.cc",
        "hash_table_op.cc",
        "hash_table_restore_op.cc",
        "hash_table_save_op.cc",
        "hash_table_update_op.cc",
        "multi_hash_table.h",
        "multi_hash_table_lookup_op.cc",
        "multi_hash_table_op.cc",
        "multi_hash_table_save_restore_ops.cc",
        "multi_hash_table_update_op.cc",
    ],
    deps = [
        ":embedding_hash_table_tf_bridge",
        ":file_utils",
        ":hash_filter_tf_bridge",
        ":multi_hash_table",
        ":parameter_sync_tf_bridge",
        "//monolith/native_training/data/training_instance:reader_util",
        "//monolith/native_training/runtime/concurrency:queue",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "monolith_internal_ops",
    alwayslink = 1,
)

cc_library(
    name = "monolith_ops_additional_deps",
    deps = select({
        "@org_tensorflow//tensorflow:framework_shared_object": ["@org_tensorflow//tensorflow/core/platform/hadoop:hadoop_file_system"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "monolith_ops_no_remote_predict",
    deps = [
        ":clip_ops",
        ":deep_insight_ops",
        ":distribution_ops",
        ":file_ops",
        ":gen_seq_mask_op",
        ":hash_filter_ops",
        ":hash_table_ops",
        ":inbatch_auc_loss_ops",
        ":logging_ops",
        ":monolith_internal_ops",
        ":monolith_ops_additional_deps",
        ":parameter_sync_ops",
        ":touched_key_set_ops",
        "//monolith/native_training/data:pb_data_ops",
        "//monolith/native_training/data/training_instance:pb_datasource_ops",
        "//monolith/native_training/layers:layer_tf_ops",
        "//monolith/native_training/optimizers:training_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "monolith_ops",
    deps = [
        ":monolith_ops_no_remote_predict",
        ":remote_predict_op_grpc",
    ],
    alwayslink = 1,
)

# if framework_shared_object is true,
# we shouldn't link the ops into tensorflow because
# we don't separate the ops/kernels implementation.
# Instead, we use dynamic load to solve this problem.
cc_library(
    name = "monolith_ops_for_tf",
    deps = select({
        "@org_tensorflow//tensorflow:framework_shared_object": [],
        "//conditions:default": [
            ":monolith_ops",
        ],
    }),
    alwayslink = 1,
)

tf_kernel_library(
    name = "monolith_ops_for_load",
    deps = select({
        "@org_tensorflow//tensorflow:framework_shared_object": [":monolith_ops"],
        "//conditions:default": [],
    }),
)

tf_gen_op_wrapper_py(
    name = "gen_monolith_ops_base",
    out = "gen_monolith_ops_base.py",
    deps = [":monolith_ops"],
)

py_library(
    name = "gen_monolith_ops",
    srcs = ["gen_monolith_ops.py"],
    data = [":libtfkernel_monolith_ops_for_load.so"],
    deps = [
        ":gen_monolith_ops_base",
        "//monolith:utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

tf_kernel_library(
    name = "distribution_ops",
    srcs = [
        "fused_embedding_to_layout.cc",
        "fused_reorder_by_indices.cc",
        "map_id_to_embedding_op.cc",
        "reduce_op.cc",
        "split_by_indices_op.cc",
        "static_reshape_op.cc",
        "unique_mapping_ops.cc",
    ],
    copts = [
        "-D_ENABLE_AVX",
        "-mavx",
        "-mfma",
    ],
    gpu_srcs = [
        "map_id_to_embedding.cu.cc",
        "reduce_op.cu.cc",
        "alloc_utils.h"
    ],
    # TODO: Figure out how to link "@org_tensorflow//tensorflow/core/kernels:cwise_lib_hdrs" for fill_functor.h
    deps = [
        "//idl:example_cc_proto",
        "//monolith/native_training/data/training_instance:data_reader",
        "//monolith/native_training/data/training_instance:parse_instance_lib",
        "//monolith/native_training/runtime/hash_table:embedding_hash_table_factory",
        "//monolith/native_training/runtime/ops:traceme",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core/kernels:gpu_device_array_for_custom_op",
    ],
)

tf_gpu_kernel_library_allow_except(
    name = "hash_filter_ops",
    srcs = [
        "hash_filter_intercept_gradient_op.cc",
        "hash_filter_op.cc",
        "hash_filter_restore_op.cc",
        "hash_filter_save_op.cc",
    ],
    deps = [
        ":file_utils",
        ":hash_filter_tf_bridge",
        "//monolith/native_training/runtime/hash_filter",
        "//monolith/native_training/runtime/hash_filter:dummy_hash_filter",
        "//monolith/native_training/runtime/hash_filter:probabilistic_filter",
        "//monolith/native_training/runtime/hash_filter:sliding_hash_filter",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ]
)

cc_library(
    name = "file_ops",
    srcs = [
        "file_ops.cc",
    ],
    deps = [
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "touched_key_set_ops",
    srcs = [
        "touched_key_set_insert_op.cc",
        "touched_key_set_op.cc",
        "touched_key_set_steal_op.cc",
    ],
    deps = [
        ":touched_key_set_tf_bridge",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "gen_seq_mask_op",
    srcs = [
        "gen_seq_mask.cc",
    ],
    deps = [
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "inbatch_auc_loss_ops",
    srcs = [
        "inbatch_auc_loss.cc",
    ],
    deps = [
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "remote_predict_op_lib",
    hdrs = ["remote_predict_op.h"],
    deps = [
        ":agent_heartbeat",
        ":tracelib",
        "//monolith/native_training/runtime/common:metrics",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_glog//:glog",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow_serving//tensorflow_serving/apis:prediction_service_proto",
    ],
)

cc_library(
    name = "prediction_service_grpc",
    srcs = [
        "prediction_service_grpc.cc",
    ],
    hdrs = [
        "prediction_service_grpc.h",
    ],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@org_tensorflow_serving//tensorflow_serving/apis:prediction_service_proto",
    ],
)

cc_library(
    name = "remote_predict_op_grpc",
    srcs = ["remote_predict_op_grpc.cc"],
    deps = [
        ":prediction_service_grpc",
        ":remote_predict_op_lib",
    ],
    alwayslink = 1,
)
alias(
    name = "remote_predict_op",
    actual = ":remote_predict_op_grpc",
)
tf_gpu_kernel_library_allow_except(
    name = "parameter_sync_ops",
    srcs = ["parameter_sync_ops.cc"],
    deps = [
        ":parameter_sync_tf_bridge",
        "@com_github_grpc_grpc//:grpc++_reflection",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ]
)

proto_library(
    name = "logging_ops_proto",
    srcs = ["logging_ops.proto"],
)

cc_proto_library(
    name = "logging_ops_cc_proto",
    visibility = ["//visibility:public"],
    deps = [":logging_ops_proto"],
)

py_proto_library(
    name = "logging_ops_py_proto",
    srcs = ["logging_ops.proto"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "logging_ops",
    srcs = [
        "logging_ops.cc",
    ],
    deps = [
        ":logging_ops_cc_proto",
        "//monolith/native_training/runtime/common:metrics",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "deep_insight_client_tf_bridge",
    hdrs = ["deep_insight_client_tf_bridge.h"],
    deps = [
        "//monolith/native_training/runtime/deep_insight",
        "@com_google_absl//absl/strings:str_format",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core/kernels:ops_util_hdrs",
    ],
)

cc_library(
    name = "deep_insight_ops",
    srcs = [
        "deep_insight_ops.cc",
    ],
    deps = [
        ":deep_insight_client_tf_bridge",
    ],
    alwayslink = 1,
)

cc_library(
    name = "agent_heartbeat",
    srcs = [
        "agent_heartbeat.cc",
    ],
    hdrs = [
        "agent_heartbeat.h",
    ],
    deps = [
        ":net_utils",
        "//monolith/agent_service:agent_service_cc_proto_grpc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_glog//:glog",
        "@org_tensorflow//tensorflow/core/platform:logging",
        "@org_tensorflow_serving//tensorflow_serving/apis:prediction_service_proto",
    ],
)

tf_cc_test(
    name = "agent_heartbeat_test",
    srcs = ["agent_heartbeat_test.cc"],
    extra_copts = [
        "-DTEST_USE_GRPC",
    ],
    deps = [
        ":agent_heartbeat",
        ":prediction_service_grpc",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "net_utils",
    srcs = ["net_utils.cc"],
    hdrs = ["net_utils.h"],
)

cc_test(
    name = "net_utils_test",
    srcs = ["net_utils_test.cc"],
    deps = [
        ":net_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "serving_deps_with_framework_shared_object",
    srcs = ["@org_tensorflow//tensorflow:libtensorflow_framework.so.2"],
    deps = [
        "@org_tensorflow//tensorflow/core:distributed_tensorflow_dependencies",
        "@org_tensorflow//tensorflow/core/distributed_runtime/rpc:grpc_runtime",
    ],
)

# Expose monolith ops for tf serving
# we may need to change it to tf_gpu_kernel_library_allow_except later
cc_library(
    name = "serving_ops_cc",
    srcs = [
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":monolith_ops_no_remote_predict",
        ":remote_predict_op",
    ] + select({
        "@org_tensorflow//tensorflow:framework_shared_object": [":serving_deps_with_framework_shared_object"],
        "//conditions:default": [],
    }),
    alwayslink = 1,
)
