load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")
load("@pip_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_custom_op_library")

package(
    default_visibility = [
        "//monolith:__subpackages__",
    ],
)

py_library(
    name = "monolith_export",
    srcs = ["monolith_export.py"],
    visibility = ["//visibility:public"],
)

py_binary(
    name = "demo",
    srcs = ["demo.py"],
    deps = [
        ":cpu_training",
        ":model",
        ":native_task",
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        ":feature",
        ":input",
        ":native_task",
        "//monolith/core:base_model_params",
        "//monolith/core:model_registry",
        "//monolith/native_training/metric:deep_insight_ops",
    ],
)

py_library(
    name = "input",
    srcs = ["input.py"],
)

py_library(
    name = "test_utils",
    testonly = 1,
    srcs = ["test_utils.py"],
    deps = [
        ":entry",
        ":utils",
        "//monolith/native_training/runtime/hash_table:embedding_hash_table_py_proto",
    ],
)

py_library(
    name = "clip_ops",
    srcs = ["clip_ops.py"],
    deps = [
        ":device_utils",
        "//monolith:utils",
        "//monolith/native_training/runtime/ops:gen_monolith_ops",
    ],
)

py_test(
    name = "clip_ops_test",
    srcs = ["clip_ops_test.py"],
    deps = [
        ":clip_ops",
    ],
)

py_proto_library(
    name = "hash_table_ops_py_proto",
    srcs = ["hash_table_ops.proto"],
)

py_library(
    name = "hash_table_ops",
    srcs = ["hash_table_ops.py"],
    deps = [
        ":basic_restore_hook",
        ":distributed_serving_ops",
        ":entry",
        ":feature",
        ":graph_meta",
        ":hash_filter_ops",
        ":hash_table_ops_py_proto",
        ":hash_table_utils",
        ":save_utils",
        ":utils",
        "//monolith:utils",
        "//monolith/native_training/model_export:export_context",
        "//monolith/native_training/runtime/hash_table:embedding_hash_table_py_proto",
        "//monolith/native_training/runtime/ops:gen_monolith_ops",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "hash_table_ops_test",
    srcs = ["hash_table_ops_test.py"],
    deps = [
        ":hash_filter_ops",
        ":hash_table_ops",
        ":learning_rate_functions",
    ],
)

py_test(
    name = "gpu_hash_table_ops_test",
    srcs = ["gpu_hash_table_ops_test.py"],
    deps = [
        ":hash_filter_ops",
        ":hash_table_ops",
        ":learning_rate_functions",
    ],
)

py_proto_library(
    name = "multi_hash_table_ops_py_proto",
    srcs = ["multi_hash_table_ops.proto"],
)

py_library(
    name = "multi_hash_table_ops",
    srcs = ["multi_hash_table_ops.py"],
    deps = [
        ":basic_restore_hook",
        ":distributed_serving_ops",
        ":entry",
        ":feature",
        ":graph_meta",
        ":hash_filter_ops",
        ":hash_table_utils",
        ":multi_hash_table_ops_py_proto",
        ":multi_type_hash_table",
        ":save_utils",
        ":utils",
        "//monolith:utils",
        "//monolith/native_training/model_export:export_context",
        "//monolith/native_training/proto:ckpt_info_py_proto",
        "//monolith/native_training/runtime/hash_table:embedding_hash_table_py_proto",
        "//monolith/native_training/runtime/ops:gen_monolith_ops",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "multi_hash_table_ops_test",
    srcs = ["multi_hash_table_ops_test.py"],
    deps = [
        ":hash_filter_ops",
        ":learning_rate_functions",
        ":multi_hash_table_ops",
        ":test_utils",
        ":utils",
    ],
)

py_binary(
    name = "hash_table_utils",
    srcs = ["hash_table_utils.py"],
)

py_test(
    name = "hash_table_utils_test",
    srcs = ["hash_table_utils_test.py"],
    deps = [
        ":hash_table_ops",
        ":hash_table_utils",
    ],
)

py_binary(
    name = "hash_table_ops_benchmark",
    srcs = ["hash_table_ops_benchmark.py"],
    deps = [
        ":hash_filter_ops",
        ":hash_table_ops",
    ],
)

py_library(
    name = "distribution_ops",
    srcs = ["distribution_ops.py"],
    deps = [
        "//idl:example_py_proto",
        "//monolith:utils",
        "//monolith/native_training/runtime/ops:gen_monolith_ops",
    ],
)

py_test(
    name = "distribution_ops_test",
    srcs = ["distribution_ops_test.py"],
    deps = [
        ":distribution_ops",
    ],
)

py_test(
    name = "fused_embedding_to_layout_test",
    srcs = ["fused_embedding_to_layout_test.py"],
    deps = [
        ":distribution_ops",
        "//monolith/native_training/data:parsers_py",
    ],
)

py_test(
    name = "distribution_ops_fused_test",
    srcs = ["distribution_ops_fused_test.py"],
    deps = [
        ":distribution_ops",
    ],
)

py_binary(
    name = "distribution_ops_benchmark",
    srcs = ["distribution_ops_benchmark.py"],
    deps = [
        ":distribution_ops",
    ],
)

py_binary(
    name = "distribution_ops_fused_benchmark",
    srcs = ["distribution_ops_fused_benchmark.py"],
    deps = [
        ":distribution_ops",
    ],
)

py_library(
    name = "distributed_ps",
    srcs = ["distributed_ps.py"],
    deps = [
        ":distribution_ops",
        ":hash_table_ops",
        ":hash_table_utils",
        ":logging_ops",
        ":multi_type_hash_table",
        ":native_task_context",
        ":tensor_utils",
        ":utils",
        "//monolith/native_training/data:parsers_py",
    ],
)

py_test(
    name = "distributed_ps_test",
    srcs = ["distributed_ps_test.py"],
    deps = [
        ":distributed_ps",
        ":distributed_ps_factory",
        ":hash_filter_ops",
        ":learning_rate_functions",
        ":multi_hash_table_ops",
        ":test_utils",
        "//monolith/native_training/data:feature_utils_py",
    ],
)

py_library(
    name = "distributed_ps_factory",
    srcs = ["distributed_ps_factory.py"],
    deps = [
        ":distributed_ps",
        ":distributed_ps_sync",
        ":distribution_ops",
        ":embedding_combiners",
        ":entry",
        ":hash_filter_ops",
        ":hash_table_ops",
        ":multi_hash_table_ops",
        ":multi_type_hash_table",
        ":utils",
    ],
)

py_test(
    name = "distributed_ps_factory_test",
    srcs = ["distributed_ps_factory_test.py"],
    deps = [
        ":distributed_ps_factory",
        ":hash_filter_ops",
        ":test_utils",
    ],
)

py_binary(
    name = "distributed_ps_benchmark",
    srcs = ["distributed_ps_benchmark.py"],
    deps = [
        ":distributed_ps",
        ":hash_filter_ops",
        ":multi_type_hash_table",
        "//monolith/native_training/runtime/hash_table:embedding_hash_table_py_proto",
    ],
)

py_library(
    name = "embedding_combiners",
    srcs = ["embedding_combiners.py"],
    deps = [
        ":device_utils",
        ":distribution_ops",
        ":ragged_utils",
    ],
)

py_test(
    name = "embedding_combiners_test",
    srcs = ["embedding_combiners_test.py"],
    deps = [
        ":embedding_combiners",
    ],
)

py_library(
    name = "distributed_ps_sync",
    srcs = ["distributed_ps_sync.py"],
    deps = [
        ":distributed_ps",
        ":distribution_ops",
        ":multi_type_hash_table",
        ":prefetch_queue",
    ],
)

py_test(
    name = "distributed_ps_sync_test",
    srcs = ["distributed_ps_sync_test.py"],
    deps = [
        ":distributed_ps",
        ":distributed_ps_sync",
        ":learning_rate_functions",
        ":multi_hash_table_ops",
        ":test_utils",
    ],
)

py_library(
    name = "multi_type_hash_table",
    srcs = ["multi_type_hash_table.py"],
    deps = [
        ":device_utils",
        ":distribution_ops",
        ":entry",
        ":hash_filter_ops",
        ":hash_table_ops",
        ":hash_table_utils",
        ":prefetch_queue",
    ],
)

py_test(
    name = "multi_type_hash_table_test",
    srcs = ["multi_type_hash_table_test.py"],
    deps = [
        ":hash_filter_ops",
        ":hash_table_ops",
        ":learning_rate_functions",
        ":multi_type_hash_table",
        ":test_utils",
        ":utils",
    ],
)

py_library(
    name = "hash_filter_ops",
    srcs = ["hash_filter_ops.py"],
    deps = [
        ":basic_restore_hook",
        ":save_utils",
        ":utils",
        "//monolith/native_training/model_export:export_context",
        "//monolith/native_training/runtime/ops:gen_monolith_ops",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "hash_filter_ops_test",
    srcs = ["hash_filter_ops_test.py"],
    deps = [
        ":hash_filter_ops",
        "//monolith/native_training/runtime/hash_table:embedding_hash_table_py_proto",
    ],
)

py_library(
    name = "touched_key_set_ops",
    srcs = ["touched_key_set_ops.py"],
    deps = [
        "//monolith/native_training/runtime/ops:gen_monolith_ops",
    ],
)

py_test(
    name = "touched_key_set_ops_test",
    srcs = ["touched_key_set_ops_test.py"],
    deps = [":touched_key_set_ops"],
)

py_library(
    name = "distributed_serving_ops",
    srcs = ["distributed_serving_ops.py"],
    deps = [
        "//monolith/agent_service:agent",
        "//monolith/native_training/runtime/ops:gen_monolith_ops",
        "//monolith/native_training/runtime/parameter_sync:parameter_sync_py_proto",
    ],
)

py_test(
    name = "distributed_serving_ops_test",
    srcs = ["distributed_serving_ops_test.py"],
    deps = [
        ":distributed_serving_ops",
        ":hash_table_ops",
    ],
)

py_library(
    name = "entry",
    srcs = ["entry.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":monolith_export",
        "//monolith:utils",
        "//monolith/native_training/runtime/hash_table:embedding_hash_table_py_proto",
    ],
)

py_test(
    name = "entry_test",
    srcs = ["entry_test.py"],
    deps = [
        ":entry",
        ":learning_rate_functions",
    ],
)

py_library(
    name = "feature",
    srcs = ["feature.py"],
    visibility = ["//visibility:public"],
    deps = [
        "embedding_combiners",
        ":device_utils",
        ":distribution_ops",
        ":entry",
        ":learning_rate_functions",
        ":monolith_export",
        ":ragged_utils",
        "//monolith:utils",
        "//monolith/native_training/model_export:export_context",
        "//monolith/native_training/runtime/hash_table:embedding_hash_table_py_proto",
    ],
)

py_library(
    name = "distribution_utils",
    srcs = ["distribution_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "feature_utils",
    srcs = ["feature_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":clip_ops",
        ":feature",
        ":native_task",
        ":prefetch_queue",
    ],
)

py_test(
    name = "feature_utils_test",
    srcs = ["feature_utils_test.py"],
    deps = [
        ":feature_utils",
        ":prefetch_queue",
    ],
)

py_test(
    name = "feature_test",
    srcs = ["feature_test.py"],
    deps = [
        ":feature",
        ":learning_rate_functions",
    ],
)

py_library(
    name = "native_task",
    srcs = ["native_task.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":feature",
        ":prefetch_queue",
        "//monolith/core:base_task",
        "//monolith/native_training/model_export:export_context",
    ],
)

py_library(
    name = "native_model",
    srcs = ["native_model.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":dense_reload_utils",
        ":entry",
        ":estimator",
        ":feature",
        ":file_ops",
        ":monolith_export",
        ":native_task",
        ":native_task_context",
        ":utils",
        "//monolith:utils",
        "//monolith/core:base_layer",
        "//monolith/native_training:feature_utils",
        "//monolith/native_training/data:feature_list",
        "//monolith/native_training/data:utils",
        "//monolith/native_training/layers",
        "//monolith/native_training/metric:metric_hook",
        "//monolith/native_training/metric:utils",
        "//monolith/native_training/model_dump:dump_utils",
    ],
)

py_library(
    name = "cpu_training_additional_deps",
)

py_library(
    name = "cpu_training",
    srcs = ["cpu_training.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":barrier_ops",
        ":basic_restore_hook",
        ":cluster_manager",
        ":cpu_training_additional_deps",
        ":device_utils",
        ":distributed_ps_factory",
        ":distribution_ops",
        ":distribution_utils",
        ":env_utils",
        ":feature",
        ":hash_table_ops",
        ":hash_table_utils",
        ":logging_ops",
        ":multi_type_hash_table",
        ":native_task",
        ":native_task_context",
        ":net_utils",
        ":prefetch_queue",
        ":ps_benchmark",
        ":save_utils",
        ":service_discovery",
        ":session_run_hooks",
        ":signal_utils",
        ":sync_hooks",
        ":sync_training_hooks",
        ":tensor_utils",
        ":utils",
        ":variables",
        ":yarn_runtime",
        "//monolith/agent_service:replica_manager",
        "//monolith/core:hyperparams",
        "//monolith/native_training:dense_reload_utils",
        "//monolith/native_training:hash_filter_ops",
        "//monolith/native_training/alert",
        "//monolith/native_training/data/training_instance:parser_utils",
        "//monolith/native_training/hooks:ckpt_hooks",
        "//monolith/native_training/hooks:ckpt_info",
        "//monolith/native_training/hooks:controller_hooks",
        "//monolith/native_training/hooks:feature_engineering_hooks",
        "//monolith/native_training/hooks:hook_utils",
        "//monolith/native_training/hooks:ps_check_hooks",
        "//monolith/native_training/hooks:session_hooks",
        "//monolith/native_training/metric:deep_insight_ops",
        "//monolith/native_training/metric:metric_hook",
        "//monolith/native_training/model_dump:dump_utils",
        "//monolith/native_training/model_export",
        "//monolith/native_training/model_export:export_context",
        "//monolith/native_training/model_export:export_hooks",
        "//monolith/native_training/model_export:export_utils",
        "//monolith/native_training/model_export:saved_model_exporters",
        "//monolith/native_training/proto:debugging_info_py_proto",
        "//monolith/native_training/runtime/hash_table:embedding_hash_table_py_proto",
    ],
)

py_test(
    name = "cpu_training_test",
    size = "large",
    srcs = ["cpu_training_test.py"],
    data = ["cpu_training_distributed_test_binary"],
    shard_count = 5,
    deps = [
        ":cpu_training",
        ":native_task",
        ":service_discovery",
        "//monolith/native_training/debugging:debugging_server",
    ],
)

py_test(
    name = "cpu_training_multi_hash_table_test",
    size = "large",
    srcs = ["cpu_training_test.py"],
    args = ["--use_native_multi_hash_table"],
    main = "cpu_training_test.py",
    shard_count = 5,
    deps = [
        ":cpu_training_test",
    ],
)

py_binary(
    name = "cpu_training_distributed_test_binary",
    srcs = ["cpu_training_distributed_test_binary.py"],
    deps = [
        ":cluster_manager",
        ":cpu_training",
        ":native_task",
        ":service_discovery",
    ],
)

py_test(
    name = "cpu_sync_training_test",
    srcs = ["cpu_sync_training_test.py"],
    data = ["@org_tensorflow//tensorflow:libtensorflow_framework.so.2"],
    deps = [
        ":cpu_training",
        ":native_task",
    ],
)

py_library(
    name = "native_task_context",
    srcs = ["native_task_context.py"],
    deps = ["//monolith/agent_service:backends"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "//idl:proto_parser_py_proto",
        "//monolith/core:base_layer",
        "//monolith/core:hyperparams",
        "//monolith/core:py_utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    deps = [":utils"],
)

py_library(
    name = "file_ops",
    srcs = ["file_ops.py"],
    deps = [
        "//monolith/native_training/runtime/ops:gen_monolith_ops",
    ],
)

py_test(
    name = "file_ops_test",
    srcs = ["file_ops_test.py"],
    deps = [
        ":file_ops",
        "//monolith:utils",
    ],
)

py_library(
    name = "env_utils",
    srcs = ["env_utils.py"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "env_utils_test",
    srcs = ["env_utils_test.py"],
    deps = [
        ":env_utils",
    ],
)

py_library(
    name = "logging_ops",
    srcs = ["logging_ops.py"],
    deps = [
        "//monolith:utils",
        "//monolith/native_training/runtime/ops:gen_monolith_ops",
        "//monolith/native_training/runtime/ops:logging_ops_py_proto",
    ],
)

py_test(
    name = "logging_ops_test",
    srcs = ["logging_ops_test.py"],
    deps = [
        ":logging_ops",
    ],
)

py_library(
    name = "session_run_hooks",
    srcs = ["session_run_hooks.py"],
)

py_test(
    name = "session_run_hooks_test",
    srcs = ["session_run_hooks_test.py"],
    deps = [
        ":session_run_hooks",
        requirement("freezegun"),
    ],
)

py_library(
    name = "static_reshape_op",
    srcs = ["static_reshape_op.py"],
    deps = [
        "//monolith/native_training/runtime/ops:gen_monolith_ops",
    ],
)

py_test(
    name = "static_reshape_op_test",
    srcs = ["static_reshape_op_test.py"],
    deps = [
        ":static_reshape_op",
    ],
)

py_library(
    name = "sync_training_hooks_addtional_deps",
)

py_library(
    name = "sync_training_hooks",
    srcs = ["sync_training_hooks.py"],
    deps = [
        ":distributed_serving_ops",
        ":sync_training_hooks_addtional_deps",
        "//monolith/agent_service:backends",
    ],
)

py_library(
    name = "service_discovery",
    srcs = ["service_discovery.py"],
    deps = [
        ":consul",
        ":zk_utils",
    ],
)

py_test(
    name = "service_discovery_test",
    srcs = ["service_discovery_test.py"],
    deps = [
        ":service_discovery",
    ],
)

py_library(
    name = "ragged_utils",
    srcs = ["ragged_utils.py"],
    deps = [
        "//monolith/native_training/runtime/ops:gen_monolith_ops",
    ],
)

py_test(
    name = "ragged_utils_test",
    srcs = ["ragged_utils_test.py"],
    deps = [
        ":ragged_utils",
    ],
)

py_library(
    name = "tensor_utils",
    srcs = ["tensor_utils.py"],
    deps = [
        ":static_reshape_op",
    ],
)

py_test(
    name = "tensor_utils_test",
    srcs = ["tensor_utils_test.py"],
    deps = [
        ":tensor_utils",
    ],
)

py_library(
    name = "graph_meta",
    srcs = ["graph_meta.py"],
)

py_library(
    name = "consul",
    srcs = ["consul.py"],
)

py_test(
    name = "consul_test",
    srcs = ["consul_test.py"],
    deps = [
        ":consul",
    ],
)


py_library(
    name = "barrier_ops",
    srcs = ["barrier_ops.py"],
    deps = [
        ":basic_restore_hook",
    ],
)

py_test(
    name = "barrier_ops_test",
    srcs = ["barrier_ops_test.py"],
    deps = [
        ":barrier_ops",
    ],
)

py_library(
    name = "basic_restore_hook",
    srcs = ["basic_restore_hook.py"],
)

py_test(
    name = "basic_restore_hook_test",
    srcs = ["basic_restore_hook_test.py"],
    deps = [
        ":basic_restore_hook",
    ],
)

py_library(
    name = "prefetch_queue",
    srcs = ["prefetch_queue.py"],
    deps = [
        ":nested_tensors",
        ":utils",
    ],
)

py_test(
    name = "prefetch_queue_test",
    srcs = ["prefetch_queue_test.py"],
    deps = [
        ":prefetch_queue",
    ],
)

py_proto_library(
    name = "monolith_checkpoint_state_proto",
    srcs = [":monolith_checkpoint_state.proto"],
)

py_library(
    name = "save_utils",
    srcs = ["save_utils.py"],
    deps = [
        ":dense_reload_utils",
        ":monolith_checkpoint_state_proto",
        ":session_run_hooks",
        ":utils",
        "//monolith/native_training:native_task_context",
        "//monolith/native_training/metric:cli",
    ],
)

py_test(
    name = "save_utils_test",
    srcs = ["save_utils_test.py"],
    deps = [
        ":save_utils",
        requirement("freezegun"),
    ],
)

py_library(
    name = "sync_hooks",
    srcs = ["sync_hooks.py"],
)

py_test(
    name = "sync_hooks_test",
    srcs = ["sync_hooks_test.py"],
    deps = [
        ":sync_hooks",
    ],
)

py_test(
    name = "restore_test",
    srcs = ["restore_test.py"],
    deps = [
        ":hash_table_ops",
        ":save_utils",
        ":utils",
    ],
)

py_library(
    name = "variables",
    srcs = ["variables.py"],
    deps = [
        ":graph_meta",
    ],
)

py_test(
    name = "variables_test",
    srcs = ["variables_test.py"],
    deps = [
        ":test_utils",
        ":variables",
    ],
)

py_library(
    name = "ps_benchmark",
    srcs = ["ps_benchmark.py"],
    deps = [
        ":logging_ops",
        ":native_task",
        ":utils",
        "//monolith/native_training/optimizers:adamom",
    ],
)

py_test(
    name = "ps_benchmark_test",
    srcs = ["ps_benchmark_test.py"],
    deps = [
        ":cpu_training",
        ":ps_benchmark",
    ],
)

py_library(
    name = "estimator",
    srcs = ["estimator.py"],
    deps = [
        ":cpu_training",
        ":env_utils",
        ":monolith_export",
        ":native_task",
        ":runner_utils",
        ":service_discovery",
        ":zk_utils",
        "//monolith:utils",
        "//monolith/agent_service:backends",
        "//monolith/agent_service:replica_manager",
        "//monolith/agent_service:utils",
        "//monolith/native_training/data:item_pool_hook",
        "//monolith/native_training/model_export:saved_model_exporters",
    ],
)

# This test is buggy.
py_test(
    name = "estimator_test",
    srcs = ["estimator_test.py"],
    deps = [
        ":cpu_training",
        ":estimator",
        ":input",
        ":model",
        ":native_task",
        ":service_discovery",
        ":utils",
        "//monolith/native_training/data/training_instance:instance_dataset_ops_py",
        "//monolith/native_training/data/training_instance:parse_instance_ops_py",
        "//monolith/native_training/model_export:saved_model_exporters",
    ],
)

# This test is buggy.
# py_test(
#     name = "estimator_dist_test",
#     srcs = ["estimator_dist_test.py"],
#     deps = [
#         ":cpu_training",
#         ":estimator",
#         ":input",
#         ":model",
#         ":native_task",
#         ":service_discovery",
#         ":utils",
#         "//monolith/native_training/data/training_instance:instance_dataset_ops_py",
#         "//monolith/native_training/data/training_instance:parse_instance_ops_py",
#         "//monolith/native_training/model_export:saved_model_exporters",
#         "//monolith/native_training/tasks/reckon:params",
#     ],
# )

py_library(
    name = "zk_utils",
    srcs = ["zk_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":env_utils",
        requirement("kazoo"),
    ],
)


py_library(
    name = "device_utils",
    srcs = ["device_utils.py"],
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "device_utils_test",
    srcs = ["device_utils_test.py"],
    deps = [
        ":device_utils",
    ],
)

py_library(
    name = "gflags_utils",
    srcs = ["gflags_utils.py"],
)

py_test(
    name = "gflags_utils_test",
    srcs = ["gflags_utils_test.py"],
    deps = [
        ":cpu_training",
        ":gflags_utils",
        ":runner_utils",
    ],
)

py_library(
    name = "runner_utils",
    srcs = ["runner_utils.py"],
    deps = [
        ":cpu_training",
        ":env_utils",
        ":gflags_utils",
        ":monolith_checkpoint_state_proto",
        ":service_discovery",
    ],
)

py_test(
    name = "runner_utils_test",
    srcs = ["runner_utils_test.py"],
    deps = [
        ":runner_utils",
    ],
)

py_library(
    name = "learning_rate_functions",
    srcs = ["learning_rate_functions.py"],
)

py_test(
    name = "learning_rate_functions_test",
    srcs = ["learning_rate_functions_test.py"],
    deps = [
        ":learning_rate_functions",
    ],
)

py_library(
    name = "yarn_runtime",
    srcs = ["yarn_runtime.py"],
    deps = [
        "//monolith/native_training/proto:primus_am_service_py_proto",
        "//monolith/native_training/proto:primus_am_service_py_proto_grpc",
    ],
)

py_test(
    name = "yarn_runtime_test",
    srcs = ["yarn_runtime_test.py"],
    deps = [
        "yarn_runtime",
    ],
)

py_library(
    name = "net_utils",
    srcs = ["net_utils.py"],
    deps = [
        requirement("netifaces"),
        "//monolith:utils",
    ],
)

py_test(
    name = "net_utils_test",
    srcs = ["net_utils_test.py"],
    deps = [
        ":net_utils",
    ],
)

py_library(
    name = "cluster_manager",
    srcs = ["cluster_manager.py"],
    deps = [
        ":service_discovery",
        "//monolith/native_training/metric:cli",
    ],
)

py_test(
    name = "cluster_manager_test",
    srcs = ["cluster_manager_test.py"],
    deps = [
        ":cluster_manager",
    ],
)

py_library(
    name = "signal_utils",
    srcs = ["signal_utils.py"],
)

py_test(
    name = "signal_utils_test",
    srcs = ["signal_utils_test.py"],
    deps = [
        ":signal_utils",
    ],
)

py_library(
    name = "gen_seq_mask",
    srcs = ["gen_seq_mask.py"],
    deps = [
        "//monolith/native_training/runtime/ops:gen_monolith_ops",
    ],
)

py_test(
    name = "gen_seq_mask_test",
    srcs = ["gen_seq_mask_test.py"],
    deps = [
        ":gen_seq_mask",
        "//monolith:utils",
    ],
)

py_test(
    name = "serving_ps_test",
    srcs = ["serving_ps_test.py"],
    data = ["//idl:example_cc_proto"],
    deps = [
        ":distribution_ops",
        "//idl:example_py_proto",
        "//monolith:utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "dense_reload_utils",
    srcs = ["dense_reload_utils.py"],
    deps = [
        ":basic_restore_hook",
        "//monolith/native_training/model_export:export_context",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "dense_reload_utils_test",
    srcs = ["dense_reload_utils_test.py"],
    deps = [
        ":dense_reload_utils",
        "//monolith:utils",
    ],
)

py_library(
    name = "nested_tensors",
    srcs = ["nested_tensors.py"],
)

py_test(
    name = "nested_tensors_test",
    srcs = ["nested_tensors_test.py"],
    deps = [
        ":nested_tensors",
    ],
)
