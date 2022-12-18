load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_visibility = [
        "//monolith/native_training:__subpackages__",
        "//monolith/native_training/data/training_instance:__subpackages__",
    ],
)

py_library(
    name = "str_queue",
    srcs = ["str_queue.py"],
)

py_test(
    name = "str_queue_test",
    srcs = ["str_queue_test.py"],
    deps = [":str_queue"],
)

py_library(
    name = "distributed_dataset",
    srcs = ["distributed_dataset.py"],
    deps = [
        ":str_queue",
        "//monolith/native_training:native_task_context",
        "//monolith/native_training:utils",
        "//monolith/native_training/hooks:session_hooks",
    ],
)

py_test(
    name = "distributed_dataset_test",
    srcs = ["distributed_dataset_test.py"],
    deps = [
        ":distributed_dataset",
    ],
)
