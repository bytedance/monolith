load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "kafka_producer",
    srcs = ["kafka_producer.py"],
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "kafka_receiver",
    srcs = ["kafka_receiver.py"],
    deps = [
        "//monolith/native_training:native_model",
    ],
)

py_binary(
    name = "demo_model",
    srcs = ["demo_model.py"],
    deps = [
        ":kafka_producer",
        ":kafka_receiver",
        "//monolith/native_training:native_model",
    ],
)

py_binary(
    name = "demo_local_runner",
    srcs = ["demo_local_runner.py"],
    deps = [
        ":demo_model",
    ],
)
