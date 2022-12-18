
py_library(
    name = "tlearner_arch",
    deps = [
        ":ream",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "ream",
    deps = [
        ":client",
        ":common",
    ],
)

py_library(
    name = "client",
    srcs = ["ream/client/ream_client.py"]
)

py_library(
    name = "common",
    deps = [
        ":constant",
    ],
)

py_library(
    name = "constant",
    srcs = ["ream/common/constant.py"]
)