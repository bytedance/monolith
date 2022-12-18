load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "dpssdk",
    srcs = glob(["dpstoken/*.py"]),
    imports = ["dpstoken/"],
    visibility = ["//visibility:public"],
    deps= [requirement("cryptography")]
)