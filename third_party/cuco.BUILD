# May be deprecated later
licenses(["notice"])

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts")

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

cc_library(
    name = "cuco_hash_table",
    hdrs = glob(["include/**"]),
    include_prefix = "third_party/cuco_hash_table",
    strip_include_prefix = "include",
    includes = [
        "include",
    ],
    copts = [
        "-std=c++17",
    ],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
    ],
)