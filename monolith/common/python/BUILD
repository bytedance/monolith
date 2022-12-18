load("@rules_python//python:defs.bzl", "py_library")

filegroup(
    name = "libtcmalloc",
    srcs = ["@gperftools//:libtcmalloc"],
)

filegroup(
    name = "mem_profiling_internal_deps",
)

py_library(
    name = "mem_profiling",
    srcs = ["mem_profiling.py"],
    data = [
        ":libtcmalloc",
        ":mem_profiling_internal_deps",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//monolith:utils",
    ],
)
