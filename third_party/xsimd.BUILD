package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD 3-Clause

exports_files(["LICENSE"])

cc_library(
    name = "xsimd",
    srcs = [],
    hdrs = glob(
        [
            "include/xsimd/*.hpp",
            "include/xsimd/arch/*.hpp",
            "include/xsimd/arch/generic/*.hpp",
            "include/xsimd/config/*.hpp",
            "include/xsimd/math/*.hpp",
            "include/xsimd/memory/*.hpp",
            "include/xsimd/stl/*.hpp",
            "include/xsimd/types/*.hpp",
        ],
        exclude = [
        ],
    ),
    copts = [],
    defines = [],
    includes = [
        "include",
    ],
    linkopts = [],
    visibility = ["//visibility:public"],
    deps = [
    ],
)