# Description:
#   Brotli library

licenses(["notice"])  # MIT license

exports_files(["LICENSE"])

cc_library(
    name = "brotli",
    srcs = glob([
        "c/common/*.c",
        "c/common/*.h",
        "c/dec/*.c",
        "c/dec/*.h",
        "c/enc/*.c",
        "c/enc/*.h",
        "c/include/brotli/*.h",
    ]),
    hdrs = [],
    defines = [],
    includes = [
        "c/dec",
        "c/include",
    ],
    linkopts = [],
    visibility = ["//visibility:public"],
)