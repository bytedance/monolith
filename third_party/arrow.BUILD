# Description:
#   Apache Arrow library

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE.txt"])

genrule(
    name = "arrow_util_config",
    srcs = ["cpp/src/arrow/util/config.h.cmake"],
    outs = ["cpp/src/arrow/util/config.h"],
    cmd = ("sed " +
           "-e 's/@ARROW_VERSION_MAJOR@/3/g' " +
           "-e 's/@ARROW_VERSION_MINOR@/0/g' " +
           "-e 's/@ARROW_VERSION_PATCH@/0/g' " +
           "-e 's/cmakedefine ARROW_USE_NATIVE_INT128/undef ARROW_USE_NATIVE_INT128/g' " +
           "-e 's/cmakedefine/define/g' " +
           "$< >$@"),
)

genrule(
    name = "parquet_version_h",
    srcs = ["cpp/src/parquet/parquet_version.h.in"],
    outs = ["cpp/src/parquet/parquet_version.h"],
    cmd = ("sed " +
           "-e 's/@PARQUET_VERSION_MAJOR@/1/g' " +
           "-e 's/@PARQUET_VERSION_MINOR@/5/g' " +
           "-e 's/@PARQUET_VERSION_PATCH@/1/g' " +
           "$< >$@"),
)

cc_library(
    name = "arrow",
    srcs = glob(
        [
            "cpp/src/arrow/*.cc",
            "cpp/src/arrow/array/*.cc",
            "cpp/src/arrow/csv/*.cc",
            "cpp/src/arrow/io/*.cc",
            "cpp/src/arrow/ipc/*.cc",
            "cpp/src/arrow/json/*.cc",
            "cpp/src/arrow/tensor/*.cc",
            "cpp/src/arrow/util/*.cc",
            "cpp/src/arrow/vendored/optional.hpp",
            "cpp/src/arrow/vendored/string_view.hpp",
            "cpp/src/arrow/vendored/variant.hpp",
            "cpp/src/arrow/**/*.h",
            "cpp/src/parquet/**/*.h",
            "cpp/src/parquet/**/*.cc",
            "cpp/src/generated/*.h",
            "cpp/src/generated/*.cpp",
            "cpp/thirdparty/flatbuffers/include/flatbuffers/*.h",
            "cpp/src/arrow/vendored/uriparser/*.c",
            "cpp/src/arrow/vendored/base64.cpp",
            "cpp/src/arrow/compute/**/*.cc",
            "cpp/src/arrow/vendored/datetime/*.h",
            "cpp/src/arrow/vendored/datetime/*.cpp",
            "cpp/src/arrow/vendored/pcg/*.hpp",
        ],
        exclude = [
            "cpp/src/**/*_benchmark.cc",
            "cpp/src/**/*_main.cc",
            "cpp/src/**/*_nossl.cc",
            "cpp/src/**/*_test.cc",
            "cpp/src/**/test_*.cc",
            "cpp/src/**/*hdfs*.cc",
            "cpp/src/**/*fuzz*.cc",
            "cpp/src/**/file_to_stream.cc",
            "cpp/src/**/stream_to_file.cc",
            "cpp/src/arrow/util/bpacking_avx2.cc",
            "cpp/src/arrow/util/bpacking_avx512.cc",
            "cpp/src/arrow/util/bpacking_neon.cc",
            "cpp/src/arrow/util/tracing_internal.cc",
        ],
    ) + select({
        "@bazel_tools//src/conditions:windows": [
            "cpp/src/arrow/vendored/musl/strptime.c",
        ],
        "//conditions:default": [],
    }),
    hdrs = [
        # declare header from above genrule
        "cpp/src/arrow/util/config.h",
        "cpp/src/parquet/parquet_version.h",
    ],
    copts = select({
        "@bazel_tools//src/conditions:windows": [
            "/std:c++14",
        ],
        "//conditions:default": [
            "-std=c++14",
        ],
    }),
    defines = [
        "ARROW_WITH_BROTLI",
        "ARROW_WITH_SNAPPY",
        "ARROW_WITH_LZ4",
        "ARROW_WITH_ZLIB",
        "ARROW_WITH_ZSTD",
        "ARROW_WITH_BZ2",
        "ARROW_STATIC",
        "ARROW_EXPORT=",
        "PARQUET_STATIC",
        "PARQUET_EXPORT=",
        "WIN32_LEAN_AND_MEAN",
    ],
    includes = [
        "cpp/src",
        "cpp/src/arrow/vendored/xxhash",
        "cpp/thirdparty/flatbuffers/include",
    ],
    textual_hdrs = [
        "cpp/src/arrow/vendored/xxhash/xxhash.c",
    ],
    deps = [
        "@boringssl//:crypto",
        "@brotli",
        "@bzip2",
        "@double_conversion//:double-conversion",
        "@lz4",
        "@rapidjson",
        "@snappy//:snappy",
        "@thrift",
        "@xsimd",
        "@zlib",
        "@zstd",
    ],
)