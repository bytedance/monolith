load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library")
load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("@rules_proto//proto:defs.bzl", "proto_library")

proto_library(
    name = "proto_parser_proto",
    srcs = [
        "matrix/proto/feature.proto",
        "matrix/proto/line_id.proto",
        "matrix/proto/proto_parser.proto",
    ],
    visibility = ["//visibility:public"],
)

cc_proto_library(
    name = "line_id_cc_proto",
    srcs = [
        "matrix/proto/line_id.proto",
    ],
    visibility = ["//visibility:public"],
)

py_proto_library(
    name = "line_id_py_proto",
    srcs = [
        "matrix/proto/line_id.proto",
    ],
    visibility = ["//visibility:public"],
)

cc_proto_library(
    name = "proto_parser_cc_proto",
    srcs = [
        "matrix/proto/feature.proto",
        "matrix/proto/proto_parser.proto",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":line_id_cc_proto",
    ],
)

py_proto_library(
    name = "proto_parser_py_proto",
    srcs = [
        "matrix/proto/feature.proto",
        "matrix/proto/proto_parser.proto",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":line_id_py_proto",
    ],
)

cc_library(
    name = "compression_cc_float16",
    hdrs = ["matrix/compression/float16.h"],
    visibility = ["//visibility:public"],
    deps = ["//third_party/half_sourceforge_net:half"],
)

cc_library(
    name = "compression",
    srcs = ["matrix/compression/compression.cc"],
    hdrs = [
        "matrix/compression/compression.h",
        "matrix/compression/compression_qtz8mm.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":compression_cc_float16",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "compression_qtz8mm",
    srcs = ["matrix/compression/compression_qtz8mm.cc"],
    hdrs = ["matrix/compression/compression_qtz8mm.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":compression",
        ":compression_cc_float16",
        "@com_google_glog//:glog",
    ],
)


cc_proto_library(
    name = "example_cc_proto",
    srcs = [
        "matrix/proto/example.proto",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":line_id_cc_proto",
    ],
)

py_proto_library(
    name = "example_py_proto",
    srcs = [
        "matrix/proto/example.proto",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":line_id_py_proto",
    ],
)

