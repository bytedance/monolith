# Description:
#   Apache Thrift library

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "thrift",
    srcs = glob([
        "lib/cpp/src/thrift/**/*.h",
    ]) + [
        "lib/cpp/src/thrift/protocol/TProtocol.cpp",
        "lib/cpp/src/thrift/transport/TBufferTransports.cpp",
        "lib/cpp/src/thrift/transport/TTransportException.cpp",
    ],
    hdrs = [
        "compiler/cpp/src/thrift/version.h",
        "lib/cpp/src/thrift/config.h",
    ],
    includes = [
        "lib/cpp/src",
    ],
    textual_hdrs = [
        "lib/cpp/src/thrift/protocol/TBinaryProtocol.tcc",
        "lib/cpp/src/thrift/protocol/TCompactProtocol.tcc",
    ],
    deps = [
        "@boost//:any",
        "@boost//:algorithm",
        "@boost//:assert",
        "@boost//:bind",
        "@boost//:call_traits",
        "@boost//:config",
        "@boost//:core",
        "@boost//:current_function",
        "@boost//:detail",
        "@boost//:exception",
        "@boost//:function",
        "@boost//:io",
        "@boost//:iterator",
        "@boost//:limits",
        "@boost//:mpl",
        "@boost//:numeric_conversion",
        "@boost//:optional",
        "@boost//:preprocessor",
        "@boost//:smart_ptr",
        "@boost//:static_assert",
        "@boost//:timer",
        "@boost//:type",
        "@boost//:type_traits",
        "@boost//:utility",
        "@boost//:version",
        "@boost//:multiprecision"
    ],
)

genrule(
    name = "version_h",
    srcs = [
        "compiler/cpp/src/thrift/version.h.in",
    ],
    outs = [
        "compiler/cpp/src/thrift/version.h",
    ],
    cmd = "sed 's/@PACKAGE_VERSION@/0.12.0/g' $< > $@",
)

genrule(
    name = "config_h",
    srcs = ["build/cmake/config.h.in"],
    outs = ["lib/cpp/src/thrift/config.h"],
    cmd = ("sed " +
           "-e 's/cmakedefine/define/g' " +
           "-e 's/$${PACKAGE}/thrift/g' " +
           "-e 's/$${PACKAGE_BUGREPORT}//g' " +
           "-e 's/$${PACKAGE_NAME}/thrift/g' " +
           "-e 's/$${PACKAGE_TARNAME}/thrift/g' " +
           "-e 's/$${PACKAGE_URL}//g' " +
           "-e 's/$${PACKAGE_VERSION}/0.12.0/g' " +
           "-e 's/$${PACKAGE_STRING}/thrift 0.12.0/g' " +
           "$< >$@"),
)