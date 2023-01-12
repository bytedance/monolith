package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD 3-Clause

exports_files(["COPYING"])

cc_library(
    name = "snappy",
    srcs = glob(
        [
            "*.cc",
            "*.h",
        ],
        exclude = [
            "*test.*",
            "*fuzzer.*",
        ],
    ),
    hdrs = [
        "snappy-stubs-public.h",
    ],
    copts = [],
    includes = ["."],
)

genrule(
    name = "snappy_stubs_public_h",
    srcs = ["snappy-stubs-public.h.in"],
    outs = ["snappy-stubs-public.h"],
    cmd = ("sed " +
           "-e 's/$${HAVE_SYS_UIO_H_01}/HAVE_SYS_UIO_H/g' " +
           "-e 's/$${PROJECT_VERSION_MAJOR}/1/g' " +
           "-e 's/$${PROJECT_VERSION_MINOR}/1/g' " +
           "-e 's/$${PROJECT_VERSION_PATCH}/8/g' " +
           "$< >$@"),
)
