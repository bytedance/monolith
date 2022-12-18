load("@rules_cc//cc:defs.bzl", "cc_library", "cc_proto_library", "cc_test")

package(default_visibility = ["//monolith/native_training/runtime:__subpackages__"])

cc_library(
    name = "hopscotch_hash_set",
    srcs = ["hopscotch_hash_set.cc"],
    hdrs = ["hopscotch_hash_set.h"],
    deps = [
        "//monolith/native_training/runtime/concurrency:micro_one_bit_spin_lock",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/synchronization",
    ],
)

# diable this test since it is not runnable on TCE image.
# cc_test(
#     name = "hopscotch_hash_set_test",
#     srcs = ["hopscotch_hash_set_test.cc"],
#     deps = [
#         ":hopscotch_hash_set",
#         "@gperftools//:libtcmalloc",
#         "@com_google_googletest//:gtest_main",
#     ],
# )
