load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

package(default_visibility = ["//monolith/native_training/runtime:__subpackages__"])

cc_library(
    name = "cuckoohash",
    hdrs = [
        "bucket_container.hpp",
        "cuckoohash_config.hpp",
        "cuckoohash_map.hpp",
        "cuckoohash_util.hpp",
    ],
    visibility = [
        "//monolith/feature_engineering/runtime:__subpackages__",
        "//monolith/native_training/runtime:__subpackages__",
    ],
    deps = [
        "//monolith/native_training/runtime/hash_table:embedding_hash_table_interface",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "cuckoo_embedding_hash_table",
    srcs = ["cuckoo_embedding_hash_table.cc"],
    hdrs = ["cuckoo_embedding_hash_table.h"],
    deps = [
        ":cuckoohash",
        "//monolith/native_training/runtime/allocator:block_allocator",
        "//monolith/native_training/runtime/hash_table:embedding_hash_table_cc_proto",
        "//monolith/native_training/runtime/hash_table:embedding_hash_table_interface",
        "//monolith/native_training/runtime/hash_table:entry_accessor",
        "//monolith/native_training/runtime/hash_table:entry_defs",
        # TODO(zhen.li1): refactor the methods in this experimental library to normal
        # library.
        "//monolith/native_training/data/training_instance:reader_util",
    ],
)

cc_test(
    name = "cuckoo_embedding_hash_table_test",
    srcs = ["cuckoo_embedding_hash_table_test.cc"],
    deps = [
        "//monolith/native_training/runtime/hash_table:embedding_hash_table_test",
    ],
)

cc_binary(
    name = "cuckoo_embedding_hash_table_benchmark",
    srcs = ["cuckoo_embedding_hash_table_benchmark.cc"],
    deps = [
        "//monolith/native_training/runtime/concurrency:thread_pool",
        "//monolith/native_training/runtime/hash_table:embedding_hash_table_cc_proto",
        "//monolith/native_training/runtime/hash_table:embedding_hash_table_factory",
        "//monolith/native_training/runtime/hash_table:embedding_hash_table_interface",
        "@com_github_google_benchmark//:benchmark",
        "@com_google_absl//absl/random",
        "@com_google_glog//:glog",
    ],
)
