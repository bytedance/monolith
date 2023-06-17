load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_python",
    sha256 = "b6d46438523a3ec0f3cead544190ee13223a52f6a6765a29eae7b7cc24cc83a0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.1.0/rules_python-0.1.0.tar.gz",
)

http_archive(
    name = "rules_foreign_cc",
    sha256 = "c2cdcf55ffaf49366725639e45dedd449b8c3fe22b54e31625eb80ce3a240f1e",
    strip_prefix = "rules_foreign_cc-0.1.0",
    url = "https://github.com/bazelbuild/rules_foreign_cc/archive/0.1.0.zip",
)

load("@rules_foreign_cc//:workspace_definitions.bzl", "rules_foreign_cc_dependencies")

# This sets up some common toolchains for building targets. For more details, please see
# https://bazelbuild.github.io/rules_foreign_cc/0.1.0/#rules_foreign_cc_dependencies
rules_foreign_cc_dependencies()

load("//monolith:monolith_workspace.bzl", "monolith_workspace")

monolith_workspace()

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

# This is an unofficial boost build but it is useful.
git_repository(
    name = "com_github_nelhage_rules_boost",
    commit = "1e3a69bf2d5cd10c34b74f066054cd335d033d71",
    remote = "https://github.com/nelhage/rules_boost",
    shallow_since = "1591047380 -0700",
)

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")

boost_deps()

http_archive(
    name = "org_tensorflow_serving",
    patch_args = ["-p1"],
    patches = [
        "//third_party:org_tensorflow_serving/public_tf_serving.patch",
        "//third_party:org_tensorflow_serving/support_diff_dim_size_inputs.patch",
    ],
    sha256 = "8c1a4d31ec7ab041b9302348a01422e21349507c7a6f0974639386c8901b721b",
    strip_prefix = "serving-2.4.0",
    url = "https://github.com/tensorflow/serving/archive/2.4.0.tar.gz",
)

# To update TensorFlow to a new revision.
# 1. Update the 'git_commit' args below to include the new git hash.
# 2. Get the sha256 hash of the archive with a command such as...
#    curl -L https://github.com/tensorflow/tensorflow/archive/<git hash>.tar.gz | sha256sum
#    and update the 'sha256' arg with the result.
# 3. Request the new archive to be mirrored on mirror.bazel.build for more
#    reliable downloads.
load("@org_tensorflow_serving//tensorflow_serving:repo.bzl", "tensorflow_http_archive")

# Tensorflow 2.4.0
tensorflow_http_archive(
    name = "org_tensorflow",
    git_commit = "582c8d236cb079023657287c318ff26adb239002",
    patch = "//third_party:org_tensorflow/tf.patch",
    sha256 = "9c94bfec7214853750c7cacebd079348046f246ec0174d01cd36eda375117628",
)

http_archive(
    name = "rules_pkg",
    sha256 = "352c090cc3d3f9a6b4e676cf42a6047c16824959b438895a76c2989c6d7c246a",
    url = "https://github.com/bazelbuild/rules_pkg/releases/download/0.2.5/rules_pkg-0.2.5.tar.gz",
)

load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")

rules_pkg_dependencies()

load(
    "@org_tensorflow//third_party/toolchains/preconfig/generate:archives.bzl",
    "bazel_toolchains_archive",
)

bazel_toolchains_archive()

load(
    "@bazel_toolchains//repositories:repositories.bzl",
    bazel_toolchains_repositories = "repositories",
)

bazel_toolchains_repositories()

# START: Upstream TensorFlow dependencies
# TensorFlow build depends on these dependencies.
# Needs to be in-sync with TensorFlow sources.
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    ],
)

http_archive(
    name = "bazel_skylib",
    sha256 = "1dde365491125a3db70731e25658dfdd3bc5dbdfd11b840b3e987ecf043c7ca0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
    ],
)  # https://github.com/bazelbuild/bazel-skylib/releases

# END: Upstream TensorFlow dependencies

# Please add all new TensorFlow Serving and Archon dependencies in workspace.bzl.
load("//monolith:tf_serving_workspace.bzl", "tf_serving_workspace")

tf_serving_workspace()


load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

# Specify the minimum required bazel version.
load("@org_tensorflow//tensorflow:version_check.bzl", "check_bazel_version_at_least")

check_bazel_version_at_least("3.0.0")

# GPRC deps, required to match TF's.  Only after calling tf_serving_workspace()
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

http_archive(
    name = "upb",
    patch_args = ["-p1"],
    patches = ["//third_party:upb.patch"],
    sha256 = "61d0417abd60e65ed589c9deee7c124fe76a4106831f6ad39464e1525cef1454",
    strip_prefix = "upb-9effcbcb27f0a665f9f345030188c0b291e32482",
    url = "https://github.com/protocolbuffers/upb/archive/9effcbcb27f0a665f9f345030188c0b291e32482.tar.gz",
)

load("@upb//bazel:repository_defs.bzl", "bazel_version_repository")

bazel_version_repository(name = "bazel_version")

# Hedron's Compile Commands Extractor for Bazel
# https://github.com/hedronvision/bazel-compile-commands-extractor
http_archive(
    name = "hedron_compile_commands",
    strip_prefix = "bazel-compile-commands-extractor-79f8dcae6b451abb97fe76853c867792ac9ac703",

    # Replace the commit hash in both places (below) with the latest, rather than using the stale one here.
    # Even better, set up Renovate and let it do the work for you (see "Suggestion: Updates" in the README).
    url = "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/79f8dcae6b451abb97fe76853c867792ac9ac703.tar.gz",
    # When you first run this tool, it'll recommend a sha256 hash to put here with a message like: "DEBUG: Rule 'hedron_compile_commands' indicated that a canonical reproducible form can be obtained by modifying arguments sha256 = ..."
)

load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")

hedron_compile_commands_setup()


http_archive(
    name = "zstd",
    build_file = "//third_party:zstd.BUILD",
    sha256 = "a364f5162c7d1a455cc915e8e3cf5f4bd8b75d09bc0f53965b0c9ca1383c52c8",
    strip_prefix = "zstd-1.4.4",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/facebook/zstd/archive/v1.4.4.tar.gz",
        "https://github.com/facebook/zstd/archive/v1.4.4.tar.gz",
    ],
)

http_archive(
    name = "lz4",
    build_file = "//third_party:lz4.BUILD",
    sha256 = "658ba6191fa44c92280d4aa2c271b0f4fbc0e34d249578dd05e50e76d0e5efcc",
    strip_prefix = "lz4-1.9.2",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/lz4/lz4/archive/v1.9.2.tar.gz",
        "https://github.com/lz4/lz4/archive/v1.9.2.tar.gz",
    ],
)

http_archive(
    name = "kafka",
    build_file = "//third_party:kafka.BUILD",
    patch_cmds = [
        "rm -f src/win32_config.h",
        # TODO: Remove the fowllowing once librdkafka issue is resolved.
        """sed -i.bak '\\|rd_kafka_log(rk,|,/ exceeded);/ s/^/\\/\\//' src/rdkafka_cgrp.c""",
    ],
    sha256 = "f7fee59fdbf1286ec23ef0b35b2dfb41031c8727c90ced6435b8cf576f23a656",
    strip_prefix = "librdkafka-1.5.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/edenhill/librdkafka/archive/v1.5.0.tar.gz",
        "https://github.com/edenhill/librdkafka/archive/v1.5.0.tar.gz",
    ],
)

load("//third_party:repo.bzl", "tf_http_archive")

tf_http_archive(
    name = "cuCollections",
    build_file = "//third_party:cuco.BUILD",
    patch_file = "//third_party:cuCollections.patch",
    sha256 = "2e059ea1ae18173c5cc3f00989b114c431af78c674f92e35bed56367a9b8b186",
    strip_prefix = "cuCollections-1e3c5842c6e212e0bd7de9802af583e53009f4a6",
    urls = [
        "https://github.com/NVIDIA/cuCollections/archive/1e3c5842c6e212e0bd7de9802af583e53009f4a6.zip",
        "https://github.com/NVIDIA/cuCollections/archive/1e3c5842c6e212e0bd7de9802af583e53009f4a6.zip",
    ],
)

http_archive(
    name = "arrow",
    build_file = "//third_party:arrow.BUILD",
    sha256 = "57e13c62f27b710e1de54fd30faed612aefa22aa41fa2c0c3bacd204dd18a8f3",
    strip_prefix = "arrow-apache-arrow-7.0.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/apache/arrow/archive/apache-arrow-7.0.0.tar.gz",
        "https://github.com/apache/arrow/archive/apache-arrow-7.0.0.tar.gz",
    ],
)

# extra dependencies of arrow begin
http_archive(
    name = "rapidjson",
    build_file = "//third_party:rapidjson.BUILD",
    sha256 = "30bd2c428216e50400d493b38ca33a25efb1dd65f79dfc614ab0c957a3ac2c28",
    strip_prefix = "rapidjson-418331e99f859f00bdc8306f69eba67e8693c55e",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/miloyip/rapidjson/archive/418331e99f859f00bdc8306f69eba67e8693c55e.tar.gz",
        "https://github.com/miloyip/rapidjson/archive/418331e99f859f00bdc8306f69eba67e8693c55e.tar.gz",
    ],
)

http_archive(
    name = "xsimd",
    build_file = "//third_party:xsimd.BUILD",
    sha256 = "21b4700e9ef70f6c9a86952047efd8272317df4e6fee35963de9394fd9c5677f",
    strip_prefix = "xsimd-8.0.1",
    urls = [
        "https://github.com/xtensor-stack/xsimd/archive/refs/tags/8.0.1.tar.gz",
    ],
)

http_archive(
    name = "brotli",
    build_file = "//third_party:brotli.BUILD",
    sha256 = "4c61bfb0faca87219ea587326c467b95acb25555b53d1a421ffa3c8a9296ee2c",
    strip_prefix = "brotli-1.0.7",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/brotli/archive/v1.0.7.tar.gz",
        "https://github.com/google/brotli/archive/v1.0.7.tar.gz",
    ],
)

http_archive(
    name = "bzip2",
    build_file = "//third_party:bzip2.BUILD",
    sha256 = "ab5a03176ee106d3f0fa90e381da478ddae405918153cca248e682cd0c4a2269",
    strip_prefix = "bzip2-1.0.8",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz",
        "https://sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz",
    ],
)

http_archive(
    name = "thrift",
    build_file = "//third_party:thrift.BUILD",
    sha256 = "5da60088e60984f4f0801deeea628d193c33cec621e78c8a43a5d8c4055f7ad9",
    strip_prefix = "thrift-0.13.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/apache/thrift/archive/v0.13.0.tar.gz",
        "https://github.com/apache/thrift/archive/v0.13.0.tar.gz",
    ],
)

# extra dependencies of arrow end
