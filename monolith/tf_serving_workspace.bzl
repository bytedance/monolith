# The file is copied from https://github.com/tensorflow/serving/blob/master/tensorflow_serving/workspace.bzl
# The modification is that we modify //third_part to @org_tensorflow_serving//third_part

# TensorFlow Serving external dependencies that can be loaded in WORKSPACE
# files.

load("@org_tensorflow//third_party:repo.bzl", "tf_http_archive")
load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def tf_serving_workspace():
    """All TensorFlow Serving external dependencies."""

    tf_workspace(path_prefix = "", tf_repo_name = "org_tensorflow")

    # ===== gRPC dependencies =====
    native.bind(
        name = "libssl",
        actual = "@boringssl//:ssl",
    )

    # gRPC wants the existence of a cares dependence but its contents are not
    # actually important since we have set GRPC_ARES=0 in tools/bazel.rc
    native.bind(
        name = "cares",
        actual = "@grpc//third_party/nanopb:nanopb",
    )

    # ===== RapidJSON (rapidjson.org) dependencies =====
    http_archive(
        name = "com_github_tencent_rapidjson",
        urls = [
            "https://github.com/Tencent/rapidjson/archive/v1.1.0.zip",
        ],
        sha256 = "8e00c38829d6785a2dfb951bb87c6974fa07dfe488aa5b25deec4b8bc0f6a3ab",
        strip_prefix = "rapidjson-1.1.0",
        build_file = "@org_tensorflow_serving//third_party/rapidjson:BUILD",
    )

    # ===== libevent (libevent.org) dependencies =====
    http_archive(
        name = "com_github_libevent_libevent",
        urls = [
            "https://github.com/libevent/libevent/archive/release-2.1.8-stable.zip",
        ],
        sha256 = "70158101eab7ed44fd9cc34e7f247b3cae91a8e4490745d9d6eb7edc184e4d96",
        strip_prefix = "libevent-release-2.1.8-stable",
        build_file = "@org_tensorflow_serving//third_party/libevent:BUILD",
    )

    # ===== Override TF & TF Text defined 'ICU'. (we need a version that contains all data).
    http_archive(
        name = "icu",
        strip_prefix = "icu-release-64-2",
        sha256 = "dfc62618aa4bd3ca14a3df548cd65fe393155edd213e49c39f3a30ccd618fc27",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/unicode-org/icu/archive/release-64-2.zip",
            "https://github.com/unicode-org/icu/archive/release-64-2.zip",
        ],
        build_file = "@org_tensorflow_serving//third_party/icu:BUILD",
        patches = ["@org_tensorflow_serving//third_party/icu:data.patch"],
        patch_args = ["-p1", "-s"],
    )

    # ===== Pin `com_google_absl` with the same version(and patch) with Tensorflow.
    tf_http_archive(
        name = "com_google_absl",
        build_file = str(Label("@org_tensorflow//third_party:com_google_absl.BUILD")),
        # TODO: Remove the patch when https://github.com/abseil/abseil-cpp/issues/326 is resolved
        # and when TensorFlow is build against CUDA 10.2
        patch_file = str(Label("@org_tensorflow//third_party:com_google_absl_fix_mac_and_nvcc_build.patch")),
        sha256 = "f368a8476f4e2e0eccf8a7318b98dafbe30b2600f4e3cf52636e5eb145aba06a",  # SHARED_ABSL_SHA
        strip_prefix = "abseil-cpp-df3ea785d8c30a9503321a3d35ee7d35808f190d",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/abseil/abseil-cpp/archive/df3ea785d8c30a9503321a3d35ee7d35808f190d.tar.gz",
            "https://github.com/abseil/abseil-cpp/archive/df3ea785d8c30a9503321a3d35ee7d35808f190d.tar.gz",
        ],
    )

    # ===== TF.Text dependencies
    # NOTE: Before updating this version, you must update the test model
    # and double check all custom ops have a test:
    # https://github.com/tensorflow/text/blob/master/oss_scripts/model_server/save_models.py
    http_archive(
        name = "org_tensorflow_text",
        sha256 = "05cc1b0eda8f4f734cb81d4389a637d26372b8621cb4c4a7e30ee5bc1e8c63da",
        strip_prefix = "text-2.3.0",
        urls = [
            "https://github.com/tensorflow/text/archive/v2.3.0.zip",
        ],
        patches = ["@org_tensorflow_serving//third_party/tf_text:tftext.patch"],
        patch_args = ["-p1"],
        repo_mapping = {"@com_google_re2": "@com_googlesource_code_re2"},
    )

    http_archive(
        name = "com_google_sentencepiece",
        strip_prefix = "sentencepiece-1.0.0",
        sha256 = "c05901f30a1d0ed64cbcf40eba08e48894e1b0e985777217b7c9036cac631346",
        urls = [
            "https://github.com/google/sentencepiece/archive/1.0.0.zip",
        ],
    )

    http_archive(
        name = "com_google_glog",
        sha256 = "1ee310e5d0a19b9d584a855000434bb724aa744745d5b8ab1855c85bff8a8e21",
        strip_prefix = "glog-028d37889a1e80e8a07da1b8945ac706259e5fd8",
        urls = [
            "https://mirror.bazel.build/github.com/google/glog/archive/028d37889a1e80e8a07da1b8945ac706259e5fd8.tar.gz",
            "https://github.com/google/glog/archive/028d37889a1e80e8a07da1b8945ac706259e5fd8.tar.gz",
        ],
    )

    # ==== we need to modify eigen_archive to make it compilable in gcc6
    tf_http_archive(
        name = "eigen_archive",
        build_file = str(Label("@org_tensorflow//third_party:eigen.BUILD")),
        patch_file = "//third_party:eigen3/eigen_gcc6.patch",
        sha256 = "e807a6a6f3a0e8ab10adeb59bb5a9bbb113e8e1684f9b4b32f73f58fd758b4cf",  # SHARED_EIGEN_SHA
        strip_prefix = "eigen-011e0db31d1bed8b7f73662be6d57d9f30fa457a",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/gitlab.com/libeigen/eigen/-/archive/011e0db31d1bed8b7f73662be6d57d9f30fa457a/eigen-011e0db31d1bed8b7f73662be6d57d9f30fa457a.tar.gz",
            "https://gitlab.com/libeigen/eigen/-/archive/011e0db31d1bed8b7f73662be6d57d9f30fa457a/eigen-011e0db31d1bed8b7f73662be6d57d9f30fa457a.tar.gz",
        ],
    )
