load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@rules_python//python:pip.bzl", "pip_install")

def monolith_workspace():
    """Adds monolith workspace's dependencies."""
    http_archive(
        name = "msgpack",
        build_file = "//third_party:msgpack/msgpack.BUILD",
        strip_prefix = "msgpack-3.3.0",
        sha256 = "6e114d12a5ddb8cb11f669f83f32246e484a8addd0ce93f274996f1941c1f07b",
        urls = ["https://github.com/msgpack/msgpack-c/releases/download/cpp-3.3.0/msgpack-3.3.0.tar.gz"],
    )

    pip_install(
        name = "pip_deps",
        requirements = "//third_party/pip_deps:requirements.txt",
        python_interpreter = "python3",
    )

    http_archive(
        name = "gperftools",
        build_file = "//third_party:gperftools/gperftools.BUILD",
        sha256 = "81bb34f546ac8cddd064f8935805f8eb19e3c9661188e127b4c90526e944ebff",
        urls = [
            "https://github.com/gperftools/gperftools/releases/download/gperftools-2.7/gperftools-2.7.zip",
        ],
        patches = ["//third_party:gperftools/gperftools.patch"],
        patch_args = ["-p1"],
        strip_prefix = "gperftools-2.7",
    )

