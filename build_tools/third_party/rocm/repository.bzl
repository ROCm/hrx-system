# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Repository rule exposing ROCm header overlays."""

_DEPENDENCY_MODES = ["pinned", "package", "auto"]

def _display_name(repository_ctx):
    return repository_ctx.attr.display_name or repository_ctx.name

def _dependency_mode(repository_ctx):
    rocm_mode = repository_ctx.getenv("IREE_ROCM_DEPENDENCY_MODE")
    if rocm_mode:
        mode = rocm_mode
    elif repository_ctx.getenv("IREE_ROCM_PATH"):
        mode = "package"
    else:
        mode = repository_ctx.getenv("IREE_DEPENDENCY_MODE") or "pinned"
    mode = mode.lower()
    if mode not in _DEPENDENCY_MODES:
        fail(
            "IREE_ROCM_DEPENDENCY_MODE/IREE_DEPENDENCY_MODE must be one of {}; got '{}'".format(
                ", ".join(_DEPENDENCY_MODES),
                mode,
            ),
        )
    return mode

def _pinned_source_urls(repository_ctx):
    if repository_ctx.attr.urls:
        return repository_ctx.attr.urls
    if repository_ctx.attr.url:
        return [repository_ctx.attr.url]
    return []

def _symlink_system_include_dir(repository_ctx):
    display_name = _display_name(repository_ctx)
    rocm_path = repository_ctx.getenv("IREE_ROCM_PATH")
    if not rocm_path:
        fail(
            (
                "{} requires --repo_env=IREE_ROCM_PATH=/path/to/rocm " +
                "when ROCm package dependency mode is selected. Run python " +
                "build_tools/bazel/configure.py with a ROCm-backed driver " +
                "and -DIREE_ROCM_PATH=/path/to/rocm to generate " +
                ".bazelrc.configured."
            ).format(display_name),
        )

    rocm_root = repository_ctx.path(rocm_path)
    if not rocm_root.exists:
        fail("Configured IREE_ROCM_PATH does not exist: {}".format(rocm_path))

    include_dir = repository_ctx.path(rocm_path + "/include")
    if not include_dir.exists:
        fail("Configured IREE_ROCM_PATH has no include directory: {}".format(
            rocm_path,
        ))

    repository_ctx.symlink(include_dir, "include")

def _download_pinned_source(repository_ctx):
    display_name = _display_name(repository_ctx)
    urls = _pinned_source_urls(repository_ctx)
    if not urls:
        fail(
            (
                "{} does not have pinned source metadata. Provide " +
                "--repo_env=IREE_ROCM_PATH=/path/to/rocm to use ROCm package " +
                "mode, or disable the feature that requires this dependency."
            ).format(display_name),
        )
    if not repository_ctx.attr.sha256:
        fail("{} pinned source metadata requires sha256".format(display_name))

    repository_ctx.download_and_extract(
        url = urls,
        sha256 = repository_ctx.attr.sha256,
        stripPrefix = repository_ctx.attr.strip_prefix,
    )

def _rocm_repository_impl(repository_ctx):
    mode = _dependency_mode(repository_ctx)
    if mode == "package" or (
        mode == "auto" and repository_ctx.getenv("IREE_ROCM_PATH")
    ):
        _symlink_system_include_dir(repository_ctx)
    else:
        _download_pinned_source(repository_ctx)

    repository_ctx.file(
        "BUILD.bazel",
        repository_ctx.read(repository_ctx.attr.build_file),
    )

rocm_repository = repository_rule(
    implementation = _rocm_repository_impl,
    attrs = {
        "build_file": attr.label(
            allow_single_file = True,
            mandatory = True,
            doc = "BUILD file overlay for the configured ROCm root.",
        ),
        "display_name": attr.string(
            doc = "Human-readable dependency name for diagnostics.",
        ),
        "sha256": attr.string(
            doc = "SHA256 digest for the pinned source archive.",
        ),
        "strip_prefix": attr.string(
            doc = "Directory prefix to strip from the pinned source archive.",
        ),
        "url": attr.string(
            doc = "URL for the pinned source archive.",
        ),
        "urls": attr.string_list(
            doc = "Fallback URLs for the pinned source archive.",
        ),
    },
    environ = [
        "IREE_DEPENDENCY_MODE",
        "IREE_ROCM_DEPENDENCY_MODE",
        "IREE_ROCM_PATH",
    ],
    doc = "Exposes pinned or configured-package ROCm headers as a Bazel repository.",
)
