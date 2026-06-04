# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Repository rule exposing header overlays from a configured ROCm root."""

def _rocm_repository_impl(repository_ctx):
    rocm_path = repository_ctx.getenv("IREE_ROCM_PATH")
    if not rocm_path:
        display_name = repository_ctx.attr.display_name or repository_ctx.name
        fail(
            (
                "{} requires --repo_env=IREE_ROCM_PATH=/path/to/rocm. " +
                "Run python build_tools/bazel/configure.py with a ROCm-backed " +
                "driver and -DIREE_ROCM_PATH=/path/to/rocm to generate " +
                ".bazelrc.configured."
            ).format(
                display_name,
            ),
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
    },
    environ = ["IREE_ROCM_PATH"],
    doc = "Exposes a configured ROCm root as a Bazel repository.",
)
