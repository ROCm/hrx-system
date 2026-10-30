# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Configured corpus fixture for Bazel-to-CMake conversion tests."""

load("//loom/build_tools/bazel:defs.bzl", "loom_corpus_catalog", "loom_corpus_manifest")

CONFIGURED_MANIFEST = loom_corpus_manifest(
    name = "configured",
    package = "//loom/build_tools/bazel_to_cmake/testdata/configured",
    configs = {
        "first.loom": {
            "first.columns": "4",
            "first.rows": "5",
        },
        "second.loom": {
            "second.capacity": "65",
            "second.empty": "",
        },
    },
    srcs = [
        "first.loom",
        "second.loom",
    ],
)

CONFIGURED_CORPUS = loom_corpus_catalog([CONFIGURED_MANIFEST])
