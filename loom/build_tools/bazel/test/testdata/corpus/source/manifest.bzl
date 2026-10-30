# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Semantic source manifest fixture for Loom corpus rule analysis tests."""

load("//loom/build_tools/bazel:defs.bzl", "loom_corpus_manifest")

SAMPLE_CORPUS = loom_corpus_manifest(
    name = "sample",
    package = "//loom/build_tools/bazel/test/testdata/corpus/source",
    configs = {
        "nested/fixture.loom": {
            "sample.value": "42",
        },
    },
    srcs = [
        "excluded.loom",
        "nested/fixture.loom",
        "other.loom",
    ],
)
