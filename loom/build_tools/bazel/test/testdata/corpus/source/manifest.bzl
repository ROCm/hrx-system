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
    scenario_srcs = [
        "nested/fixture.loom",
    ],
    legacy_case_srcs = [
        "excluded.loom",
        "other.loom",
    ],
)
