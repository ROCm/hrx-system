# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target-neutral source catalog for Loom semantic conformance."""

load(
    "//loom/build_tools/bazel:defs.bzl",
    "loom_corpus_catalog",
    "loom_legacy_case_corpus",
    "loom_scenario_corpus",
)
load("//loom/src/loom/test/corpus/control:manifest.bzl", "CONTROL_CORPUS")
load("//loom/src/loom/test/corpus/function:manifest.bzl", "FUNCTION_CORPUS")
load("//loom/src/loom/test/corpus/kernel:manifest.bzl", "KERNEL_CORPUS")
load("//loom/src/loom/test/corpus/memory:manifest.bzl", "MEMORY_CORPUS")
load("//loom/src/loom/test/corpus/numeric:manifest.bzl", "NUMERIC_CORPUS")

LOOM_CORPUS = loom_corpus_catalog([
    NUMERIC_CORPUS,
    CONTROL_CORPUS,
    MEMORY_CORPUS,
    KERNEL_CORPUS,
    FUNCTION_CORPUS,
])

LOOM_SCENARIOS = loom_scenario_corpus([
    NUMERIC_CORPUS,
    CONTROL_CORPUS,
    MEMORY_CORPUS,
    KERNEL_CORPUS,
    FUNCTION_CORPUS,
])

LOOM_LEGACY_CASES = loom_legacy_case_corpus([
    NUMERIC_CORPUS,
    CONTROL_CORPUS,
    MEMORY_CORPUS,
    KERNEL_CORPUS,
    FUNCTION_CORPUS,
])
