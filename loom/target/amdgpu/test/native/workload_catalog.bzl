# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target-neutral application programs qualified by the native AMDGPU path."""

load("//loom/build_tools/bazel:defs.bzl", "loom_corpus_catalog")
load("//loom/src/loom/test/corpus/authoring:manifest.bzl", "AUTHORING_CORPUS")
load("//loom/src/loom/test/corpus/authoring/hip:manifest.bzl", "HIP_AUTHORING_CORPUS")
load("//loom/src/loom/test/corpus/checked_benchmarks:manifest.bzl", "CHECKED_BENCHMARK_CORPUS")
load("//loom/src/loom/test/corpus/encoding:manifest.bzl", "ENCODING_CORPUS")

AMDGPU_WORKLOAD_CORPUS = loom_corpus_catalog([
    AUTHORING_CORPUS,
    HIP_AUTHORING_CORPUS,
    CHECKED_BENCHMARK_CORPUS,
    ENCODING_CORPUS,
])
