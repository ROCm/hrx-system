# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Correctness programs for portable HIP authoring patterns."""

load("//loom/build_tools/bazel:defs.bzl", "loom_corpus_manifest")

HIP_AUTHORING_CORPUS = loom_corpus_manifest(
    name = "hip_authoring",
    package = "//loom/src/loom/test/corpus/authoring/hip",
    srcs = [
        "packed_field_contracts.loom",
        "q8_block_unroll.loom",
        "q8_load_width.loom",
        "q8_q4_signedness.loom",
        "shared_memory_tile.loom",
        "shared_memory_transpose.loom",
        "shared_memory_vector_tile.loom",
        "shared_memory_wave_tile.loom",
        "target_provider_selection.loom",
        "template_math_legalization.loom",
    ],
)
