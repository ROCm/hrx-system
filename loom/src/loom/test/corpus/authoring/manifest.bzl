# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Authored application-pattern correctness programs."""

load("//loom/build_tools/bazel:defs.bzl", "loom_corpus_manifest")

AUTHORING_CORPUS = loom_corpus_manifest(
    name = "authoring",
    package = "//loom/src/loom/test/corpus/authoring",
    configs = {
        "indexed_row_gather_f32.loom": {
            "indexed_row_gather_f32.column_capacity": "4",
            "indexed_row_gather_f32.row_capacity": "5",
            "indexed_row_gather_f32.source_row_capacity": "2",
        },
        "memset_i8.loom": {
            "memset_i8.element_capacity": "1009",
        },
        "mlp_down_projection_residual_bf16.loom": {
            "mlp_down_projection_residual_bf16.row_capacity": "2",
        },
    },
    srcs = [
        "indexed_row_gather_f32.loom",
        "memset_i8.loom",
        "mlp_down_projection_residual_bf16.loom",
        "wide_vector_conversion.loom",
    ],
)
