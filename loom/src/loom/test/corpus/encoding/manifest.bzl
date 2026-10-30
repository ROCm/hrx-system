# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Correctness programs for encoded numeric representations."""

load("//loom/build_tools/bazel:defs.bzl", "loom_corpus_manifest")

ENCODING_CORPUS = loom_corpus_manifest(
    name = "encoding",
    package = "//loom/src/loom/test/corpus/encoding",
    configs = {
        "fp8_encode.loom": {
            "fp8_encode.row_capacity": "65536",
        },
        "fp8_finite_storage_decode.loom": {
            "finite_storage_decode.input_capacity": "4608",
            "finite_storage_decode.output_capacity": "16",
            "finite_storage_decode.row_capacity": "65",
        },
    },
    srcs = [
        "fp8_encode.loom",
        "fp8_finite_storage_decode.loom",
        "narrow_integer_carriers.loom",
    ],
)
