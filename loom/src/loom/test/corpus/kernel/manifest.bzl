# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Authored source inventory for kernel semantic conformance."""

load("//loom/build_tools/bazel:defs.bzl", "loom_corpus_manifest")

KERNEL_CORPUS = loom_corpus_manifest(
    name = "kernel",
    package = "//loom/src/loom/test/corpus/kernel",
    scenario_srcs = [
        "subgroup/shuffle_participation.loom",
        "subgroup/transport.loom",
        "subgroup/transport_carrier.loom",
    ],
    legacy_case_srcs = [
        "subgroup/active_predicate.loom",
        "subgroup/ballot.loom",
        "subgroup/shuffle_dynamic.loom",
        "workgroup/loop_state.loom",
        "workgroup/reduce_partial.loom",
        "workgroup/reduce_tree.loom",
    ],
)
