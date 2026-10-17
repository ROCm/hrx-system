# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Authored source inventory for control semantic conformance."""

load("//loom/build_tools/bazel:defs.bzl", "loom_corpus_manifest")

CONTROL_CORPUS = loom_corpus_manifest(
    name = "control",
    package = "//loom/src/loom/test/corpus/control",
    srcs = [
        "branch/cfg.loom",
        "branch/effects.loom",
        "branch/guards.loom",
        "branch/half.loom",
        "branch/publication.loom",
        "branch/results.loom",
        "branch/structured.loom",
        "branch/uniform_predicate.loom",
        "loop/callable.loom",
        "loop/carried_lane_liveness.loom",
        "loop/condition.loom",
        "loop/count.loom",
        "loop/predicate_lifetimes.loom",
        "loop/rotation/offsets.loom",
        "loop/rotation/values.loom",
        "loop/rotation/views.loom",
        "loop/scalar_state.loom",
        "loop/sequential_slices.loom",
        "loop/termination.loom",
        "loop/vector_recurrence.loom",
        "loop/wide_index_recurrence.loom",
        "schedule/address_domains.loom",
        "schedule/guarded_recurrence.loom",
        "schedule/ordered_read_ahead.loom",
        "schedule/unroll_scope.loom",
    ],
)
