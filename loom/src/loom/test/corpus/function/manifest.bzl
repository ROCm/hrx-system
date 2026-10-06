# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Authored source inventory for function semantic conformance."""

load("//loom/build_tools/bazel:defs.bzl", "loom_corpus_manifest")

FUNCTION_CORPUS = loom_corpus_manifest(
    name = "function",
    package = "//loom/src/loom/test/corpus/function",
    scenario_srcs = [
    ],
    legacy_case_srcs = [
        "buffer/arguments.loom",
        "buffer/call_overflow.loom",
        "buffer/calls.loom",
        "call/calls.loom",
        "call/direct.loom",
        "call/overflow.loom",
        "call/spill.loom",
        "rodata.loom",
        "template/expansion.loom",
        "template/library.loom",
        "template/motion.loom",
        "template/relations.loom",
    ],
)
