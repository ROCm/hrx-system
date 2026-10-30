# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Execution profiles owned by the hosted WebAssembly target provider."""

load("//loom/build_tools/bazel:defs.bzl", "loom_execution_profile")
load(
    "//loom/requirements:defs.bzl",
    "ARCH_VM",
    "ARCH_WASM",
    "ARTIFACT_VM_BINARY",
    "ARTIFACT_WASM_BINARY",
    "EXECUTE_IREE_VM",
    "EXECUTE_WASM_HOST",
)

WASM_HOSTED_PROFILE = loom_execution_profile(
    name = "wasm_hosted",
    build_requirements = [
        ARCH_VM,
        ARCH_WASM,
        ARTIFACT_VM_BINARY,
        ARTIFACT_WASM_BINARY,
        EXECUTE_IREE_VM,
        EXECUTE_WASM_HOST,
    ],
    executor = "hosted",
    runner = "//loom/src/loom/tools/loom-test:loom-test-wasi",
    target_class = "cpu",
    target_family = "wasm",
)
