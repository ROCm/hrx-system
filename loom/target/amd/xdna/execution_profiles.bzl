# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public execution profiles owned by the Loom XDNA target provider."""

load("//build_tools/bazel:test_resources.bzl", "GPU_DEVICE_RESOURCE_GROUP")
load("//libamdf/requirements:defs.bzl", "LIBAMDF", "LIBAMDF_XDNA", "XDNA_RESOURCE")
load("//loom/build_tools/bazel:defs.bzl", "loom_execution_profile")
load(
    "//loom/requirements:defs.bzl",
    "EMIT_XDNA",
    "EXECUTE_IREE_HAL",
    "TARGET_ARCH_XDNA",
)
load("//runtime/requirements:defs.bzl", "HAL_AMD")

XDNA_HARDWARE_PROFILE = loom_execution_profile(
    name = "xdna_hardware",
    build_requirements = [
        LIBAMDF,
        LIBAMDF_XDNA,
        TARGET_ARCH_XDNA,
        EMIT_XDNA,
        EXECUTE_IREE_HAL,
        HAL_AMD,
    ],
    executor = "hardware",
    resource_group = GPU_DEVICE_RESOURCE_GROUP,
    run_requirements = [XDNA_RESOURCE],
    runner_args = ["--device=xdna"],
    target_class = "npu",
    target_family = "xdna",
)
