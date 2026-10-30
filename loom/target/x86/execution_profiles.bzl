# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native x86 execution through the task HAL."""

load("//loom/build_tools/bazel:defs.bzl", "loom_execution_profile")
load(
    "//loom/requirements:defs.bzl",
    "ARCH_VM",
    "ARCH_X86",
    "ARTIFACT_VM_BINARY",
    "ARTIFACT_X86_ELF",
    "EXECUTE_IREE_HAL",
    "EXECUTE_IREE_VM",
)
load(
    "//runtime/requirements:defs.bzl",
    "HAL_EXECUTABLE_LOADER_EMBEDDED_ELF",
    "HAL_TASK",
)

X86_TASK_HARDWARE_PROFILE = loom_execution_profile(
    name = "x86_task_hardware",
    build_requirements = [
        ARCH_X86,
        ARCH_VM,
        ARTIFACT_X86_ELF,
        ARTIFACT_VM_BINARY,
        EXECUTE_IREE_HAL,
        EXECUTE_IREE_VM,
        HAL_TASK,
        HAL_EXECUTABLE_LOADER_EMBEDDED_ELF,
    ],
    executor = "hardware",
    runner_args = [
        "--device=task",
        "--task_topology_max_group_count=2",
    ],
    target_class = "cpu",
    target_family = "x86",
)
