# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Execution profiles owned by the Loom AMDGPU target provider."""

load("//loom/build_tools/bazel:defs.bzl", "loom_execution_profile")
load(
    "//loom/requirements:defs.bzl",
    "EMIT_AMDGPU",
    "EXECUTE_IREE_HAL",
    "TARGET_ARCH_AMDGPU",
)
load(
    "//runtime/requirements:defs.bzl",
    "AMDGPU_RESOURCE",
    "HAL_AMDGPU",
)

_AMDGPU_BUILD_REQUIREMENTS = [
    TARGET_ARCH_AMDGPU,
    EMIT_AMDGPU,
    EXECUTE_IREE_HAL,
    HAL_AMDGPU,
]

_AMDGPU_PROFILE_ARGUMENTS = ["--device=amdgpu"]

AMDGPU_HARDWARE_PROFILE = loom_execution_profile(
    name = "amdgpu_hardware",
    build_requirements = _AMDGPU_BUILD_REQUIREMENTS,
    executor = "hardware",
    resource_group = "loom-amdgpu-tests",
    run_requirements = [AMDGPU_RESOURCE],
    runner_args = _AMDGPU_PROFILE_ARGUMENTS,
    target_class = "gpu",
    target_family = "amdgpu",
)

AMDGPU_ACCESS_PROFILE = loom_execution_profile(
    name = "amdgpu_access",
    build_requirements = _AMDGPU_BUILD_REQUIREMENTS,
    executor = "hardware",
    resource_group = "loom-amdgpu-tests",
    run_requirements = [AMDGPU_RESOURCE],
    runner_args = _AMDGPU_PROFILE_ARGUMENTS + ["--sanitizer=access"],
    tags = ["notsan"],
    target_class = "gpu",
    target_family = "amdgpu",
)

AMDGPU_ASAN_PROFILE = loom_execution_profile(
    name = "amdgpu_asan",
    build_requirements = _AMDGPU_BUILD_REQUIREMENTS,
    executor = "hardware",
    resource_group = "loom-amdgpu-tests",
    run_requirements = [AMDGPU_RESOURCE],
    runner_args = _AMDGPU_PROFILE_ARGUMENTS + [
        "--config=asan_launch_bound.tile_capacity=1",
        "--sanitizer=asan",
        "--sanitizer-reporting=report-only",
        "--amdgpu_asan=true",
        "--amdgpu_asan_report_policy=report-only",
    ],
    tags = ["notsan"],
    target_class = "gpu",
    target_family = "amdgpu",
)

AMDGPU_TSAN_PROFILE = loom_execution_profile(
    name = "amdgpu_tsan",
    build_requirements = _AMDGPU_BUILD_REQUIREMENTS,
    executor = "hardware",
    resource_group = "loom-amdgpu-tests",
    run_requirements = [AMDGPU_RESOURCE],
    runner_args = _AMDGPU_PROFILE_ARGUMENTS + [
        "--sanitizer=tsan",
        "--sanitizer-reporting=report-only",
        "--amdgpu_tsan=true",
        "--amdgpu_tsan_report_policy=report-only",
    ],
    target_class = "gpu",
    target_family = "amdgpu",
)
