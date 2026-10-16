# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public execution profiles owned by the Loom AMDGPU target provider."""

load("//build_tools/bazel:test_resources.bzl", "GPU_DEVICE_RESOURCE_GROUP")
load(
    "//libamdf/requirements:defs.bzl",
    "LIBAMDF",
    "LIBAMDF_GPU",
    LIBAMDF_AMDGPU_RESOURCE = "AMDGPU_RESOURCE",
)
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
    "HAL_AMD",
    "HAL_AMDGPU",
)

def amdgpu_execution_profile(
        name,
        runner_args = [],
        tags = []):
    """Defines AMDGPU execution with shared device and resource requirements.

    Callers own instrumentation and diagnostic reporting.
    Workload configuration and case selection remain on the test declaration.
    """
    return loom_execution_profile(
        name = name,
        build_requirements = [
            TARGET_ARCH_AMDGPU,
            EMIT_AMDGPU,
            EXECUTE_IREE_HAL,
            HAL_AMDGPU,
        ],
        executor = "hardware",
        resource_group = GPU_DEVICE_RESOURCE_GROUP,
        run_requirements = [AMDGPU_RESOURCE],
        runner_args = ["--device=amdgpu"] + runner_args,
        tags = tags,
        target_class = "gpu",
        target_family = "amdgpu",
    )

AMDGPU_HARDWARE_PROFILE = amdgpu_execution_profile(
    name = "amdgpu_hardware",
)

def rdna_execution_profile(name, runner_args = [], tags = [], device = "rdna"):
    """Defines native AMD GPU execution with libamdf hardware requirements."""
    return loom_execution_profile(
        name = name,
        build_requirements = [
            LIBAMDF,
            LIBAMDF_GPU,
            TARGET_ARCH_AMDGPU,
            EMIT_AMDGPU,
            EXECUTE_IREE_HAL,
            HAL_AMD,
        ],
        executor = "hardware",
        resource_group = GPU_DEVICE_RESOURCE_GROUP,
        run_requirements = [LIBAMDF_AMDGPU_RESOURCE],
        runner_args = ["--device=" + device] + runner_args,
        tags = tags,
        target_class = "gpu",
        target_family = "amdgpu",
    )

RDNA_HARDWARE_PROFILE = rdna_execution_profile(name = "rdna_hardware")

RDNA_ASAN_PROFILE = rdna_execution_profile(
    name = "rdna_asan",
    runner_args = ["--sanitizer=asan", "--sanitizer-reporting=report-only"],
    tags = ["notsan"],
)

RDNA_TSAN_PROFILE = rdna_execution_profile(
    name = "rdna_tsan",
    device = "rdna://?tsan_report_policy=report-only",
    runner_args = ["--sanitizer=tsan", "--sanitizer-reporting=report-only"],
)

RDNA_TSAN_BYTE_GRANULE_PROFILE = rdna_execution_profile(
    name = "rdna_tsan_byte_granule",
    device = "rdna://?tsan_report_policy=report-only&tsan_memory_granule_shift=0",
    runner_args = ["--sanitizer=tsan", "--sanitizer-reporting=report-only"],
)

AMDGPU_ACCESS_PROFILE = amdgpu_execution_profile(
    name = "amdgpu_access",
    runner_args = ["--sanitizer=access"],
    tags = ["notsan"],
)

AMDGPU_ASAN_PROFILE = amdgpu_execution_profile(
    name = "amdgpu_asan",
    runner_args = [
        "--sanitizer=asan",
        "--sanitizer-reporting=report-only",
        "--amdgpu_asan=true",
        "--amdgpu_asan_report_policy=report-only",
    ],
    tags = ["notsan"],
)

AMDGPU_TSAN_PROFILE = amdgpu_execution_profile(
    name = "amdgpu_tsan",
    runner_args = [
        "--sanitizer=tsan",
        "--sanitizer-reporting=report-only",
        "--amdgpu_tsan=true",
        "--amdgpu_tsan_report_policy=report-only",
    ],
)

# A deliberately small queue shadow arena used to exercise the compiler's
# exact-boundary and capacity-report paths without allocating a production-size
# sanitizer shadow during the test.
AMDGPU_TSAN_CAPACITY_PROFILE = amdgpu_execution_profile(
    name = "amdgpu_tsan_capacity",
    runner_args = [
        "--sanitizer=tsan",
        "--sanitizer-reporting=report-only",
        "--amdgpu_tsan=true",
        "--amdgpu_tsan_report_policy=report-only",
        "--amdgpu_tsan_shadow_size=1440",
    ],
)
