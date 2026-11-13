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
    "ARCH_AMDGPU",
    "ARCH_VM",
    "ARTIFACT_AMDGPU_ELF",
    "ARTIFACT_VM_BINARY",
    "EXECUTE_IREE_HAL",
    "EXECUTE_IREE_VM",
)
load(
    "//runtime/requirements:defs.bzl",
    "HAL_AMD_GPU",
)

def amd_gpu_execution_profile(name, runner_args = [], tags = [], device = "amdgpu"):
    """Defines native AMD GPU execution with libamdf hardware requirements."""
    return loom_execution_profile(
        name = name,
        build_requirements = [
            LIBAMDF,
            LIBAMDF_GPU,
            ARCH_AMDGPU,
            ARCH_VM,
            ARTIFACT_AMDGPU_ELF,
            ARTIFACT_VM_BINARY,
            EXECUTE_IREE_HAL,
            EXECUTE_IREE_VM,
            HAL_AMD_GPU,
        ],
        executor = "hardware",
        resource_group = GPU_DEVICE_RESOURCE_GROUP,
        run_requirements = [LIBAMDF_AMDGPU_RESOURCE],
        runner_args = ["--device=" + device] + runner_args,
        tags = tags,
        target_class = "gpu",
        target_family = "amdgpu",
    )

AMD_GPU_HARDWARE_PROFILE = amd_gpu_execution_profile(name = "amd_gpu_hardware")

AMD_GPU_ACCESS_PROFILE = amd_gpu_execution_profile(
    name = "amd_gpu_access",
    runner_args = ["--sanitizer=access"],
    tags = ["notsan", "requires-amd-gpu-feedback"],
)

AMD_GPU_ASAN_PROFILE = amd_gpu_execution_profile(
    name = "amd_gpu_asan",
    runner_args = ["--sanitizer=asan", "--sanitizer-reporting=report-only"],
    tags = ["notsan", "requires-amd-gpu-feedback"],
)

AMD_GPU_TSAN_PROFILE = amd_gpu_execution_profile(
    name = "amd_gpu_tsan",
    tags = ["requires-amd-gpu-feedback"],
    device = "amdgpu://?tsan_report_policy=report-only",
    runner_args = ["--sanitizer=tsan", "--sanitizer-reporting=report-only"],
)

# A small queue shadow exercises exact-boundary and capacity-report paths.
AMD_GPU_TSAN_CAPACITY_PROFILE = amd_gpu_execution_profile(
    name = "amd_gpu_tsan_capacity",
    tags = ["requires-amd-gpu-feedback"],
    device = "amdgpu://?tsan_report_policy=report-only&tsan_shadow_size=1440",
    runner_args = ["--sanitizer=tsan", "--sanitizer-reporting=report-only"],
)
