# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Execution profiles owned by the Loom SPIR-V target provider."""

load("//build_tools/bazel:test_resources.bzl", "GPU_DEVICE_RESOURCE_GROUP")
load("//build_tools/vulkan/requirements:defs.bzl", "VULKAN_DEVICE_RESOURCE")
load("//loom/build_tools/bazel:defs.bzl", "loom_execution_profile")
load(
    "//loom/requirements:defs.bzl",
    "ARCH_SPIRV",
    "ARTIFACT_SPIRV_BINARY",
    "EXECUTE_IREE_HAL",
)
load(
    "//runtime/requirements:defs.bzl",
    "HAL_VULKAN",
)

def _spirv_vulkan_hardware_profile(
        name,
        runner_args = []):
    return loom_execution_profile(
        name = name,
        build_requirements = [
            ARCH_SPIRV,
            ARTIFACT_SPIRV_BINARY,
            EXECUTE_IREE_HAL,
            HAL_VULKAN,
        ],
        executor = "hardware",
        resource_group = GPU_DEVICE_RESOURCE_GROUP,
        run_requirements = [VULKAN_DEVICE_RESOURCE],
        runner_args = [
            "--device=vulkan",
            "--vulkan_validation_layers=false",
        ] + runner_args,
        target_class = "gpu",
        target_family = "spirv",
    )

SPIRV_VULKAN_HARDWARE_PROFILE = _spirv_vulkan_hardware_profile(
    name = "spirv_vulkan_hardware",
)

SPIRV_VULKAN_EXPLICIT_TARGET_HARDWARE_PROFILE = _spirv_vulkan_hardware_profile(
    name = "spirv_vulkan_explicit_target_hardware",
    runner_args = ["--target=spirv:vulkan1.3+bda"],
)
