# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Execution profiles owned by the Loom SPIR-V target provider."""

load("//build_tools/sanitizer:suppressions.bzl", "vulkan_suppressions")
load("//loom/build_tools/bazel:defs.bzl", "loom_execution_profile")
load(
    "//loom/requirements:defs.bzl",
    "EMIT_SPIRV",
    "EXECUTE_IREE_HAL",
    "TARGET_ARCH_SPIRV",
)
load(
    "//runtime/requirements:defs.bzl",
    "HAL_VULKAN",
    "VULKAN_DEVICE_RESOURCE",
)

SPIRV_VULKAN_HARDWARE_PROFILE = loom_execution_profile(
    name = "spirv_vulkan_hardware",
    build_requirements = [
        TARGET_ARCH_SPIRV,
        EMIT_SPIRV,
        EXECUTE_IREE_HAL,
        HAL_VULKAN,
    ],
    executor = "hardware",
    resource_group = "loom-vulkan-tests",
    run_requirements = [VULKAN_DEVICE_RESOURCE],
    runner_args = [
        "--device=vulkan",
        "--vulkan_validation_layers=false",
    ],
    sanitizer_suppressions = vulkan_suppressions,
    target_class = "gpu",
    target_family = "spirv",
)
