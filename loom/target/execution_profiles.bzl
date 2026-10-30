# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native hardware policies for shared GPU source tests."""

load(
    "//loom/target/amdgpu:execution_profiles.bzl",
    "AMD_GPU_HARDWARE_PROFILE",
)
load(
    "//loom/target/spirv:execution_profiles.bzl",
    "SPIRV_VULKAN_HARDWARE_PROFILE",
)

# Each child executes on a compatible local device; offline compiler profiles
# separately qualify code generation for explicit architecture selectors.
GPU_HARDWARE_PROFILES = [
    AMD_GPU_HARDWARE_PROFILE,
    SPIRV_VULKAN_HARDWARE_PROFILE,
]
