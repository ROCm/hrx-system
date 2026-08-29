# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runtime build and run requirements."""

load(
    "//build_tools/bazel:requirements.bzl",
    "build_requirement",
    "run_requirement",
)

HAL_AMDGPU = build_requirement(
    id = "runtime.hal.amdgpu",
    label = Label("//runtime/requirements:hal_amdgpu"),
    enabled_by = Label("//runtime/config/hal:driver_amdgpu"),
    cmake_condition = "IREE_HAL_DRIVER_AMDGPU",
)

HAL_VULKAN = build_requirement(
    id = "runtime.hal.vulkan",
    label = Label("//runtime/requirements:hal_vulkan"),
    enabled_by = Label("//runtime/config/hal:driver_vulkan"),
    cmake_condition = "IREE_HAL_DRIVER_VULKAN",
)

HAL_WEBGPU = build_requirement(
    id = "runtime.hal.webgpu",
    label = Label("//runtime/requirements:hal_webgpu"),
    enabled_by = Label("//runtime/config/hal:driver_webgpu"),
    cmake_condition = "IREE_HAL_DRIVER_WEBGPU",
)

AMDGPU_RESOURCE = run_requirement(
    id = "runtime.resource.amd_gpu",
    label = Label("//runtime/requirements:amd_gpu"),
    cmake_label = "runtime-resource=amd-gpu",
    skip_contract = "Tests skip when no compatible AMD GPU/HSA agent is available.",
)

VULKAN_DEVICE_RESOURCE = run_requirement(
    id = "runtime.resource.vulkan_device",
    label = Label("//runtime/requirements:vulkan_device"),
    cmake_label = "runtime-resource=vulkan-device",
    skip_contract = "Tests skip when no compatible Vulkan device is available.",
)

WEBGPU_DEVICE_RESOURCE = run_requirement(
    id = "runtime.resource.webgpu_device",
    label = Label("//runtime/requirements:webgpu_device"),
    cmake_label = "runtime-resource=webgpu-device",
    skip_contract = "Tests skip when no compatible WebGPU/Dawn device is available.",
)

REQUIREMENTS = [
    HAL_AMDGPU,
    HAL_VULKAN,
    HAL_WEBGPU,
    AMDGPU_RESOURCE,
    VULKAN_DEVICE_RESOURCE,
    WEBGPU_DEVICE_RESOURCE,
]
