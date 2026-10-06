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

HAL_EXECUTABLE_LOADER_EMBEDDED_ELF = build_requirement(
    id = "runtime.hal.executable_loader.embedded_elf",
    label = Label("//runtime/requirements:hal_executable_loader_embedded_elf"),
    enabled_by = Label("//runtime/config/hal:executable_loader_embedded_elf"),
    cmake_condition = "IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF",
)

HAL_TASK = build_requirement(
    id = "runtime.hal.task",
    label = Label("//runtime/requirements:hal_task"),
    enabled_by = Label("//runtime/config/hal:driver_task"),
    cmake_condition = "IREE_HAL_DRIVER_TASK",
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

HAL_XDNA = build_requirement(
    id = "runtime.hal.xdna",
    label = Label("//runtime/requirements:hal_xdna"),
    enabled_by = Label("//runtime/config/hal:driver_xdna"),
    cmake_condition = "IREE_HAL_DRIVER_XDNA",
)

AMDGPU_RESOURCE = run_requirement(
    id = "runtime.resource.amd_gpu",
    label = Label("//runtime/requirements:amd_gpu"),
    cmake_label = "runtime-resource=amd-gpu",
    skip_contract = "Tests skip when no compatible AMD GPU/HSA agent is available.",
)

WEBGPU_DEVICE_RESOURCE = run_requirement(
    id = "runtime.resource.webgpu_device",
    label = Label("//runtime/requirements:webgpu_device"),
    cmake_label = "runtime-resource=webgpu-device",
    skip_contract = "Tests skip when no compatible WebGPU/Dawn device is available.",
)

REQUIREMENTS = [
    HAL_AMDGPU,
    HAL_EXECUTABLE_LOADER_EMBEDDED_ELF,
    HAL_TASK,
    HAL_VULKAN,
    HAL_WEBGPU,
    HAL_XDNA,
    AMDGPU_RESOURCE,
    WEBGPU_DEVICE_RESOURCE,
]
