# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runtime package policy."""

load(
    "//build_tools/bazel:package_policy.bzl",
    "apply_target_policy",
    "apply_test_policy",
    "collect_package_policy",
    "package_policy",
)
load(
    "//runtime/requirements:defs.bzl",
    "AMDGPU_RESOURCE",
    "HAL_AMDGPU",
    "HAL_CUDA",
    "HAL_HIP",
    "HAL_VULKAN",
    "HAL_WEBGPU",
    "NVIDIA_GPU_RESOURCE",
    "VULKAN_DEVICE_RESOURCE",
    "WEBGPU_DEVICE_RESOURCE",
)

PACKAGE_POLICIES = [
    package_policy(
        packages = ["runtime/src/iree/hal/drivers/amdgpu/..."],
        build_requirements = [HAL_AMDGPU],
    ),
    package_policy(
        packages = ["runtime/src/iree/hal/drivers/amdgpu/..."],
        run_requirements = [AMDGPU_RESOURCE],
        resource_group = "iree-hal-drivers-amdgpu-tests",
    ),
    package_policy(
        packages = ["runtime/src/iree/hal/drivers/cuda/..."],
        build_requirements = [HAL_CUDA],
    ),
    package_policy(
        packages = ["runtime/src/iree/hal/drivers/cuda/cts/..."],
        run_requirements = [NVIDIA_GPU_RESOURCE],
        resource_group = "iree-hal-drivers-cuda-tests",
    ),
    package_policy(
        packages = ["runtime/src/iree/hal/drivers/hip/..."],
        build_requirements = [HAL_HIP],
    ),
    package_policy(
        packages = [
            "runtime/src/iree/hal/drivers/hip",
            "runtime/src/iree/hal/drivers/hip/cts/...",
        ],
        run_requirements = [AMDGPU_RESOURCE],
        resource_group = "iree-hal-drivers-hip-tests",
    ),
    package_policy(
        packages = ["runtime/src/iree/hal/drivers/vulkan/..."],
        build_requirements = [HAL_VULKAN],
    ),
    package_policy(
        packages = ["runtime/src/iree/hal/drivers/vulkan/cts/..."],
        run_requirements = [VULKAN_DEVICE_RESOURCE],
        resource_group = "iree-hal-drivers-vulkan-tests",
    ),
    package_policy(
        packages = ["runtime/src/iree/hal/drivers/webgpu/..."],
        build_requirements = [HAL_WEBGPU],
    ),
    package_policy(
        packages = ["runtime/src/iree/hal/drivers/webgpu/cts/..."],
        run_requirements = [WEBGPU_DEVICE_RESOURCE],
        resource_group = "iree-hal-drivers-webgpu-tests",
    ),
]

def _current_policy():
    return collect_package_policy(native.package_name(), PACKAGE_POLICIES)

def apply_runtime_target_policy(kwargs, name = None):
    return apply_target_policy(kwargs, _current_policy(), name = name)

def apply_runtime_test_policy(kwargs, name = None):
    return apply_test_policy(kwargs, _current_policy(), name = name)
