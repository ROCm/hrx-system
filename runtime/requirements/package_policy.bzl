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
load("//build_tools/bazel:test_resources.bzl", "GPU_DEVICE_RESOURCE_GROUP")
load(
    "//build_tools/vulkan/requirements:defs.bzl",
    "VULKAN_DEVICE_RESOURCE",
)
load("//libamdf/requirements:defs.bzl", "LIBAMDF", "LIBAMDF_XDNA", "XDNA_RESOURCE")
load(
    "//runtime/requirements:defs.bzl",
    "AMDGPU_RESOURCE",
    "HAL_AMDGPU",
    "HAL_VULKAN",
    "HAL_WEBGPU",
    "HAL_XDNA",
    "WEBGPU_DEVICE_RESOURCE",
)

PACKAGE_POLICIES = [
    package_policy(
        packages = ["runtime/src/iree/hal/drivers/amd/xdna/registration"],
        build_requirements = [HAL_XDNA, LIBAMDF, LIBAMDF_XDNA],
    ),
    package_policy(
        packages = ["runtime/src/iree/hal/drivers/amd/xdna/cts"],
        build_requirements = [LIBAMDF, LIBAMDF_XDNA],
        run_requirements = [XDNA_RESOURCE],
        resource_group = GPU_DEVICE_RESOURCE_GROUP,
    ),
    package_policy(
        packages = [
            "runtime/src/iree/hal/drivers/amd",
            "runtime/src/iree/hal/drivers/amd/xdna",
        ],
        build_requirements = [LIBAMDF],
    ),
    package_policy(
        packages = ["runtime/src/iree/hal/drivers/amdgpu/..."],
        excluded_packages = [
            "runtime/src/iree/hal/drivers/amdgpu/abi",
        ],
        build_requirements = [HAL_AMDGPU],
    ),
    package_policy(
        packages = ["runtime/src/iree/hal/drivers/amdgpu/..."],
        excluded_packages = [
            "runtime/src/iree/hal/drivers/amdgpu/abi",
            "runtime/src/iree/hal/drivers/amdgpu/target/...",
        ],
        run_requirements = [AMDGPU_RESOURCE],
        resource_group = GPU_DEVICE_RESOURCE_GROUP,
    ),
    package_policy(
        packages = ["runtime/src/iree/hal/drivers/vulkan/..."],
        build_requirements = [HAL_VULKAN],
    ),
    package_policy(
        packages = ["runtime/src/iree/hal/drivers/vulkan/cts/..."],
        run_requirements = [VULKAN_DEVICE_RESOURCE],
        resource_group = GPU_DEVICE_RESOURCE_GROUP,
    ),
    package_policy(
        packages = ["runtime/src/iree/hal/drivers/webgpu/..."],
        build_requirements = [HAL_WEBGPU],
    ),
    package_policy(
        packages = ["runtime/src/iree/hal/drivers/webgpu/cts/..."],
        run_requirements = [WEBGPU_DEVICE_RESOURCE],
        resource_group = GPU_DEVICE_RESOURCE_GROUP,
    ),
]

def _current_policy():
    return collect_package_policy(native.package_name(), PACKAGE_POLICIES)

def apply_runtime_target_policy(kwargs, name = None):
    return apply_target_policy(kwargs, _current_policy(), name = name)

def apply_runtime_test_policy(kwargs, name = None):
    return apply_test_policy(kwargs, _current_policy(), name = name)
