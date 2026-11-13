# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared target and resource policy for the HRX CTS."""

load("//build_tools/bazel:test_resources.bzl", "GPU_DEVICE_RESOURCE_GROUP")

AMD_GPU_CTS_RESOURCE_GROUP = GPU_DEVICE_RESOURCE_GROUP

AMD_GPU_CTS_TAGS = [
    "iree-run-requirement=libamdf.resource.amd_gpu",
    "runtime-resource=amd-gpu",
]

# GPU consumer tests use the native AMD GPU backend and its device toolchain.
AMDGPU_CTS_COMPATIBILITY_CONDITIONS = {
    "//runtime/config/hal:driver_amd_gpu": [],
    "//conditions:default": ["@platforms//:incompatible"],
}
