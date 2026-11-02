# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Vulkan API compilation and device execution requirements."""

load("//build_tools/bazel:requirements.bzl", "build_requirement", "run_requirement")

VULKAN_API = build_requirement(
    id = "vulkan.api",
    label = Label("//build_tools/vulkan/requirements:api"),
    enabled_by = Label("//build_tools/vulkan/config:available"),
)

VULKAN_DEVICE_RESOURCE = run_requirement(
    id = "vulkan.resource.device",
    label = Label("//build_tools/vulkan/requirements:device"),
    cmake_label = "runtime-resource=vulkan-device",
    skip_contract = "Tests skip when no compatible Vulkan device is available.",
)

REQUIREMENTS = [VULKAN_API, VULKAN_DEVICE_RESOURCE]
