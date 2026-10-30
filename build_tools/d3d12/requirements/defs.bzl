# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""D3D12 API compilation and device execution requirements."""

load("//build_tools/bazel:requirements.bzl", "build_requirement", "run_requirement")

D3D12_API = build_requirement(
    id = "d3d12.api",
    label = Label("//build_tools/d3d12/requirements:api"),
    enabled_by = Label("//build_tools/d3d12/config:available"),
)

D3D12_DEVICE_RESOURCE = run_requirement(
    id = "d3d12.resource.device",
    label = Label("//build_tools/d3d12/requirements:device"),
    cmake_label = "runtime-resource=d3d12-device",
    skip_contract = "Tests skip when no compatible D3D12 device is available.",
)

REQUIREMENTS = [D3D12_API, D3D12_DEVICE_RESOURCE]
