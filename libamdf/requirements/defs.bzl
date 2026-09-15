# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""libamdf build and run requirements."""

load("//build_tools/bazel:requirements.bzl", "build_requirement", "run_requirement")

LIBAMDF = build_requirement(
    id = "libamdf",
    label = Label("//libamdf/requirements:libamdf"),
    enabled_by = Label("//libamdf/config:enabled_setting"),
    cmake_condition = "AMDF_BUILD",
)

LIBAMDF_GPU = build_requirement(
    id = "libamdf.gpu",
    label = Label("//libamdf/requirements:gpu"),
    enabled_by = Label("//libamdf/config/family:gpu"),
    cmake_condition = "AMDF_FAMILY_RDNA OR AMDF_FAMILY_CDNA",
)

LIBAMDF_XDNA = build_requirement(
    id = "libamdf.xdna",
    label = Label("//libamdf/requirements:xdna"),
    enabled_by = Label("//libamdf/config/family:xdna"),
    cmake_condition = "AMDF_FAMILY_XDNA",
)

AMDGPU_RESOURCE = run_requirement(
    id = "libamdf.resource.amd_gpu",
    label = Label("//libamdf/requirements:amd_gpu"),
    cmake_label = "runtime-resource=amd-gpu",
    skip_contract = "Tests skip when no qualified native AMD GPU endpoint is available.",
)

XDNA_RESOURCE = run_requirement(
    id = "libamdf.resource.xdna",
    label = Label("//libamdf/requirements:xdna_device"),
    cmake_label = "runtime-resource=amd-xdna",
    skip_contract = "Tests skip when no qualified native XDNA endpoint is available.",
)

REQUIREMENTS = [
    LIBAMDF,
    LIBAMDF_GPU,
    LIBAMDF_XDNA,
    AMDGPU_RESOURCE,
    XDNA_RESOURCE,
]
