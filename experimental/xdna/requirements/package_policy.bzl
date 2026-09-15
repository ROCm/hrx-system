# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Requirements mirrored by generated CMake targets."""

load("//build_tools/bazel:package_policy.bzl", "package_policy")
load("//libamdf/requirements:defs.bzl", "XDNA_RESOURCE")
load(":defs.bzl", "REQUIREMENTS")

PACKAGE_POLICIES = [
    package_policy(
        packages = ["experimental/xdna/..."],
        build_requirements = REQUIREMENTS,
    ),
    package_policy(
        packages = [
            "experimental/xdna/benchmarks/...",
            "experimental/xdna/cts/...",
        ],
        run_requirements = [XDNA_RESOURCE],
        # Shares native devices with libamdf and GPU/XDNA interop CTS.
        resource_group = "iree-hal-drivers-amdgpu-tests",
    ),
]
