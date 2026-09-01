# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public Bazel API for Loom source repositories."""

load(
    ":loom_binary.bzl",
    _LoomBinaryInfo = "LoomBinaryInfo",
    _loom_command_binary = "loom_command_binary",
    _loom_kernel_binary = "loom_kernel_binary",
)
load(
    ":loom_library.bzl",
    _LoomExecutionTestInfo = "LoomExecutionTestInfo",
    _LoomLibraryInfo = "LoomLibraryInfo",
    _loom_execution_profile = "loom_execution_profile",
    _loom_kernel_library = "loom_kernel_library",
    _loom_library = "loom_library",
    _loom_test = "loom_test",
    _loom_test_suite = "loom_test_suite",
)
load(
    ":loom_module.bzl",
    _loom_module = "loom_module",
)
load(
    ":loom_target_profile.bzl",
    _LoomAmdgpuTargetProfileInfo = "LoomAmdgpuTargetProfileInfo",
    _LoomTargetProfileInfo = "LoomTargetProfileInfo",
    _loom_amdgpu_target_profile = "loom_amdgpu_target_profile",
)
load(
    ":loom_toolchain.bzl",
    _loom_tools_toolchains = "loom_tools_toolchains",
)

LoomBinaryInfo = _LoomBinaryInfo
LoomExecutionTestInfo = _LoomExecutionTestInfo
LoomLibraryInfo = _LoomLibraryInfo
LoomAmdgpuTargetProfileInfo = _LoomAmdgpuTargetProfileInfo
LoomTargetProfileInfo = _LoomTargetProfileInfo
loom_amdgpu_target_profile = _loom_amdgpu_target_profile
loom_command_binary = _loom_command_binary
loom_execution_profile = _loom_execution_profile
loom_kernel_binary = _loom_kernel_binary
loom_kernel_library = _loom_kernel_library
loom_library = _loom_library
loom_module = _loom_module
loom_test = _loom_test
loom_test_suite = _loom_test_suite
loom_tools_toolchains = _loom_tools_toolchains
