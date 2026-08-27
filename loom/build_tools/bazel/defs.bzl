# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public Bazel API for Loom source repositories."""

load(
    ":loom_library.bzl",
    _LoomCompilationInfo = "LoomCompilationInfo",
    _LoomCompileTargetInfo = "LoomCompileTargetInfo",
    _LoomExecutionTestInfo = "LoomExecutionTestInfo",
    _LoomLibraryInfo = "LoomLibraryInfo",
    _loom_compile = "loom_compile",
    _loom_compile_target = "loom_compile_target",
    _loom_execution_profile = "loom_execution_profile",
    _loom_kernel_library = "loom_kernel_library",
    _loom_library = "loom_library",
    _loom_test_library = "loom_test_library",
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

LoomCompilationInfo = _LoomCompilationInfo
LoomCompileTargetInfo = _LoomCompileTargetInfo
LoomExecutionTestInfo = _LoomExecutionTestInfo
LoomLibraryInfo = _LoomLibraryInfo
LoomAmdgpuTargetProfileInfo = _LoomAmdgpuTargetProfileInfo
LoomTargetProfileInfo = _LoomTargetProfileInfo
loom_compile = _loom_compile
loom_compile_target = _loom_compile_target
loom_amdgpu_target_profile = _loom_amdgpu_target_profile
loom_execution_profile = _loom_execution_profile
loom_kernel_library = _loom_kernel_library
loom_library = _loom_library
loom_module = _loom_module
loom_test_library = _loom_test_library
loom_tools_toolchains = _loom_tools_toolchains
